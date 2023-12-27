import copy
import os
import time

import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common import (get_camera_from_tensor, get_samples, get_tensor_from_camera)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Tracker(object):
    def __init__(self, cfg, args, slam):
        self.cfg = cfg
        self.args = args

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.idx = slam.idx
        self.nice = slam.nice
        self.bound = slam.bound
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.shared_c = slam.shared_c
        self.renderer = slam.renderer
        self.gt_c2w_list = slam.gt_c2w_list
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.shared_decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list

        self.cam_lr = cfg['tracking']['lr']
        self.device = cfg['tracking']['device']
        self.num_cam_iters = cfg['tracking']['iters']
        self.gt_camera = cfg['tracking']['gt_camera']
        self.tracking_pixels = cfg['tracking']['pixels']
        self.seperate_LR = cfg['tracking']['seperate_LR']
        self.w_color_loss = cfg['tracking']['w_color_loss']
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.handle_dynamic = cfg['tracking']['handle_dynamic']
        self.use_color_in_tracking = cfg['tracking']['use_color_in_tracking']
        self.const_speed_assumption = cfg['tracking']['const_speed_assumption']

        self.every_frame = cfg['mapping']['every_frame']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']

        self.prev_mapping_idx = -1
        self.frame_reader = get_dataset(
            cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, shuffle=False, num_workers=1)
        self.visualizer = Visualizer(freq=cfg['tracking']['vis_freq'], inside_freq=cfg['tracking']['vis_inside_freq'],
                                     vis_dir=os.path.join(self.output, 'vis' if 'Demo' in self.output else 'tracking_vis'),
                                     renderer=self.renderer, verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    # 相机位姿迭代优化
    def optimize_cam_in_batch(self, camera_tensor, gt_color, gt_depth, batch_size, optimizer):
        """
        Do one iteration of camera iteration. Sample pixels, render depth/color, calculate loss and backpropagation.

        Args:
            camera_tensor (tensor): camera tensor.
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            batch_size (int): batch size, number of sampling rays.
            optimizer (torch.optim): camera optimizer.

        Returns:
            loss (float): The value of loss.
        """
        device = self.device    # 设备
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy# 获取内外参
        optimizer.zero_grad()   # 清零梯度
        c2w = get_camera_from_tensor(camera_tensor) # 返回值是 RT， R 的表示是旋转矩阵
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H
        batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(# 获取要采样的像素
            Hedge, H-Hedge, Wedge, W-Wedge, batch_size, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
        if self.nice:
            # should pre-filter those out of bounding box depth value
            # 过滤调不在 bound 边界内的深度值，以确保处理的光线在场景边界内
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                t = (self.bound.unsqueeze(0).to(device)-det_rays_o)/det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

        depth, uncertainty, color = self.renderer.render_batch_ray(# Render color, depth and uncertainty of a batch of rays.
            self.c, self.decoders, batch_rays_d, batch_rays_o,  self.device, stage='color',  gt_depth=batch_gt_depth)

        uncertainty = uncertainty.detach()
        if self.handle_dynamic:# 启用动态处理的情况
            tmp = torch.abs(batch_gt_depth-depth)/torch.sqrt(uncertainty+1e-10)
            mask = (tmp < 10*tmp.median()) & (batch_gt_depth > 0)
        else:
            mask = batch_gt_depth > 0

        # 计算 loss （基于深度） 误差绝对值/方差                                👇 确保被除数不为0
        loss = (torch.abs(batch_gt_depth-depth) / torch.sqrt(uncertainty+1e-10))[mask].sum()

        if self.use_color_in_tracking:# 启用颜色跟踪的情况
            color_loss = torch.abs(batch_gt_color - color)[mask].sum()
            loss += self.w_color_loss*color_loss # 加入 loss

        loss.backward() # 触发反向传播，计算梯度
        optimizer.step()
        optimizer.zero_grad()   # 清空梯度
        return loss.item()      # 返回 loss

    def update_para_from_mapping(self):
        """
        Update the parameters of scene representation from the mapping thread.

        """
        if self.mapping_idx[0] != self.prev_mapping_idx:# 判断当前索引和前一帧索引是否一致，不一致说明更新帧了
            if self.verbose:
                print('Tracking: update the parameters from mapping')
            self.decoders = copy.deepcopy(self.shared_decoders).to(self.device)# 更新解码器（网络）
            for key, val in self.shared_c.items():  # 遍历特征网格的列表
                val = val.clone().to(self.device)
                self.c[key] = val
            self.prev_mapping_idx = self.mapping_idx[0].clone()

    def run(self):
        device = self.device
        self.c = {}
        if self.verbose:
            pbar = self.frame_loader
        else:
            pbar = tqdm(self.frame_loader)

        for idx, gt_color, gt_depth, gt_c2w in pbar:
            if not self.verbose:
                pbar.set_description(f"Tracking Frame {idx[0]}")

            idx = idx[0]
            gt_depth = gt_depth[0]
            gt_color = gt_color[0]
            gt_c2w = gt_c2w[0]

            # 同步策略：严格/
            if self.sync_method == 'strict':
                # strictly mapping and then tracking
                # initiate mapping every self.every_frame frames
                if idx > 0 and (idx % self.every_frame == 1 or self.every_frame == 1):
                    while self.mapping_idx[0] != idx-1:
                        time.sleep(0.1)
                    pre_c2w = self.estimate_c2w_list[idx-1].to(device)# 更新位姿
            elif self.sync_method == 'loose':
                # mapping idx can be later than tracking idx is within the bound of
                # [-self.every_frame-self.every_frame//2, -self.every_frame+self.every_frame//2]
                while self.mapping_idx[0] < idx-self.every_frame-self.every_frame//2:
                    time.sleep(0.1)
            elif self.sync_method == 'free':
                # pure parallel, if mesh/vis happens may cause inbalance
                pass

            # 从 mapping 更新参数
            self.update_para_from_mapping()

            if self.verbose:
                print(Fore.MAGENTA)
                print("Tracking Frame ",  idx.item())
                print(Style.RESET_ALL)

            if idx == 0 or self.gt_camera:# 索引为0
                c2w = gt_c2w
                if not self.no_vis_on_first_frame:
                    self.visualizer.vis(idx, 0, gt_depth, gt_color, c2w, self.c, self.decoders)

            else:# 估计当前帧先验（位姿）
                gt_camera_tensor = get_tensor_from_camera(gt_c2w)

                # 估计新一帧相机的位姿
                if self.const_speed_assumption and idx-2 >= 0:# 恒定速度假设
                    pre_c2w = pre_c2w.float()
                    delta = pre_c2w@self.estimate_c2w_list[idx-2].to(device).float().inverse()
                    estimated_new_cam_c2w = delta@pre_c2w
                else:
                    estimated_new_cam_c2w = pre_c2w

                # 创建相机位姿优化器
                camera_tensor = get_tensor_from_camera(estimated_new_cam_c2w.detach())
                if self.seperate_LR:# 分离 旋转和平移的学习率
                    camera_tensor = camera_tensor.to(device).detach()
                    T = camera_tensor[-3:]      # 平移 后三列
                    quad = camera_tensor[:4]    # 旋转 前四列
                    cam_para_list_quad = [quad] # 将其变成列表
                    quad = Variable(quad, requires_grad=True)   # 改变其数据类型
                    T = Variable(T, requires_grad=True)         # 改变其数据类型
                    camera_tensor = torch.cat([quad, T], 0) # 重新将 quad 和 T 拼接在一起
                    cam_para_list_T = [T]       # 保存为列表
                    cam_para_list_quad = [quad] # 保存为列表
                    optimizer_camera = torch.optim.Adam([{'params': cam_para_list_T, 'lr': self.cam_lr},# 创建优化器
                                                         {'params': cam_para_list_quad, 'lr': self.cam_lr*0.2}])
                else:
                    camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                    cam_para_list = [camera_tensor]
                    optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)# 创建优化器

                # 在 for 循环中要用到的变量
                initial_loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()# 平均绝对误差
                candidate_cam_tensor = None# 最优相机位姿
                current_min_loss = 10000000000.# 最小 loss
                for cam_iter in range(self.num_cam_iters):
                    # 独立学习率
                    if self.seperate_LR:
                        camera_tensor = torch.cat([quad, T], 0).to(self.device)

                    # 可视化
                    self.visualizer.vis(idx, cam_iter, gt_depth, gt_color, camera_tensor, self.c, self.decoders)

                    # 批量优化相机位姿，得到 loss
                    loss = self.optimize_cam_in_batch(
                        camera_tensor, gt_color, gt_depth, self.tracking_pixels, optimizer_camera)

                    if cam_iter == 0:
                        initial_loss = loss

                    loss_camera_tensor = torch.abs(gt_camera_tensor.to(device)-camera_tensor).mean().item()
                    if self.verbose:
                        if cam_iter == self.num_cam_iters-1:
                            print(
                                f'Re-rendering loss: {initial_loss:.2f}->{loss:.2f} ' +
                                f'camera tensor error: {initial_loss_camera_tensor:.4f}->{loss_camera_tensor:.4f}')
                    if loss < current_min_loss:
                        current_min_loss = loss# 更新当前最小loss
                        candidate_cam_tensor = camera_tensor.clone().detach()
                bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(self.device)
                # 更新最优相机姿态
                c2w = get_camera_from_tensor(candidate_cam_tensor.clone().detach())
                c2w = torch.cat([c2w, bottom], dim=0)
            ################# 保存计算出的相机位姿 ###################
            self.estimate_c2w_list[idx] = c2w.clone().cpu() # 保存 c2w 值到 list 中 ！！
            self.gt_c2w_list[idx] = gt_c2w.clone().cpu()    # 保存 gt_c2w
            pre_c2w = c2w.clone()   # 设置 pre_c2w 值，用于（用速度不变假设）更新位姿
            self.idx[0] = idx
            if self.low_gpu_mem:
                torch.cuda.empty_cache()
