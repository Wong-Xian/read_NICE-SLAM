import os
import time

import cv2
import numpy as np
import torch
from colorama import Fore, Style
from torch.autograd import Variable

from src.common import (get_camera_from_tensor, get_samples,
                        get_tensor_from_camera, random_select)
from src.utils.datasets import get_dataset
from src.utils.Visualizer import Visualizer


class Mapper(object):
    """
    Mapper thread. Note that coarse mapper also uses this code.

    """

    def __init__(self, cfg, args, slam, coarse_mapper=False):

        self.cfg = cfg
        self.args = args
        self.coarse_mapper = coarse_mapper

        self.idx = slam.idx
        self.nice = slam.nice
        self.c = slam.shared_c
        self.bound = slam.bound
        self.logger = slam.logger
        self.mesher = slam.mesher
        self.output = slam.output
        self.verbose = slam.verbose
        self.renderer = slam.renderer
        self.low_gpu_mem = slam.low_gpu_mem
        self.mapping_idx = slam.mapping_idx
        self.mapping_cnt = slam.mapping_cnt
        self.decoders = slam.shared_decoders
        self.estimate_c2w_list = slam.estimate_c2w_list
        self.mapping_first_frame = slam.mapping_first_frame

        self.scale = cfg['scale']
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.sync_method = cfg['sync_method']

        self.device = cfg['mapping']['device']
        self.fix_fine = cfg['mapping']['fix_fine']
        self.eval_rec = cfg['meshing']['eval_rec']
        self.BA = False  # Even if BA is enabled, it starts only when there are at least 4 keyframes
        self.BA_cam_lr = cfg['mapping']['BA_cam_lr']
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.fix_color = cfg['mapping']['fix_color']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.num_joint_iters = cfg['mapping']['iters']
        self.clean_mesh = cfg['meshing']['clean_mesh']
        self.every_frame = cfg['mapping']['every_frame']
        self.color_refine = cfg['mapping']['color_refine']
        self.w_color_loss = cfg['mapping']['w_color_loss']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.fine_iter_ratio = cfg['mapping']['fine_iter_ratio']
        self.middle_iter_ratio = cfg['mapping']['middle_iter_ratio']
        self.mesh_coarse_level = cfg['meshing']['mesh_coarse_level']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.frustum_feature_selection = cfg['mapping']['frustum_feature_selection']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.save_selected_keyframes_info = cfg['mapping']['save_selected_keyframes_info']
        if self.save_selected_keyframes_info:
            self.selected_keyframes = {}

        if self.nice:
            if coarse_mapper:
                self.keyframe_selection_method = 'global'

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)  # 获取数据集 数据集类实例化的对象
        self.n_img = len(self.frame_reader)     # 获取数据集中的图像数量
        if 'Demo' not in self.output:
            # 初始化可视化对象
            self.visualizer = Visualizer(freq=cfg['mapping']['vis_freq'], inside_freq=cfg['mapping']['vis_inside_freq'],
                                         vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                         verbose=self.verbose, device=self.device)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy

    '''
    根据当前相机的位姿和深度图选择需要被优化的 grid ，输出与 feature grids 空间尺寸相同的 0-1 mask
        1、根据 grid 的尺寸创建一个对应尺寸的三维点云，若是 coarse 层，直接返回全为 1 的mask
        2、根据相机内外参，将点云投影到平面图像，保存点云在图像平面的坐标
        3、用 cv2.remap() 函数从深度图中取对应深度
        4、根据图像的长宽范围、深度范围得到 mask ， reshape 到与 grid 相同的尺寸。
    '''
    def get_mask_from_c2w(self, c2w, key, val_shape, depth_np):
        """
        Frustum feature selection based on current camera pose and depth image.

        Args:
            c2w (tensor): camera pose of current frame.
            key (str): name of this feature grid.
            val_shape (tensor): shape of the grid.
            depth_np (numpy.array): depth image of current frame.

        Returns:
            mask (tensor): mask for selected optimizable feature.
            points (tensor): corresponding point coordinates.
        """
        H, W, fx, fy, cx, cy, = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        
        # 1、根据 grid 的尺寸创建一个对应尺寸的三维点云，若是 coarse 层，直接返回全为 1 的mask
        X, Y, Z = torch.meshgrid(torch.linspace(self.bound[0][0], self.bound[0][1], val_shape[2]),
                                 torch.linspace(self.bound[1][0], self.bound[1][1], val_shape[1]),
                                 torch.linspace(self.bound[2][0], self.bound[2][1], val_shape[0]))

        # 2、根据相机内外参，将点云投影到平面图像，保存点云在图像平面的坐标
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)
        if key == 'grid_coarse':
            mask = np.ones(val_shape[::-1]).astype(np.bool)
            return mask
        points_bak = points.clone()
        c2w = c2w.cpu().numpy()
        w2c = np.linalg.inv(c2w)
        ones = np.ones_like(points[:, 0]).reshape(-1, 1)
        homo_vertices = np.concatenate(
            [points, ones], axis=1).reshape(-1, 4, 1)
        cam_cord_homo = w2c@homo_vertices
        cam_cord = cam_cord_homo[:, :3]
        K = np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)
        cam_cord[:, 0] *= -1
        uv = K@cam_cord
        z = uv[:, -1:]+1e-5
        uv = uv[:, :2]/z
        uv = uv.astype(np.float32)

        # 3、用 cv2.remap() 函数从深度图中取对应深度
        remap_chunk = int(3e4)
        depths = []
        for i in range(0, uv.shape[0], remap_chunk):
            depths += [cv2.remap(depth_np,
                                 uv[i:i+remap_chunk, 0],
                                 uv[i:i+remap_chunk, 1],
                                 interpolation=cv2.INTER_LINEAR)[:, 0].reshape(-1, 1)]
        depths = np.concatenate(depths, axis=0)
        edge = 0
        
        # 4、根据图像的长宽范围、深度范围得到 mask ， reshape 到与 grid 相同的尺寸。
        mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
            (uv[:, 1] < H-edge)*(uv[:, 1] > edge)

        # For ray with depth==0, fill it with maximum depth
        zero_mask = (depths == 0)
        depths[zero_mask] = np.max(depths)

        # depth test
        mask = mask & (0 <= -z[:, :, 0]) & (-z[:, :, 0] <= depths+0.5)
        mask = mask.reshape(-1)

        # add feature grid near cam center
        ray_o = c2w[:3, 3]
        ray_o = torch.from_numpy(ray_o).unsqueeze(0)

        dist = points_bak-ray_o
        dist = torch.sum(dist*dist, axis=1)
        mask2 = dist < 0.5*0.5
        mask2 = mask2.cpu().numpy()
        mask = mask | mask2

        points = points[mask]
        mask = mask.reshape(val_shape[2], val_shape[1], val_shape[0])
        return mask

    '''
    计算 overlap 
        1、在当前帧采样 100 条 ray ，每条 ray 采样 16 个点，投影到每个历史关键帧中
        2、计算落入历史帧视野范围内的点的比例，作为 overlap 的依据
        3、将完全无 overlap 的关键帧去除，剩下的关键帧随机选取 k-2 个
    '''
    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, keyframe_dict, k, N_samples=16, pixels=100):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color (tensor): ground truth color image of the current frame.
            gt_depth (tensor): ground truth depth image of the current frame.
            c2w (tensor): camera to world matrix (3*4 or 4*4 both fine).
            keyframe_dict (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            N_samples (int, optional): number of samples/points per ray. Defaults to 16.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 100.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy

        # 1、在当前帧采样 100 条 ray ，每条 ray 采样 16 个点，投影到每个历史关键帧中
        rays_o, rays_d, gt_depth, gt_color = get_samples(
            0, H, 0, W, pixels, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)

        gt_depth = gt_depth.reshape(-1, 1)
        gt_depth = gt_depth.repeat(1, N_samples)
        t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
        near = gt_depth*0.8
        far = gt_depth+0.5
        z_vals = near * (1.-t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # p = o + r(方向) * di
        vertices = pts.reshape(-1, 3).cpu().numpy()
        list_keyframe = []

        # 2、计算落入历史帧视野范围内的点的比例，作为 overlap 的依据
        for keyframeid, keyframe in enumerate(keyframe_dict):
            c2w = keyframe['est_c2w'].cpu().numpy()
            w2c = np.linalg.inv(c2w)
            ones = np.ones_like(vertices[:, 0]).reshape(-1, 1)
            homo_vertices = np.concatenate(
                [vertices, ones], axis=1).reshape(-1, 4, 1)  # (N, 4)
            cam_cord_homo = w2c@homo_vertices  # (N, 4, 1)=(4,4)*(N, 4, 1)
            cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
            K = np.array([[fx, .0, cx], [.0, fy, cy],
                         [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1
            uv = K@cam_cord
            z = uv[:, -1:]+1e-5
            uv = uv[:, :2]/z
            uv = uv.astype(np.float32)
            edge = 20
            mask = (uv[:, 0] < W-edge)*(uv[:, 0] > edge) * \
                (uv[:, 1] < H-edge)*(uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)
            percent_inside = mask.sum()/uv.shape[0]
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # 3、将完全无 overlap 的关键帧去除，剩下的关键帧随机选取 k-2 个
        list_keyframe = sorted(# 排序
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        selected_keyframe_list = [dic['id']
                                  for dic in list_keyframe if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])
        return selected_keyframe_list   # 返回选取的关键帧

    def optimize_map(self, num_joint_iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict, keyframe_list, cur_c2w):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if local BA enabled).
        从关键帧提取像素，优化场景表达和相机位姿
        Args:
            num_joint_iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): list of keyframes info dictionary.
            keyframe_list (list): list ofkeyframe index.
            cur_c2w (tensor): the estimated camera to world matrix of current frame. 

        Returns:
            cur_c2w/None (tensor/None): return the updated cur_c2w, return None if no BA
        """
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        c = self.c
        cfg = self.cfg
        device = self.device
        bottom = torch.from_numpy(np.array(
            [0, 0, 0, 1.]).reshape([1, 4])).type(torch.float32).to(device)

        '''
        对关键帧的处理
            1、选取关键帧
            2、选择待优化的特征网格
            3、构建优化器
            4、循环优化位姿和地图
        '''

        # 1、选关键帧
        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                num = self.mapping_window_size-2
                optimize_frame = random_select(len(self.keyframe_dict)-1, num)
            elif self.keyframe_selection_method == 'overlap':
                num = self.mapping_window_size-2
                optimize_frame = self.keyframe_selection_overlap(
                    cur_gt_color, cur_gt_depth, cur_c2w, keyframe_dict[:-1], num)

        # add the last keyframe and the current frame(use -1 to denote)
        oldest_frame = None
        if len(keyframe_list) > 0:
            optimize_frame = optimize_frame + [len(keyframe_list)-1]
            oldest_frame = min(optimize_frame)
        optimize_frame += [-1]  # 把-1加到原列表的末尾

        if self.save_selected_keyframes_info:
            keyframes_info = []
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    frame_idx = keyframe_list[frame]
                    tmp_gt_c2w = keyframe_dict[frame]['gt_c2w']
                    tmp_est_c2w = keyframe_dict[frame]['est_c2w']
                else:
                    frame_idx = idx
                    tmp_gt_c2w = gt_cur_c2w
                    tmp_est_c2w = cur_c2w
                keyframes_info.append(
                    {'idx': frame_idx, 'gt_c2w': tmp_gt_c2w, 'est_c2w': tmp_est_c2w})
            self.selected_keyframes[idx] = keyframes_info

        pixs_per_image = self.mapping_pixels//len(optimize_frame)

        decoders_para_list = []
        
        # 2、选择待优化的特征网格
        coarse_grid_para = []
        middle_grid_para = []
        fine_grid_para = []
        color_grid_para = []
        
        gt_depth_np = cur_gt_depth.cpu().numpy()
        if self.nice:
            if self.frustum_feature_selection:
                masked_c_grad = {}
                mask_c2w = cur_c2w
            for key, val in c.items():  # 遍历特征网格中的每个元素
                if not self.frustum_feature_selection:# 没启用视锥特征选择
                    val = Variable(val.to(device), requires_grad=True)
                    c[key] = val
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val)
                    elif key == 'grid_color':
                        color_grid_para.append(val)

                else:# 启用视锥特征选择
                    mask = self.get_mask_from_c2w(mask_c2w, key, val.shape[2:], gt_depth_np)
                    # 下一行代码 将一个NumPy数组转换为PyTorch张量，并将其维度重新排列以适应特定的数据格式，然后在张量的第一个和第二个维度上添加新维度，并在张量的第三个和第四个维度上重复元素
                    mask = torch.from_numpy(mask).permute(2, 1, 0).unsqueeze(0).unsqueeze(0).repeat(1, val.shape[1], 1, 1, 1)
                    val = val.to(device)
                    # val_grad is the optimizable part, other parameters will be fixed
                    val_grad = val[mask].clone()
                    val_grad = Variable(val_grad.to(device), requires_grad=True)
                    masked_c_grad[key] = val_grad
                    masked_c_grad[key+'mask'] = mask
                    if key == 'grid_coarse':
                        coarse_grid_para.append(val_grad)
                    elif key == 'grid_middle':
                        middle_grid_para.append(val_grad)
                    elif key == 'grid_fine':
                        fine_grid_para.append(val_grad)
                    elif key == 'grid_color':
                        color_grid_para.append(val_grad)

        if self.nice:
            if not self.fix_fine:
                decoders_para_list += list(
                    self.decoders.fine_decoder.parameters())
            if not self.fix_color:
                decoders_para_list += list(
                    self.decoders.color_decoder.parameters())
        else:   # iMAP 的情况
            decoders_para_list += list(self.decoders.parameters())

        if self.BA:
            camera_tensor_list = []
            gt_camera_tensor_list = []
            for frame in optimize_frame:# 遍历要优化的帧
                # the oldest frame should be fixed to avoid drifting
                if frame != oldest_frame:   # 保持最旧帧不变，避免漂移
                    if frame != -1:# 不是当前帧
                        c2w = keyframe_dict[frame]['est_c2w']# 获取估计的相机姿态
                        gt_c2w = keyframe_dict[frame]['gt_c2w']# 获取gt相机姿态
                    else:# 是当前帧
                        c2w = cur_c2w# 使用当前的估计相机姿态
                        gt_c2w = gt_cur_c2w # 使用 gt 相机姿态
                    camera_tensor = get_tensor_from_camera(c2w)
                    camera_tensor = Variable(camera_tensor.to(device), requires_grad=True)
                    camera_tensor_list.append(camera_tensor)
                    gt_camera_tensor = get_tensor_from_camera(gt_c2w)
                    gt_camera_tensor_list.append(gt_camera_tensor)

        # 3、构建优化器
        if self.nice:
            if self.BA: # BA 是 true 的情况（默认是 false）
                # The corresponding lr will be set according to which stage the optimization is in
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])# 有 BA 一同优化相机位姿
            else:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': coarse_grid_para, 'lr': 0},
                                              {'params': middle_grid_para, 'lr': 0},
                                              {'params': fine_grid_para, 'lr': 0},
                                              {'params': color_grid_para, 'lr': 0}])
        else:   # imap*, single MLP
            if self.BA:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                              {'params': camera_tensor_list, 'lr': 0}])
            else:
                optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0}])
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=200, gamma=0.8)


        # 4、在 for 循环中，对地图优化若干次，具体步骤如下
            # 4.1 根据 mapper 的不同设置 stage，对于 coarse mapper， stage 为 ‘coarse’
                                            #对于 fine mapper， 先 middle stage，再 fine stage
                    # 每个 stage 的 iter 数量根据 middle_iter_ratio 和 fine_iter_ratio 设置
                    # 各个 stage 中优化变量的学习率在 configs/nice-slam.yaml 中设置
            # 4.2 遍历 optimize_frame 中的每个关键帧，调用 get_samples() 函数根据相机位姿采样一定数量
                # 的 ray，加入 list 中，总计采样 5000 条
            # 4.3 调用 render_batch_ray() 函数，得到 render 后的 rgb 和 depth
            # 4.4 render 后的 rgb 和 depth 和真值计算 loss ，并反向传播
            # 4.5 将优化后的 grid 更新到全局 feature_grid
        for joint_iter in range(num_joint_iters):
            if self.nice:
                if self.frustum_feature_selection:
                    for key, val in c.items():
                        if (self.coarse_mapper and 'coarse' in key) or \
                                ((not self.coarse_mapper) and ('coarse' not in key)):
                            val_grad = masked_c_grad[key]
                            mask = masked_c_grad[key+'mask']
                            val = val.to(device)
                            val[mask] = val_grad
                            c[key] = val

                # 4.1
                if self.coarse_mapper:
                    self.stage = 'coarse'
                elif joint_iter <= int(num_joint_iters*self.middle_iter_ratio):
                    self.stage = 'middle'
                elif joint_iter <= int(num_joint_iters*self.fine_iter_ratio):
                    self.stage = 'fine'
                else:
                    self.stage = 'color'

                # 优化器中不同的学习率
                optimizer.param_groups[0]['lr'] = cfg['mapping']['stage'][self.stage]['decoders_lr']*lr_factor
                optimizer.param_groups[1]['lr'] = cfg['mapping']['stage'][self.stage]['coarse_lr']*lr_factor
                optimizer.param_groups[2]['lr'] = cfg['mapping']['stage'][self.stage]['middle_lr']*lr_factor
                optimizer.param_groups[3]['lr'] = cfg['mapping']['stage'][self.stage]['fine_lr']*lr_factor
                optimizer.param_groups[4]['lr'] = cfg['mapping']['stage'][self.stage]['color_lr']*lr_factor
                if self.BA:
                    if self.stage == 'color':
                        optimizer.param_groups[5]['lr'] = self.BA_cam_lr
            else:
                self.stage = 'color'
                optimizer.param_groups[0]['lr'] = cfg['mapping']['imap_decoders_lr']
                if self.BA:
                    optimizer.param_groups[1]['lr'] = self.BA_cam_lr

            if (not (idx == 0 and self.no_vis_on_first_frame)) and ('Demo' not in self.output):
                self.visualizer.vis(
                    idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, self.c, self.decoders)

            optimizer.zero_grad()
            batch_rays_d_list = []
            batch_rays_o_list = []
            batch_gt_depth_list = []
            batch_gt_color_list = []

            # 4.2
            camera_tensor_id = 0
            for frame in optimize_frame:
                if frame != -1:
                    gt_depth = keyframe_dict[frame]['depth'].to(device)
                    gt_color = keyframe_dict[frame]['color'].to(device)
                    if self.BA and frame != oldest_frame:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        camera_tensor_id += 1
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = keyframe_dict[frame]['est_c2w']

                else:
                    gt_depth = cur_gt_depth.to(device)
                    gt_color = cur_gt_color.to(device)
                    if self.BA:
                        camera_tensor = camera_tensor_list[camera_tensor_id]
                        c2w = get_camera_from_tensor(camera_tensor)
                    else:
                        c2w = cur_c2w

                batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color = get_samples(
                    0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2w, gt_depth, gt_color, self.device)
                batch_rays_o_list.append(batch_rays_o.float())
                batch_rays_d_list.append(batch_rays_d.float())
                batch_gt_depth_list.append(batch_gt_depth.float())
                batch_gt_color_list.append(batch_gt_color.float())

            batch_rays_d = torch.cat(batch_rays_d_list)
            batch_rays_o = torch.cat(batch_rays_o_list)
            batch_gt_depth = torch.cat(batch_gt_depth_list)
            batch_gt_color = torch.cat(batch_gt_color_list)

            # 4.3
            if self.nice:
                # should pre-filter those out of bounding box depth value
                with torch.no_grad():
                    det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)  # (N, 3, 1)
                    t = (self.bound.unsqueeze(0).to(
                        device)-det_rays_o)/det_rays_d
                    t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                    inside_mask = t >= batch_gt_depth
                batch_rays_d = batch_rays_d[inside_mask]
                batch_rays_o = batch_rays_o[inside_mask]
                batch_gt_depth = batch_gt_depth[inside_mask]
                batch_gt_color = batch_gt_color[inside_mask]
            # 渲染
            ret = self.renderer.render_batch_ray(c, self.decoders, batch_rays_d,
                                                 batch_rays_o, device, self.stage,
                                                 gt_depth=None if self.coarse_mapper else batch_gt_depth)
            depth, uncertainty, color = ret

            # 4.4
            depth_mask = (batch_gt_depth > 0)
            loss = torch.abs(batch_gt_depth[depth_mask]-depth[depth_mask]).sum()
            if ((not self.nice) or (self.stage == 'color')):
                color_loss = torch.abs(batch_gt_color - color).sum()
                weighted_color_loss = self.w_color_loss*color_loss
                loss += weighted_color_loss

            # for imap*, it uses volume density
            regulation = (not self.occupancy)
            if regulation:
                point_sigma = self.renderer.regulation(
                    c, self.decoders, batch_rays_d, batch_rays_o, batch_gt_depth, device, self.stage)
                regulation_loss = torch.abs(point_sigma).sum()
                loss += 0.0005*regulation_loss

            loss.backward(retain_graph=False)
            optimizer.step()
            if not self.nice:
                # for imap*
                scheduler.step()
            optimizer.zero_grad()

            # 4.5
            # put selected and updated features back to the grid
            if self.nice and self.frustum_feature_selection:
                for key, val in c.items():
                    if (self.coarse_mapper and 'coarse' in key) or \
                            ((not self.coarse_mapper) and ('coarse' not in key)):
                        val_grad = masked_c_grad[key]
                        mask = masked_c_grad[key+'mask']
                        val = val.detach()
                        val[mask] = val_grad.clone().detach()
                        c[key] = val
        # 5、启动 BA 将相机维兹更新回 keyframe_dict 中
        if self.BA:
            # put the updated camera poses back
            camera_tensor_id = 0
            for id, frame in enumerate(optimize_frame):
                if frame != -1:
                    if frame != oldest_frame:
                        c2w = get_camera_from_tensor(camera_tensor_list[camera_tensor_id].detach())
                        c2w = torch.cat([c2w, bottom], dim=0)
                        camera_tensor_id += 1
                        keyframe_dict[frame]['est_c2w'] = c2w.clone()
                else:
                    c2w = get_camera_from_tensor(camera_tensor_list[-1].detach())
                    c2w = torch.cat([c2w, bottom], dim=0)
                    cur_c2w = c2w.clone()
        if self.BA:
            return cur_c2w
        else:
            return None

    def run(self):
        # 保存参数
        cfg = self.cfg
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]  # 取数据集中的第一个样本

        self.estimate_c2w_list[0] = gt_c2w.cpu()    # 保存相机-世界的坐标变化
        init = True # 定义一个flag
        prev_idx = -1
        while (1):  # 大循环 通过内部的 break 来跳出
            while True: # 小循环 通过内部的 break 来跳出
                idx = self.idx[0].clone()   # 对象属性idx的第一个值，克隆一份给局部变量idx
                if idx == self.n_img-1: # 获取到的值是最后一个值，即列表中只有1个值
                    break
                    
                # 同步方法
                if self.sync_method == 'strict':
                    if idx % self.every_frame == 0 and idx != prev_idx:# 是关键帧 且 不等于上一帧
                        break
                elif self.sync_method == 'loose':
                    if idx == 0 or idx >= prev_idx+self.every_frame//2:# 不是第一帧 或 当前
                        break
                elif self.sync_method == 'free':
                    break
                time.sleep(0.1)
            prev_idx = idx

            if self.verbose:
                print(Fore.GREEN)
                prefix = 'Coarse ' if self.coarse_mapper else ''
                print(prefix+"Mapping Frame ", idx.item())
                print(Style.RESET_ALL)

            _, gt_color, gt_depth, gt_c2w = self.frame_reader[idx]  # 获取 frame_reader 中 idx 样本的数据

            if not init:    # init 为 false 时（第二帧以后）
                lr_factor = cfg['mapping']['lr_factor']     # 读取学习率
                num_joint_iters = cfg['mapping']['iters']   # 读取迭代次数

                # here provides a color refinement postprocess 颜色细化
                if idx == self.n_img-1 and self.color_refine and not self.coarse_mapper:
                    outer_joint_iters = 5
                    self.mapping_window_size *= 2
                    self.middle_iter_ratio = 0.0
                    self.fine_iter_ratio = 0.0
                    num_joint_iters *= 5
                    self.fix_color = True
                    self.frustum_feature_selection = False
                else:
                    if self.nice:
                        outer_joint_iters = 1
                    else:
                        outer_joint_iters = 3

            else:   # init 为 true 时
                outer_joint_iters = 1
                lr_factor = cfg['mapping']['lr_first_factor']
                num_joint_iters = cfg['mapping']['iters_first']

            cur_c2w = self.estimate_c2w_list[idx].to(self.device)# 把估计的相机-世界坐标变换 转换成 GPU 张量，保存在 cur_c2w
            num_joint_iters = num_joint_iters//outer_joint_iters # 整除，减少内循环迭代次数
            for outer_joint_iter in range(outer_joint_iters):   # 迭代
                # keyframe_list 长度大于4，config 文件中BA 为true，当前是精建图模式，三个条件同时满足，BA为true
                self.BA = (len(self.keyframe_list) > 4) and cfg['mapping']['BA'] and (not self.coarse_mapper)

                # 重要函数调用，进行 map 优化
                _ = self.optimize_map(num_joint_iters, lr_factor, idx, gt_color, gt_depth,
                                      gt_c2w, self.keyframe_dict, self.keyframe_list, cur_c2w=cur_c2w)
                                      
                if self.BA: # 需要进行 BA 的情况
                    cur_c2w = _
                    self.estimate_c2w_list[idx] = cur_c2w   # 保存旋转矩阵

                # add new frame to keyframe set
                if outer_joint_iter == outer_joint_iters-1:
                    if (idx % self.keyframe_every == 0 or (idx == self.n_img-2)) and (idx not in self.keyframe_list):
                        self.keyframe_list.append(idx)
                        self.keyframe_dict.append({'gt_c2w':gt_c2w.cpu(), 'idx':idx, 'color':gt_color.cpu(), 'depth':gt_depth.cpu(), 'est_c2w':cur_c2w.clone()})

            if self.low_gpu_mem:
                torch.cuda.empty_cache()

            init = False
            # mapping of first frame is done, can begin tracking
            # NICE-SLAM.py 里的 tracking 函数，用于跳出 while(1)，进而执行 tracking.run
            self.mapping_first_frame[0] = 1

            if not self.coarse_mapper:  # 不是粗糙地图的情况
                # log 日志
                if ((not (idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) \
                        or idx == self.n_img-1:
                    self.logger.log(idx, self.keyframe_dict, self.keyframe_list,# 记录当前关键帧信息
                                    selected_keyframes=self.selected_keyframes
                                    if self.save_selected_keyframes_info else None)

                self.mapping_idx[0] = idx
                self.mapping_cnt[0] += 1

                # 生成 mesh
                if (idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame)):
                    mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'    # 生成输出文件的文件名的字符串
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)# 生成mesh文件

                if idx == self.n_img-1:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'
                    self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict, self.estimate_c2w_list,
                                         idx,  self.device, show_forecast=self.mesh_coarse_level,
                                         clean_mesh=self.clean_mesh, get_mask_use_all_frames=False)
                    os.system(f"cp {mesh_out_file} {self.output}/mesh/{idx:05d}_mesh.ply")
                    if self.eval_rec:
                        mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                        self.mesher.get_mesh(mesh_out_file, self.c, self.decoders, self.keyframe_dict,
                                             self.estimate_c2w_list, idx, self.device, show_forecast=False,
                                             clean_mesh=self.clean_mesh, get_mask_use_all_frames=True)
                    break

            if idx == self.n_img-1:
                break
