import argparse
import os
import time

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader

from src import config
from src.tools.viz import SLAMFrontend
from src.utils.datasets import get_dataset

# 假设命令为
# python visualizer.py configs/Apartment/apartment.yaml --output output/vis/Apartment
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments to visualize the SLAM process.')
    parser.add_argument('config', type=str, help='Path to config file.')
    # config 参数为 'configs/Apartment/apartment.yaml'
    parser.add_argument('--input_folder', type=str,
        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
        help='output folder, this have higher priority, can overwrite the one in config file')
    # output 参数为 'output/vis/Apartment'
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    parser.set_defaults(nice=True)
    parser.add_argument('--save_rendering', action='store_true',
                        help='save rendering video to `vis.mp4` in output folder ')
    parser.add_argument('--vis_input_frame', action='store_true', help='visualize input frames')
    parser.add_argument('--no_gt_traj', action='store_true', help='not visualize gt trajectory')
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')
    # 加载 'configs/Apartment/apartment.yaml' 和 'configs/nice_slam.yaml' 两个配置文件
    
    scale = cfg['scale']# 获取 scale 参数 在 nice-slam.yaml 文件中为 1

    # 命令行有 output 参数，就使用命令行的路径，否则用 yaml 中的路径；此处 'output/vis/Apartment'
    output = cfg['data']['output'] if args.output is None else args.output
    
    if args.vis_input_frame:    # 命令行参数里有 --vis_input_frame 的情况
        frame_reader = get_dataset(cfg, args, scale, device='cpu')# 根据 cfg['dataset'] 实例化 azure 类型数据集，其他传入参数未使用
        frame_loader = DataLoader(frame_reader, batch_size=1, shuffle=False, num_workers=4)# 加载数据
    ckptsdir = f'{output}/ckpts'    # ckptsdir = output/vis/Apartment/ckpts
    if os.path.exists(ckptsdir):    # checkpoint 路径存在
        ckpts = [os.path.join(ckptsdir, f) for f in sorted(os.listdir(ckptsdir)) if 'tar' in f]
        # 上一条语句产生的效果
        # ckpts = output/vis/Apartment/ckpts/00500.tar
        #         output/vis/Apartment/ckpts/01000.tar
        #         output/vis/Apartment/ckpts/01500.tar
        #         output/vis/Apartment/ckpts/02000.tar
        #           ......
        if len(ckpts) > 0:  # 列表不为空
            ckpt_path = ckpts[-1]   # 获取最后一个 tar 文件
            print('Get ckpt :', ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))# 加载最后一个tar文件
            estimate_c2w_list = ckpt['estimate_c2w_list']# 获取其中的 estimate_c2w_list 这个列表
            gt_c2w_list = ckpt['gt_c2w_list']# 获取其中的 gt_c2w_list
            N = ckpt['idx']# 获取 idx 值
    estimate_c2w_list[:, :3, 3] /= scale    # 缩放，此处除以1
    gt_c2w_list[:, :3, 3] /= scale          # 缩放，此处除以1
    estimate_c2w_list = estimate_c2w_list.cpu().numpy() # 转换成 numpy 数组
    gt_c2w_list = gt_c2w_list.cpu().numpy()             # 转换成 numpy 数组

    # 实例化类
    frontend = SLAMFrontend(output, init_pose=estimate_c2w_list[0], cam_scale=0.3,
                            save_rendering=args.save_rendering, near=0,
                            estimate_c2w_list=estimate_c2w_list, gt_c2w_list=gt_c2w_list).start()

    for i in tqdm(range(0, N+1)):
        # 命令行里有 vis_input_frame 参数的情况，每两帧显示一次图片，用于加速
        if args.vis_input_frame and i % 2 == 0:
            idx, gt_color, gt_depth, gt_c2w = frame_reader[i]
            depth_np = gt_depth.numpy()
            color_np = (gt_color.numpy()*255).astype(np.uint8)
            depth_np = depth_np/np.max(depth_np)*255
            depth_np = np.clip(depth_np, 0, 255).astype(np.uint8)
            depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_JET)
            color_np = np.clip(color_np, 0, 255)
            whole = np.concatenate([color_np, depth_np], axis=0)
            H, W, _ = whole.shape
            whole = cv2.resize(whole, (W//4, H//4))
            cv2.imshow(f'Input RGB-D Sequence', whole[:, :, ::-1])
            cv2.waitKey(1)
        time.sleep(0.03)

        meshfile = f'{output}/mesh/{i:05d}_mesh.ply'    # 生成文件名
        # meshfile = 'output/vis/Apartment/mesh/xxxxx_mesh.ply'
        
        if os.path.isfile(meshfile):    # 如果有这个文件的话
            frontend.update_mesh(meshfile)  # 将 meshfile 存放入队列
        
        frontend.update_pose(1, estimate_c2w_list[i], gt=False) # 更新位姿（放入队列）
        
        if not args.no_gt_traj:
            frontend.update_pose(1, gt_c2w_list[i], gt=True)    # 更新位姿（放入队列）
        # the visualizer might get stucked if update every frame
        # with a long sequence (10000+ frames)
        if i % 10 == 0: # 每 10 帧更新一次相机轨迹
            frontend.update_cam_trajectory(i, gt=False)         # 更新相机轨迹（放入队列）
            if not args.no_gt_traj:
                frontend.update_cam_trajectory(i, gt=True)      # 更新相机轨迹（放入队列）

    if args.save_rendering: # 根据命令行决定是否保存渲染过程的动画
        time.sleep(1)
        os.system(f"/usr/bin/ffmpeg -f image2 -r 30 -pattern_type glob -i '{output}/tmp_rendering/*.jpg' -y {output}/vis.mp4")
