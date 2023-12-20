import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

torch.multiprocessing.set_sharing_strategy('file_system')


class NICE_SLAM():  # 自定义类
    """
    NICE_SLAM main class.
    Mainly allocate shared resources, and dispatch mapping and tracking process.
    """

    def __init__(self, cfg, args):  # 构造函数

        self.cfg = cfg
        self.args = args
        self.nice = args.nice

        # 保存 cfg 中的参数 即从命令行中获得的参数
        self.coarse = cfg['coarse']
        self.occupancy = cfg['occupancy']
        self.low_gpu_mem = cfg['low_gpu_mem']
        self.verbose = cfg['verbose']
        self.dataset = cfg['dataset']
        self.coarse_bound_enlarge = cfg['model']['coarse_bound_enlarge']
        
        if args.output is None:
            self.output = cfg['data']['output']
        else:
            self.output = args.output
        
        self.ckptsdir = os.path.join(self.output, 'ckpts')  # 这个变量保存的是 output 和 'ckpts' 两个字符串的拼接
        os.makedirs(self.output, exist_ok=True)             # 创建目录
        os.makedirs(self.ckptsdir, exist_ok=True)           # 创建目录
        os.makedirs(f'{self.output}/mesh', exist_ok=True)   # 创建目录
        
        # 保存相机参数
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam()   # 更新相机参数

        model = config.get_model(cfg, nice=self.nice)  # 获取神经网络的模板
        self.shared_decoders = model    # 保存为对象属性
        self.scale = cfg['scale']
        self.load_bound(cfg)    # 加载场景边界参数
        
        if self.nice:
            self.load_pretrain(cfg) # 加载预训练的网络
            self.grid_init(cfg)
        else:
            self.shared_c = {}

        # need to use spawn
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale)
        self.n_img = len(self.frame_reader)
        
        # 初始化共享内存
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.estimate_c2w_list.share_memory_()  # 将该变量共享到多进程

        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list.share_memory_()
        
        self.idx = torch.zeros((1)).int()
        self.idx.share_memory_()
        
        self.mapping_first_frame = torch.zeros((1)).int()
        self.mapping_first_frame.share_memory_()
        
        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int()
        self.mapping_idx.share_memory_()
        
        self.mapping_cnt = torch.zeros((1)).int()  # counter for mapping
        self.mapping_cnt.share_memory_()
        
        # 将共享变量移至指定设备
        for key, val in self.shared_c.items():
            val = val.to(self.cfg['mapping']['device'])
            val.share_memory_()
            self.shared_c[key] = val
        
        self.shared_decoders = self.shared_decoders.to(self.cfg['mapping']['device'])
        self.shared_decoders.share_memory()
        
        self.renderer = Renderer(cfg, args, self)       # 渲染线程
        self.mesher = Mesher(cfg, args, self)           # 网格线程
        self.logger = Logger(cfg, args, self)           # 日志线程
        self.mapper = Mapper(cfg, args, self, coarse_mapper=False)  # 建图线程
        if self.coarse:
            self.coarse_mapper = Mapper(cfg, args, self, coarse_mapper=True)# 粗网格的建图线程
        self.tracker = Tracker(cfg, args, self)         # 跟踪线程
        
        self.print_output_desc()    # 输出描述信息

    def print_output_desc(self):
        print(f"INFO: The output folder is {self.output}")
        if 'Demo' in self.output:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/vis/")
        else:
            print(
                f"INFO: The GT, generated and residual depth/color images can be found under " +
                f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']:
            crop_size = self.cfg['cam']['crop_size']
            sx = crop_size[1] / self.W
            sy = crop_size[0] / self.H
            self.fx = sx*self.fx
            self.fy = sy*self.fy
            self.cx = sx*self.cx
            self.cy = sy*self.cy
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        if self.cfg['cam']['crop_edge'] > 0:
            self.H -= self.cfg['cam']['crop_edge']*2
            self.W -= self.cfg['cam']['crop_edge']*2
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        """
        Pass the scene bound parameters to different decoders and self.
        加载场景边界参数
        Args:
            cfg (dict): parsed config dict.
        """
        # scale the bound if there is a global scaling factor
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale)
        bound_divisible = cfg['grid_len']['bound_divisible']
        # enlarge the bound a bit to allow it divisible by bound_divisible
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_divisible).int()+1)*bound_divisible+self.bound[:, 0]
        if self.nice:
            self.shared_decoders.bound = self.bound
            self.shared_decoders.middle_decoder.bound = self.bound
            self.shared_decoders.fine_decoder.bound = self.bound
            self.shared_decoders.color_decoder.bound = self.bound
            if self.coarse:
                self.shared_decoders.coarse_decoder.bound = self.bound*self.coarse_bound_enlarge

    def load_pretrain(self, cfg):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.

        Args:
            cfg (dict): parsed config dict
        """

        if self.coarse:
            ckpt = torch.load(cfg['pretrained_decoders']['coarse'], map_location=cfg['mapping']['device'])
            coarse_dict = {}
            for key, val in ckpt['model'].items():
                if ('decoder' in key) and ('encoder' not in key):
                    key = key[8:]
                    coarse_dict[key] = val
            self.shared_decoders.coarse_decoder.load_state_dict(coarse_dict)    # 将配置好的字典装入粗网络的状态字典

        ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'], map_location=cfg['mapping']['device'])
        middle_dict = {}
        fine_dict = {}

        for key, val in ckpt['model'].items():
            if ('decoder' in key) and ('encoder' not in key):
                if 'coarse' in key:
                    key = key[8+7:]
                    middle_dict[key] = val
                elif 'fine' in key:
                    key = key[8+5:]
                    fine_dict[key] = val
        self.shared_decoders.middle_decoder.load_state_dict(middle_dict)    # 配置好的字典装入状态字典
        self.shared_decoders.fine_decoder.load_state_dict(fine_dict)        # 配置好的字典装入状态字典

    def grid_init(self, cfg):
        """
        Initialize the hierarchical feature grids.

        Args:
            cfg (dict): parsed config dict.
        """
        # 初始化粗网格
        if self.coarse:
            coarse_grid_len = cfg['grid_len']['coarse']
            self.coarse_grid_len = coarse_grid_len
        
        # 初始化中网格
        middle_grid_len = cfg['grid_len']['middle']
        self.middle_grid_len = middle_grid_len
        
        # 初始化精网格
        fine_grid_len = cfg['grid_len']['fine']
        self.fine_grid_len = fine_grid_len
        
        # 初始化颜色网格
        color_grid_len = cfg['grid_len']['color']
        self.color_grid_len = color_grid_len

        c = {}
        c_dim = cfg['model']['c_dim']
        xyz_len = self.bound[:, 1]-self.bound[:, 0]

        # If you have questions regarding the swap of axis 0 and 2,
        # please refer to https://github.com/cvg/nice-slam/issues/24

        if self.coarse:
            coarse_key = 'grid_coarse'
            coarse_val_shape = list(map(int, (xyz_len*self.coarse_bound_enlarge/coarse_grid_len).tolist()))# 计算出需要多少个粗粒度表面网格才能覆盖整个数据集
            coarse_val_shape[0], coarse_val_shape[2] = coarse_val_shape[2], coarse_val_shape[0]# 第一个和第三个交换位置
            self.coarse_val_shape = coarse_val_shape
            val_shape = [1, c_dim, *coarse_val_shape]   # *coarse_val_shape表示将coarse_val_shape中的每个元素作为独立的参数传递给val_shape
            coarse_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)# 这个大张量将用于表示数据集中的粗粒度表面，其中每个元素表示一个粗粒度表面网格的权重
            c[coarse_key] = coarse_val

        middle_key = 'grid_middle'
        middle_val_shape = list(map(int, (xyz_len/middle_grid_len).tolist()))
        middle_val_shape[0], middle_val_shape[2] = middle_val_shape[2], middle_val_shape[0]
        self.middle_val_shape = middle_val_shape
        val_shape = [1, c_dim, *middle_val_shape]
        middle_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[middle_key] = middle_val

        fine_key = 'grid_fine'
        fine_val_shape = list(map(int, (xyz_len/fine_grid_len).tolist()))
        fine_val_shape[0], fine_val_shape[2] = fine_val_shape[2], fine_val_shape[0]
        self.fine_val_shape = fine_val_shape
        val_shape = [1, c_dim, *fine_val_shape]
        fine_val = torch.zeros(val_shape).normal_(mean=0, std=0.0001)   # 采用更小的标准差进行初始化
        c[fine_key] = fine_val

        color_key = 'grid_color'
        color_val_shape = list(map(int, (xyz_len/color_grid_len).tolist()))
        color_val_shape[0], color_val_shape[2] = color_val_shape[2], color_val_shape[0]
        self.color_val_shape = color_val_shape
        val_shape = [1, c_dim, *color_val_shape]
        color_val = torch.zeros(val_shape).normal_(mean=0, std=0.01)
        c[color_key] = color_val

        self.shared_c = c

    def tracking(self, rank):   # 跟踪线程
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished 第一帧之后才开始tracking
        while (1):
            if self.mapping_first_frame[0] == 1:
                break
            time.sleep(1)

        self.tracker.run()

    def mapping(self, rank):    # 建图线程
        """
        Mapping Thread. (updates middle, fine, and color level)

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def coarse_mapping(self, rank): # 粗地图建图线程
        """
        Coarse mapping Thread. (updates coarse level)

        Args:
            rank (int): Thread ID.
        """

        self.coarse_mapper.run()

    def run(self):
        """
        Dispatch Threads. 创建多线程
        """
        processes = []
        for rank in range(3):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            elif rank == 2:
                if self.coarse:
                    p = mp.Process(target=self.coarse_mapping, args=(rank, ))
                else:
                    continue
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
