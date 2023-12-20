import argparse # 命令行参数解析工具
import random

import numpy as np
import torch

from src import config
from src.NICE_SLAM import NICE_SLAM


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # setup_seed(20)

    # 实例化 命令行参数解析器
    parser = argparse.ArgumentParser(description='Arguments for running the NICE-SLAM/iMAP*.')
    
    # 向对象中添加参数
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str, help='output folder, this have higher priority, can overwrite the one in config file')
    
    # 实例化 互斥参数组（在parser对象的基础上添加互斥组特性）
    nice_parser = parser.add_mutually_exclusive_group(required=False)
    
    # 向对象中添加参数
    nice_parser.add_argument('--nice', dest='nice', action='store_true')
    nice_parser.add_argument('--imap', dest='nice', action='store_false')
    
    parser.set_defaults(nice=True)  # 设置默认参数是 nice，否则是 imap
    args = parser.parse_args()  # 解析命令行参数

    # 加载配置文件 若 arg.config 是 true， 则加载 nice-slam 的配置文件，若是 false 则加载 imap 的配置文件
    cfg = config.load_config(args.config, 'configs/nice_slam.yaml' if args.nice else 'configs/imap.yaml')

    slam = NICE_SLAM(cfg, args) # 实例化 NICE-SLAM 对象，传入参数

    slam.run()                  # 运行


if __name__ == '__main__':
    main()
