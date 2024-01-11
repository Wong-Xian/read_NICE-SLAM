import os
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np
import open3d as o3d
import torch


def normalize(x):
    return x / np.linalg.norm(x)


def create_camera_actor(i, is_gt=False, scale=0.005):
    cam_points = scale * np.array([[0  ,  0,   0],
                                   [-1 , -1, 1.5],
                                   [1  , -1, 1.5],
                                   [1  ,  1, 1.5],
                                   [-1 ,  1, 1.5],
                                   [-0.5, 1, 1.5],
                                   [0.5,  1, 1.5],
                                   [0  ,1.2, 1.5]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], 
                          [4, 1], [1, 3], [2, 4],
                          [1, 0], [0, 2], [3, 0], 
                          [0, 4], [5, 7], [7, 6]])
    
    points = []
    
    for cam_line in cam_lines:
        begin_points, end_points = cam_points[cam_line[0]], cam_points[cam_line[1]]
        t_vals = np.linspace(0., 1., 100)
        begin_points, end_points
        point = begin_points[None, :] * (1.-t_vals)[:, None] + end_points[None, :] * (t_vals)[:, None]
        points.append(point)
    
    points = np.concatenate(points)
    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
    camera_actor = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(points))
    camera_actor.paint_uniform_color(color)

    return camera_actor


def draw_trajectory(queue, output, init_pose, cam_scale,
                    save_rendering, near, estimate_c2w_list, gt_c2w_list):

    draw_trajectory.queue = queue
    draw_trajectory.cameras = {}
    draw_trajectory.points = {}
    draw_trajectory.ix = 0
    draw_trajectory.warmup = 0
    draw_trajectory.mesh = None
    draw_trajectory.frame_idx = 0
    draw_trajectory.traj_actor = None
    draw_trajectory.traj_actor_gt = None
    if save_rendering:
        os.system(f"rm -rf {output}/tmp_rendering")

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        while True:
            try:
                data = draw_trajectory.queue.get_nowait()   # 从队列中取一个 帧？
                
                # pose 的情况
                if data[0] == 'pose':
                    i, pose, is_gt = data[1:]   # 获取索引，位姿，是否是gt
                    if is_gt:
                        i += 100000

                    if i in draw_trajectory.cameras:
                        cam_actor, pose_prev = draw_trajectory.cameras[i]
                        pose_change = pose @ np.linalg.inv(pose_prev)   # 计算当前位姿和上一位姿的变化

                        cam_actor.transform(pose_change)    # 根据 pose_change 更新 cam_actor
                        vis.update_geometry(cam_actor)      # 更新场景几何

                        if i in draw_trajectory.points:
                            pc = draw_trajectory.points[i]  # 获取点的位置
                            pc.transform(pose_change)       # 根据 pose_change 更新点的位置
                            vis.update_geometry(pc)         # 更新点云的场景几何

                    else:
                        cam_actor = create_camera_actor(i, is_gt, cam_scale)
                        cam_actor.transform(pose)           # 根据 pose_change 更新 cam_actor
                        vis.add_geometry(cam_actor)

                    draw_trajectory.cameras[i] = (cam_actor, pose)

                # mesh 的情况
                elif data[0] == 'mesh':
                    meshfile = data[1]
                    if draw_trajectory.mesh is not None:
                        vis.remove_geometry(draw_trajectory.mesh)
                    draw_trajectory.mesh = o3d.io.read_triangle_mesh(meshfile)
                    draw_trajectory.mesh.compute_vertex_normals()
                    # flip face orientation
                    new_triangles = np.asarray(
                        draw_trajectory.mesh.triangles)[:, ::-1]
                    draw_trajectory.mesh.triangles = o3d.utility.Vector3iVector(
                        new_triangles)
                    draw_trajectory.mesh.triangle_normals = o3d.utility.Vector3dVector(
                        -np.asarray(draw_trajectory.mesh.triangle_normals))
                    vis.add_geometry(draw_trajectory.mesh)

                # traj 的情况
                elif data[0] == 'traj':
                    i, is_gt = data[1:]

                    color = (0.0, 0.0, 0.0) if is_gt else (1.0, .0, .0)
                    traj_actor = o3d.geometry.PointCloud(
                        points=o3d.utility.Vector3dVector(gt_c2w_list[1:i, :3, 3] if is_gt else estimate_c2w_list[1:i, :3, 3]))
                    traj_actor.paint_uniform_color(color)

                    if is_gt:
                        if draw_trajectory.traj_actor_gt is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor_gt)
                            tmp = draw_trajectory.traj_actor_gt
                            del tmp
                        draw_trajectory.traj_actor_gt = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor_gt)
                    else:
                        if draw_trajectory.traj_actor is not None:
                            vis.remove_geometry(draw_trajectory.traj_actor)
                            tmp = draw_trajectory.traj_actor
                            del tmp
                        draw_trajectory.traj_actor = traj_actor
                        vis.add_geometry(draw_trajectory.traj_actor)

                # reset 的情况
                elif data[0] == 'reset':
                    draw_trajectory.warmup = -1

                    for i in draw_trajectory.points:
                        vis.remove_geometry(draw_trajectory.points[i])

                    for i in draw_trajectory.cameras:
                        vis.remove_geometry(draw_trajectory.cameras[i][0])

                    draw_trajectory.cameras = {}
                    draw_trajectory.points = {}

            except Empty:
                break

        # hack to allow interacting with vizualization during inference
        if len(draw_trajectory.cameras) >= draw_trajectory.warmup:
            cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

        vis.poll_events()
        vis.update_renderer()
        if save_rendering:
            # save the renderings, useful when making a video
            draw_trajectory.frame_idx += 1
            os.makedirs(f'{output}/tmp_rendering', exist_ok=True)
            vis.capture_screen_image(
                f'{output}/tmp_rendering/{draw_trajectory.frame_idx:06d}.jpg')

    vis = o3d.visualization.Visualizer()    # 实例化一个“可视化界面”

    vis.register_animation_callback(animation_callback) # 注册一个回调函数
    vis.create_window(window_name=output, height=1080, width=1920)# 创建窗口
    vis.get_render_option().point_size = 4
    vis.get_render_option().mesh_show_back_face = False

    ctr = vis.get_view_control()    # 可以控制视角的实例
    ctr.set_constant_z_near(near)   # 最近距离？
    ctr.set_constant_z_far(1000)    # 最远距离？

    # set the viewer's pose in the back of the first frame's pose
    param = ctr.convert_to_pinhole_camera_parameters()  # 转换成针孔相机模型参数
    init_pose[:3, 3] += 2*normalize(init_pose[:3, 2])
    init_pose[:3, 2] *= -1
    init_pose[:3, 1] *= -1
    init_pose = np.linalg.inv(init_pose)

    param.extrinsic = init_pose
    ctr.convert_from_pinhole_camera_parameters(param)

    vis.run()           # 运行窗口
    vis.destroy_window()# 销毁窗口


class SLAMFrontend:
    def __init__(self, output, init_pose, cam_scale=1, save_rendering=False,
                 near=0, estimate_c2w_list=None, gt_c2w_list=None):
        self.queue = Queue()
        self.p = Process(target=draw_trajectory, 
                         args=(self.queue, 
                               output, 
                               init_pose, 
                               cam_scale, 
                               save_rendering,
                               near, 
                               estimate_c2w_list, 
                               gt_c2w_list))

    def update_pose(self, index, pose, gt=False):
        if isinstance(pose, torch.Tensor):  # 如果 pose 是 torch.Tensor 类型
            pose = pose.cpu().numpy()       # 将其转换成 numpy 类型
        pose[:3, 2] *= -1   # 将 pose 第三列前三个数取负数
        self.queue.put_nowait(('pose', index, pose, gt))# 加入队列
        
    def update_mesh(self, path):
        self.queue.put_nowait(('mesh', path))   # 把 mesh 加入队列

    def update_cam_trajectory(self, c2w_list, gt):
        self.queue.put_nowait(('traj', c2w_list, gt))   # 把姿态和gt值加入队列

    def reset(self):
        self.queue.put_nowait(('reset', ))

    def start(self):
        self.p.start()
        return self

    def join(self):
        self.p.join()
