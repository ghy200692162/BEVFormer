#sample
# {'lidar_path': '0',
#  'token': '',
#   'prev': '',
#  'next': '', 
#  'can_bus': '', 
#  'frame_idx': 0, 
#  'sweeps': [], 
#  'cams': {'CAM_FRONT': {'data_path': '1660892076462496512.jpg', 'type': 'CAM_FRONT'}, 
#  'CAM_FRONT_RIGHT': {'data_path': '1660892076476645376.jpg', 'type': 'CAM_FRONT_RIGHT'}, 
#  'CAM_FRONT_LEFT': {'data_path': '1660892076444809728.jpg', 'type': 'CAM_FRONT_LEFT'}, 
#  'CAM_BACK': {'data_path': '1660892076486635264.jpg', 'type': 'CAM_BACK'}, 
#  'CAM_BACK_LEFT': {'data_path': '1660892076467676160.jpg', 'type': 'CAM_BACK_LEFT'}, 
#  'CAM_BACK_RIGHT': {'data_path': '1660892076486119168.jpg', 'type': 'CAM_BACK_RIGHT'}}, 
#  'scene_token': '', 'lidar2ego_translation': '', 
#  'lidar2ego_rotation': '', 
#  'ego2global_translation': '', 
#  'ego2global_rotation': '', 
#  'timestamp': 1660892076.4624965}


# 读取点云文件
import copy
import os
from typing import List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from scipy.spatial.transform import Rotation


labels = np.array([0,1,2,3,4])
classes = np.array(["unkonwn","pedestrian","bicycle","car","bus"])


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "unkonwn": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "tricycle": (220, 20, 60),  # 相比原版 mmdet3d 的 visualize 增加 tricycle
    "bicycle": (220, 20, 60)  # 相比原版 mmdet3d 的 visualize 增加 cyclist
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}


def get_lidar2image():
    lidar2cam_ext =  np.array([[ 0.0442784, -0.999015, -0.00286609, -0.328146],
                           [-0.00370547, 0.00270465, -0.999989, 1.44921],
                           [ 0.999012, 0.0442885, -0.00358206, -1.93053],
                           [ 0.0,  0.0,  0.0,  1.0]])

    cam_intrinsic_matrix = [[7327.52569900726, 0, 1973.90006706734],
                            [0, 7358.97683813612, 958.209039784179],
                            [0, 0, 1]]

    lidar2camera_r = lidar2cam_ext[:3, :3]
    lidar2camera_t = lidar2cam_ext[:3, 3]

    lidar2camera_rt = np.eye(4).astype(np.float32)
    lidar2camera_rt[:3, :3] = lidar2camera_r.T
    lidar2camera_rt[3, :3] = -lidar2camera_t
    camera_intrinsics = np.eye(4).astype(np.float32)

    camera_intrinsics[:3, :3] = np.array(cam_intrinsic_matrix)
    lidar2image = camera_intrinsics @ lidar2camera_rt.T

    return lidar2image



def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()



def _transform_car_point_to_camera(point_car):
    from scipy.spatial.transform import Rotation

    point_camera = []
        # 5号车前长焦外参数
        # transform:
        #     translation:
        #     x: 0.246242
        #     y: 1.94852
        #     z: 1.44134
        # rotation:
        #     x: -0.70822
        #     y: -0.0146754
        #     z: 0.0166455
        #     w: 0.705643

    # 定义平移向量
    translation_vector = np.array([ 0.246242, 1.94852, 1.44134 ])
    # 创建Rotation对象并使用as_quat方法获取四元数

    quaternion = np.array([-0.70822, -0.0146754, 0.0166455, 0.705643])
    print("得到的四元数：", quaternion)
    rotation = Rotation.from_quat(quaternion)
    # 使用四元数进行旋转
    point_car_rotated = rotation.apply(point_car)
    point_camera_extrin = point_car_rotated + translation_vector


    # 相机内参
    # 内参矩阵
    K = np.array([[7327.52569900726, 0, 1973.90006706734],
              [0, 7358.97683813612, 958.209039784179],
              [0, 0, 1]])
    # 畸变系数
    D = np.array([-0.172501828478052, -0.603745270942309, 0.00155951373290956, 0.00153030499564137, 1.38343696720889])
    # 将点投影到相机坐标系
    camera_point = np.dot(K, point_camera_extrin)
    # 归一化相机坐标系下的点
    camera_point_normalized = camera_point / camera_point[2]
    # 打印相机坐标系下的点
    print("相机坐标系下的点:", camera_point_normalized)
    image_height = 2160
    image_width = 3840

    # 将相机坐标系下的点投影到图像平面
    point_camera = np.array([camera_point_normalized[0] * image_width,
                        camera_point_normalized[1] * image_height])

    return point_camera

def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        print(coords)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        # labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        # labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def draw_obstacle(fpath,obstacle_params,
                    point_cloud,
                    type="2D",
                    xlim: Tuple[float, float] = (-100, 100),
                    ylim: Tuple[float, float] = (-100, 100),
                    color: Optional[Tuple[int, int, int]] = None,
                    radius: float = 15,
                    thickness: float = 25):

    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.set_aspect(1)
    ax.set_axis_off()

    # 绘制点云
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], s=1, color='white')
    for obstacle in obstacle_params:
        x, y, z, length, width, height, yaw = obstacle
        # 定义障碍物框的4个顶点
        corners = np.array([
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2]
        ])

        # 将障碍物框进行旋转
        rotation_matrix = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        rotated_corners = corners.dot(rotation_matrix.T) + np.array([x, y])
        # 绘制障碍物框 
        obstacle_polygon = Polygon(rotated_corners, closed=True, edgecolor='red', facecolor='none', linewidth=10)
        ax.add_patch(obstacle_polygon)

    # 绘制点云
    # 显示图像
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def _transform_car_point_to_lidar(point_car):
    from scipy.spatial.transform import Rotation
    # 5号车lidar的标定参数
    yaw_radians = -1.578619956970215 
    pitch_radians = -0.002790384536371908
    roll_radians = -0.004345358349382877
    # 定义平移向量
    translation_vector = np.array([0.92, 0, 2.097419023513794])
    # 创建Rotation对象并使用as_quat方法获取四元数
    rotation = Rotation.from_euler('ZYX', [yaw_radians, pitch_radians, roll_radians], degrees=False)
    quaternion = rotation.as_quat()
    # print("得到的四元数：", quaternion)
    rotation = Rotation.from_quat(quaternion)
    
    # 使用四元数进行旋转
    point_car_rotated = rotation.apply(point_car)
    point_lidar = point_car_rotated
    # 添加平移
    # point_lidar = point_car_rotated + translation_vector
    return point_lidar

def _test_visual_label_to_lidar():
    # lidar_time 和图片的measure_time 最接近的一帧图像和一帧点云
    # image 
    # measure_ment = 1660892077.200013
    # header_time =  1660892077.263186

    # pcd 13.pcd 
    # lidar_time = 1660892077.200094
    # 障碍物信息从这一时刻的可视化界面上获得
    data_path = "tools/dv_visualizer/gt_npy/13_arr.npy"
    pcd_np = np.load(data_path)
    # 原始数据坐标系
    point_car_1 = [3.1486918926239014, -9.509262084960938,  1.6899200677871704] #1660892077.214
    point_car_2 = [3.5677318572998047, -38.24081039428711, 0.9978694319725037]  #1660892076.421    
    point_car_3 = [-7.934744834899902, 37.1341438293457,1.6461652517318726]     #1660892077.271

    import torch
    point_lidar_1 = point_car_1 #_transform_car_point_to_lidar(point_car=point_car_1)
    point_lidar_2 = point_car_2 # transform_car_point_to_lidar(point_car=point_car_2)
    point_lidar_3 = point_car_3 #_transform_car_point_to_lidar(point_car=point_car_3)

    # point_lidar_1 = _transform_car_point_to_lidar(point_car=point_car_1)
    # point_lidar_2 = _transform_car_point_to_lidar(point_car=point_car_2)
    # point_lidar_3 = _transform_car_point_to_lidar(point_car=point_car_3)
    
    box1 = [
            point_lidar_1[0],
            point_lidar_1[1],
            point_lidar_1[2],
            11.161832809448242,
            2.5999999046325684,
            3.1942873001098633,
            0.007397098305516389
        ]
    box2 = [
            point_lidar_2[0],
            point_lidar_2[1],
            point_lidar_2[2],
            4.9206743240356445 ,
            2.0440545082092285,
            1.5,
            -0.0016795649347141834
        ]
    box3 = [
            point_lidar_3[0],
            point_lidar_3[1],
            point_lidar_3[2],
            10.573736190795898,
            2.5999999046325684,
            3.0411298274993896,
            -3.1340792133704456
    ]
    print(point_car_3)
    print("after tf")
    print(point_lidar_3)
    # print(box3)
    boxes = [box1,box2,box3]
    test_boxes = [box1,box2,box3]
    boxes_tensor = torch.tensor(test_boxes,dtype=torch.float32)
    label_boxes = LiDARInstance3DBoxes(tensor=boxes_tensor)
    print(boxes_tensor[0])

        # visualize_lidar("13.png",
        #                 bboxes=label_boxes,
        #                 labels=labels,
        #                 classes=classes,
        #                 lidar=pcd_np)
    draw_obstacle("13.png",point_cloud=pcd_np,obstacle_params=boxes)

def _test_visual_label_to_image():
    import torch
    cam_front_image = "tools/dv_visualizer/data/cam_front/1660892076412599808.jpg"
    image = cv2.imread(cam_front_image)
    image_np = np.array(image)
    # 1660892076462496512
    # 1660892076472
    # lidar 坐标系 
    # 图像坐标系下
    # lidar --》camera ---》 img
    #       外参        内参
    

    point_car_1 = [0.3372171223163605, 81.91390228271484, 0.7423714399337769]
    point_car_2 = [-7.974083423614502, 62.70072937011719, 1.570648193359375]

    point_lidar_1 = point_car_1 #_transform_car_point_to_lidar(point_car_1)
    point_lidar_2 = point_car_2 #_transform_car_point_to_lidar(point_car_2)

    # point_lidar_1 = _transform_car_point_to_lidar(point_car_1)
    # point_lidar_2 = _transform_car_point_to_lidar(point_car_2)

    box1 = [
        point_lidar_1[0],
        point_lidar_1[1],
        point_lidar_1[2],
        4.972667217254639, 
        2.2444114685058594, 
        1.6746292114257812,
        0.000146653819693216,
    ]

    box2 = [
        point_lidar_2[0],
        point_lidar_2[1],
        point_lidar_2[2],
        10.573736190795898,
        2.5999999046325684,
        3.0411298274993896,
        0.001461384413595719,
    ]

    transform = get_lidar2image()
    test_boxes = [box1,box2]
    boxes_tensor = torch.tensor(test_boxes,dtype=torch.float32)
    label_boxes = LiDARInstance3DBoxes(tensor=boxes_tensor)

    visualize_camera(fpath="cam_front.png",
                        bboxes=label_boxes,
                        transform=transform,
                        image=image_np,
                        labels=labels,
                        classes=classes)



if __name__ == "__main__":
    # _test_visual_label_to_image()
    _test_visual_label_to_lidar()
