import math
import numpy as np
import hashlib

from scipy.spatial.transform import Rotation


def to_quat(yaw_radians,pitch_radians,roll_radians):
    rotation = Rotation.from_euler('ZYX', [yaw_radians, pitch_radians, roll_radians], degrees=False)
    quaternion = rotation.as_quat()
    return quaternion

def llh_to_xyz(latitude, longitude, altitude):
    # 地球半径（单位：公里）
    earth_radius = 6371.0
    # 将经纬度转换为弧度
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    # 计算直角坐标系中的坐标
    x = (earth_radius + altitude) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (earth_radius + altitude) * math.cos(lat_rad) * math.sin(lon_rad)
    z = (earth_radius + altitude) * math.sin(lat_rad)
    return x, y, z


def find_closest_key(input_dict, target_key):
    # 获取字典中所有键
    all_keys = list(input_dict.keys())
    # 找到与目标键最接近的键
    closest_key = min(all_keys, key=lambda x: abs(x - target_key))
    return closest_key, input_dict[closest_key]



def compute_sensor_to_lidar(sensor2ego, lidar2ego):
    """
    计算 sensor 到 lidar 的外参

    参数：
    - sensor2ego: sensor 到 ego 的外参，[translation, rotation]
    - lidar2ego: lidar 到 ego 的外参，[translation, rotation]

    返回值：
    - sensor2lidar: sensor 到 lidar 的外参，[translation, rotation]
    """

    translation_sensor, rotation_sensor = sensor2ego
    translation_lidar, rotation_lidar = lidar2ego

    # 构建旋转矩阵
    rotation_matrix_sensor = Rotation.from_quat(rotation_sensor).as_matrix()
    rotation_matrix_lidar = Rotation.from_quat(rotation_lidar).as_matrix()

    # 构建变换矩阵
    sensor2ego_matrix = np.eye(4)
    sensor2ego_matrix[:3, :3] = rotation_matrix_sensor
    sensor2ego_matrix[:3, 3] = translation_sensor

    lidar2ego_matrix = np.eye(4)
    lidar2ego_matrix[:3, :3] = rotation_matrix_lidar
    lidar2ego_matrix[:3, 3] = translation_lidar

    # 计算 sensor 到 lidar 的变换矩阵
    sensor2lidar_matrix = np.dot(sensor2ego_matrix, np.linalg.inv(lidar2ego_matrix))

    # 提取平移和旋转
    translation_sensor2lidar = sensor2lidar_matrix[:3, 3]
    rotation_sensor2lidar = Rotation.from_matrix(sensor2lidar_matrix[:3, :3]).as_quat()

    return [translation_sensor2lidar, rotation_sensor2lidar]

    # # 示例用法
    # sensor2ego_example = [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0]]  # 例子中的 sensor2ego 外参
    # lidar2ego_example = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]  # 例子中的 lidar2ego 外参

    # sensor2lidar_example = compute_sensor_to_lidar(sensor2ego_example, lidar2ego_example)
    # print("sensor2lidar_example:", sensor2lidar_example)
