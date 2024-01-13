import math
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


def _find_closest_key(input_dict, target_key):
    # 获取字典中所有键
    all_keys = list(input_dict.keys())
    # 找到与目标键最接近的键
    closest_key = min(all_keys, key=lambda x: abs(x - target_key))
    return closest_key, input_dict[closest_key]