#生成训练用的pkl数据
#解析好的数据存放在2个位置
#image:/data/dataset/dv_bev/
#其他sensor数据：data/dvscenes/sample/
#标定相关数据：目前确实这部分数据
#annotation:缺失
#info data sheama 
#    info ={'lidar_path': lidar_path,
#     'token': sample['token'],
#     'prev': sample['prev'],
#     'next': sample['next'],
#     'can_bus': can_bus,
#     'frame_idx': frame_idx,  # temporal related info
#     'sweeps': [],
#     'cams': dict(),
#     'scene_token': sample['scene_token'],  # temporal related info
#     'lidar2ego_translation': cs_record['translation'],
#     'lidar2ego_rotation': cs_record['rotation'],
#     'ego2global_translation': pose_record['translation'],
#     'ego2global_rotation': pose_record['rotation'],
#     'timestamp': sample['timestamp'],}
import json
import csv
import math
from numpy import long
cam_front_path="data/dvscenes/sample/apollo_sensor_camera_front_narrow_image_compressed.txt"
cam_left_front_path = "data/dvscenes/sample/apollo_sensor_camera_left_front_image_compressed.txt"
cam_left_rear_path = "data/dvscenes/sample/apollo_sensor_camera_left_rear_image_compressed.txt"
cam_right_front_path = "data/dvscenes/sample/apollo_sensor_camera_right_front_image_compressed.txt"
cam_right_rear_path = "data/dvscenes/sample/apollo_sensor_camera_right_rear_image_compressed.txt"
cam_rear_path = "data/dvscenes/sample/apollo_sensor_camera_rear_image_compressed.txt"

ego_data_path = "data/dvscenes/sample/apollo_sensor_gnss_gpfpd.txt"
calibration_data_path = ""
gt_data_path = "data/dvscenes/sample/GT.csv"
#
# 构建用于查询的map，key是时间戳，value是对应传感器数据
# 方便通过时间戳聚合数据
# #
def create_dv_infos(image_path,
                    root_path,
                    out_path):
    cam_front_dict = _parse_image_data(cam_front_path)
    cam_front_right_dict = _parse_image_data(cam_right_front_path)
    cam_front_left_dict = _parse_image_data(cam_left_front_path)

    cam_back_dict = _parse_image_data(cam_rear_path)
    cam_back_right_dict = _parse_image_data(cam_right_rear_path)
    cam_back_left_dict = _parse_image_data(cam_left_rear_path)

    ego_dict = _parse_gps_data(ego_data_path)

    gt_dict = _parse_gt_data(gt_data_path)
    calibration_dict = _parse_calibration_data(calibration_data_path)
    _fill_trainval_infos(cam_front_dict,
                            cam_front_right_dict,
                            cam_front_left_dict,
                            cam_back_dict,
                            cam_back_right_dict,
                            cam_back_left_dict,ego_dict,gt_dict,calibration_dict)


#聚合图像，cam_front 为基准，找到最近的一阵图像，生成图像数组 cams
#info["cams"]["CAM_BACK"].keys()
# ['data_path', 
# 'type', 
# 'sample_data_token',
# 'sensor2ego_translation', 
# 'sensor2ego_rotation', 
# 'ego2global_translation', 
# 'ego2global_rotation', 
# 'timestamp', 
# 'sensor2lidar_rotation', 
# 'sensor2lidar_translation', 
# 'cam_intrinsic']

def _fill_trainval_infos(cam_front_dict,
                           cam_front_right_dict,
                           cam_front_left_dict,
                           cam_back_dict ,
                           cam_back_right_dict,
                           cam_back_left_dict,
                           ego_dict,gt_dict,calibration_dict):

    result = []
    #filter front camera
    # lidar 10hz,camera 20hz,一帧gt ,可以匹配到2帧数据,取时间差小的数据
    filted_cam_front_dict={}
    flag_gt_dict = {}
    for gt_timestamp in gt_dict.keys():
        candidated_cam_front_key ,candidated_frame = _find_closest_key(cam_front_dict,gt_timestamp)
        if flag_gt_dict.__contains__(gt_timestamp):
            stored_frame = flag_gt_dict[gt_timestamp]
            stored_timestamp = stored_frame["header_time"]
            if abs(stored_timestamp-gt_timestamp) > abs(candidated_cam_front_key-gt_timestamp):
                filted_cam_front_dict[candidated_cam_front_key] = candidated_frame
        else:
            flag_gt_dict[gt_timestamp] = candidated_frame
            filted_cam_front_dict[candidated_cam_front_key] = candidated_frame
    #对每一组序列，降序排列，保证时间序列有效
    from collections import OrderedDict
    filted_cam_front_dict = OrderedDict(sorted(filted_cam_front_dict.items(), key=lambda x: x[0], reverse=True))
    # obtain 6 image's information per frame
    for cam_front_key,cam_front_frame in filted_cam_front_dict.items():
        cam_front_right_timestamp ,cam_front_right_frame = _find_closest_key(cam_front_right_dict,cam_front_key)
        cam_front_left_timestamp ,cam_front_left_frame = _find_closest_key(cam_front_left_dict,cam_front_key)
        cam_back_timestamp ,cam_back_frame = _find_closest_key(cam_back_dict,cam_front_key)
        cam_back_right_timestamp ,cam_back_right_frame = _find_closest_key(cam_back_right_dict,cam_front_key)
        cam_back_left_timestamp ,cam_back_left_frame = _find_closest_key(cam_back_left_dict,cam_front_key)
        
        gt_timestamp,gt_boxes = _find_closest_key(gt_dict,cam_front_key)
        ego_timestamp,ego_info = _find_closest_key(ego_dict,cam_front_key)

        # pack cams
        cams = {
            "CAM_FRONT":_pack_cam(cam_front_frame,"CAM_FRONT"),
            "CAM_FRONT_RIGHT":_pack_cam(cam_front_right_frame,"CAM_FRONT_RIGHT"),
            "CAM_FRONT_LEFT":_pack_cam(cam_front_left_frame,"CAM_FRONT_LEFT"),
            "CAM_BACK":_pack_cam(cam_back_frame,"CAM_BACK"),
            "CAM_BACK_LEFT":_pack_cam(cam_back_right_frame,"CAM_BACK_LEFT"),
            "CAM_BACK_RIGHT":_pack_cam(cam_back_left_frame,"CAM_BACK_RIGHT"),
        }

        info = {
            "lidar_path": "",
            "token": "",
            "prev":"",
            "next": "",
            "can_bus": "",
            "frame_idx": 0,  # temporal related info
            "sweeps": [],
            "cams": cams,
            "scene_token":"",  # temporal related info
            "lidar2ego_translation":"", #cs_record['translation']
            "lidar2ego_rotation":"" ,#cs_record['rotation']
            "ego2global_translation":"",# pose_record['translation']
            "ego2global_rotation": "",#pose_record['rotation']
            "timestamp": cam_front_key,}
        
        result.append(info)
        # print(cam_front_key,
        #     cam_front_right_timestamp,
        #     cam_front_left_timestamp,
        #     cam_back_timestamp,
        #     cam_back_right_timestamp,
        #     cam_back_left_timestamp,
        #     gt_timestamp,
        #     ego_timestamp)

    return info
def _find_closest_key(input_dict, target_key):
    # 获取字典中所有键
    all_keys = list(input_dict.keys())

    # 找到与目标键最接近的键
    closest_key = min(all_keys, key=lambda x: abs(x - target_key))

    return closest_key, input_dict[closest_key]

def _pack_cam(cam_dict,type):
    cam = {
        "data_path":cam_dict["header_time"], 
        "type":type, 
        # "sample_data_token",
        # "sensor2ego_translation", 
        # "sensor2ego_rotation", 
        # "ego2global_translation", 
        # "ego2global_rotation", 
        # "timestamp", 
        # "sensor2lidar_rotation", 
        # "sensor2lidar_translation", 
        # "cam_intrinsic"
    }
    return cam

def _get_can_bus_info():
    pass

def _parse_image_data(data_path):
    cam_dict={}

    with open(data_path) as f:
        line = f.readline()
        while line:
            # print(line)
            data = json.loads(line)
            header_time = data['header']['timestamp_sec']
            filename_prefix = long(header_time * 1000000000)

            frame_id = data["frame_id"]
            filename = '{}.jpg'.format(filename_prefix)
            image_sample = {
                "header_time":header_time,
                "sequence_num":data["header"]["sequence_num"],
                "frame_id":data["header"]["frame_id"],
                "file_name":filename,
                "measurement_time": data["measurement_time"]
            }

            cam_dict[header_time]=image_sample
            line = f.readline()

        return cam_dict

def _parse_imu_data(data_path):
    imu_dict={}
    with open(data_path) as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            header_time = data['header']['timestamp_sec']
            sample = {
                "header_time":header_time,
                "measurement_time": data["measurement_time"]
            }
            imu_dict[header_time]=sample
            line = f.readline()
    return imu_dict

def _parse_gps_data(data_path):
    ego_pose={}
    with open(data_path) as f:
        line = f.readline()
        while line:
            data = json.loads(line)
            header_time = data['header']['timestamp_sec']
            sample = {
                "header_time":header_time,
                "measurement_time": data["gnss"]["measurement_time"],
                "lon":data["gnss"]["position"]["lon"],
                "lat":data["gnss"]["position"]["lat"],
                "height":data["gnss"]["position"]["height"],
                "solution_status":data["gnss"]["solution_status"],
                "heading":data["heading"]["heading"],
                "pitch":data["heading"]["pitch"],
                "roll":data["heading"]["roll"],
                "baseline_length":data["heading"]["baseline_length"],
                #"heading":{"measurement_time":1660892075.6,"baseline_length":1.671,"heading":152.718,"pitch":0.916,"roll":0.858}
            }
            ego_pose[header_time]=sample
            line = f.readline()
    return ego_pose

# suteng的真值是车体坐标系 forward +x,left +y,above +z
# nusceces数据标注是全局坐标系，训练时会转到车体坐标系，从资料看 forward +x,left +y,above +z和速腾一样
def _parse_gt_data(data_path):
    gt_dict = {}
    gt_dict_tmp= {}
    #聚合每一帧里的障碍物标注key=stamp_sec,value=[line,line]
    with open(data_path,"r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            stamp_sec = row["stamp_sec"]
            if gt_dict_tmp.__contains__(stamp_sec):
                gt_dict_tmp[stamp_sec].append(row)
            else:
                gt_dict_tmp[stamp_sec]= [row]
        #抽取障碍物信息，填充gt_box,key=stamp_sec,value=[gt_box]
        for stamp_sec, rows in gt_dict_tmp.items():
            gt_boxes = []
            gt_names = []
            gt_velocity = []
            for row in rows:
                gt_box = {
                    "obj_stamp_sec":float(row["obj_stamp_sec"]),
                    "type": int(row["type"]),
	                "type_confidence": float(row["type_confidence"]),
                    "roll": float(row["roll"]),
	                "pitch":float(row["pitch"]),
	                "yaw": float(row["yaw"]),
                    "center_x":float(row["center.x"]),
                    "center_y":float(row["center.y"]),
                    "center_z":float(row["center.z"]),
                    "height":float(row["height"]),
                    "length":float(row["length"]),
                    "width":float(row["width"])

                }
                gt_boxes.append(gt_box)
            gt_dict[float(stamp_sec)]=gt_boxes

    # return gt_dict,gt_dict_tmp
    return gt_dict




def _parse_calibration_data(data_path):
    pass

if __name__ == "__main__":
    # image_data_path="data/dvscenes/sample/apollo_sensor_camera_front_narrow_image_compressed.txt"
    # cam_front_dict = _parse_image_data(image_data_path)
    # for key,value in cam_front_dict.items():
    #     print(key,value)
    #     break
    # ego_data_path = "data/dvscenes/sample/apollo_sensor_gnss_gpfpd.txt"
    # ego_pose_dict = _parse_gps_data(ego_data_path)
    # for key,value in ego_pose_dict.items():
    #     print(key,value)
    #     break

    # gt_data_path = "data/dvscenes/sample/GT.csv"
    # gt_dict = _parse_gt_data(gt_data_path)
    # i = 0
    # for stamp_sec,gtboxes in gt_dict.items():
    #     print(type(stamp_sec),len(gtboxes))
    image_path = ""
    root_path = ""
    out_path = ""
                
    create_dv_infos(image_path,root_path,out_path)