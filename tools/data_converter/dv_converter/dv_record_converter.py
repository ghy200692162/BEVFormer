
import json,yaml
import csv
import math,os
from numpy import long
import numpy as np
import hashlib
from scipy.spatial.transform import Rotation
from collections import OrderedDict
import mmcv
from os import path as osp
from dv_utils import *
from parse_cal import *

os.environ['PYTHONPATH'] = "tools/data_converter/dv_converter"


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

dv_category = []

cam_front_path="data/dvscenes_org/sample/apollo_sensor_camera_front_narrow_image_compressed.txt"
cam_left_front_path = "data/dvscenes_org/sample/apollo_sensor_camera_left_front_image_compressed.txt"
cam_left_rear_path = "data/dvscenes_org/sample/apollo_sensor_camera_left_rear_image_compressed.txt"
cam_right_front_path = "data/dvscenes_org/sample/apollo_sensor_camera_right_front_image_compressed.txt"
cam_right_rear_path = "data/dvscenes_org/sample/apollo_sensor_camera_right_rear_image_compressed.txt"
cam_rear_path = "data/dvscenes_org/sample/apollo_sensor_camera_rear_image_compressed.txt"

ego_data_path = "data/dvscenes_org/sample/apollo_sensor_gnss_gpfpd.txt"
calibration_data_path = "data/dvscenes/icc/calibration"
gt_data_path = "data/dvscenes_org/sample/GT.csv"


# def get_cal(modality,yaml_ext_data,yaml_int_data,result_array,sensor,car_name,tag_name):
#         if modality == "camera":
#             if tag_name == "front":
#                 transform = yaml_ext_data["transform"]
#             else :
#                 transform = yaml_ext_data.get("header",{}).get(tag_name, {}).get("transform", {})
#             translation = [transform["translation"]['x'],transform["translation"]['y'],transform["translation"]['z']]
#             rotation = [transform["rotation"]["x"],transform["rotation"]["y"],transform["rotation"]["z"],transform["rotation"]["w"]]
#         elif modality == "lidar":
#             lidar_calibration = yaml_ext_data.get('lidar', [])[0].get('lidar_config', {}).get('calibration', {})
#             x = lidar_calibration.get('x')
#             y = lidar_calibration.get('y')
#             z = lidar_calibration.get('z')

#             roll = lidar_calibration.get('roll')
#             pitch = lidar_calibration.get('pitch')
#             yaw = lidar_calibration.get('yaw')
#             rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
#             translation = [x,y,z]
#             rotation = rotation.as_quat().tolist()

#         if yaml_int_data is not None:
#             camera_intrinsic = yaml_int_data['K']
#             width =  yaml_int_data['width']
#             height = yaml_int_data['height']
#         else:
#             camera_intrinsic = []
#             width =  0
#             height = 0
#         result_array.append(
#             {
#                 "token":hashlib.sha256((car_name+sensor+"cal").encode('utf-8')).hexdigest(),
#                 "sensor_token":hashlib.sha256((car_name+sensor).encode('utf-8')).hexdigest(),
#                 "translation": translation,
#                 "rotation":rotation,
#                 "height" : height,
#                 "width" : width,
#                 "camera_intrinsic":[camera_intrinsic[i:i+3] for i in range(0, len(camera_intrinsic), 3)]
#             }
#         )

# def get_yaml_data(yaml_path):
#     with open(yaml_path, 'r') as file:
#         yaml_data = yaml.safe_load(file)
#     return yaml_data


# def compute_sensor_to_lidar(sensor2ego, lidar2ego):
#     """
#     计算 sensor 到 lidar 的外参

#     参数：
#     - sensor2ego: sensor 到 ego 的外参，[translation, rotation]
#     - lidar2ego: lidar 到 ego 的外参，[translation, rotation]

#     返回值：
#     - sensor2lidar: sensor 到 lidar 的外参，[translation, rotation]
#     """

#     translation_sensor, rotation_sensor = sensor2ego
#     translation_lidar, rotation_lidar = lidar2ego

#     # 构建旋转矩阵
#     rotation_matrix_sensor = Rotation.from_quat(rotation_sensor).as_matrix()
#     rotation_matrix_lidar = Rotation.from_quat(rotation_lidar).as_matrix()

#     # 构建变换矩阵
#     sensor2ego_matrix = np.eye(4)
#     sensor2ego_matrix[:3, :3] = rotation_matrix_sensor
#     sensor2ego_matrix[:3, 3] = translation_sensor

#     lidar2ego_matrix = np.eye(4)
#     lidar2ego_matrix[:3, :3] = rotation_matrix_lidar
#     lidar2ego_matrix[:3, 3] = translation_lidar

#     # 计算 sensor 到 lidar 的变换矩阵
#     sensor2lidar_matrix = np.dot(sensor2ego_matrix, np.linalg.inv(lidar2ego_matrix))

#     # 提取平移和旋转
#     translation_sensor2lidar = sensor2lidar_matrix[:3, 3]
#     rotation_sensor2lidar = Rotation.from_matrix(sensor2lidar_matrix[:3, :3]).as_quat()

#     return [translation_sensor2lidar,sensor2lidar_matrix[:3, :3]]
################################################################################################

################################################################################################

###############################################################################################


# def get_calibration_data(data_path,sensor_output_file="",cal_sensor_file="",dump_json=False):
#     sensor_meta = []
#     cal_meta = []

#     modality_dict = {
#         "camera":["CAM_FRONT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT"],
#         "lidar":["LIDAR_TOP"]
#     }

#     cam_front_ext = get_yaml_data(os.path.join(data_path,"front_narrow_extrinsics.yaml"))
#     cam_front_int = get_yaml_data(os.path.join(data_path,"front_narrow_intrinsics.yaml"))
#     cam_front_left_int = get_yaml_data(os.path.join(data_path,"left_front_intrinsics.yaml"))
#     cam_front_right_int = get_yaml_data(os.path.join(data_path,"right_front_intrinsics.yaml"))
#     cam_back_left_int = get_yaml_data(os.path.join(data_path,"left_rear_intrinsics.yaml"))
#     cam_back_right_int = get_yaml_data(os.path.join(data_path,"right_rear_intrinsics.yaml"))

#     cam_around_ext = get_yaml_data(os.path.join(data_path,"camera_around_extrinsics.yaml"))
#     lidar_ext = get_yaml_data(os.path.join(data_path,"car.yaml"))

#     car_name = cam_front_ext['header']['car_name']
#     #token = car+sensor
#     for modality,sensors in modality_dict.items():
#         for sensor in sensors:
#             sensor_meta.append({
#                         "token": hashlib.sha256((car_name+sensor).encode('utf-8')).hexdigest(),
#                         "channel": sensor,
#                         "modality": modality
#                 })
#     # token = car+sensor+cal
#     get_cal("camera",cam_around_ext,cam_front_left_int,cal_meta,"CAM_FRONT_LEFT",car_name,"left_front")
#     get_cal("camera",cam_around_ext,cam_front_right_int,cal_meta,"CAM_FRONT_RIGHT",car_name,"right_front")
#     get_cal("camera",cam_around_ext,cam_back_left_int,cal_meta,"CAM_BACK_LEFT",car_name,"left_rear")
#     get_cal("camera",cam_around_ext,cam_back_right_int,cal_meta,"CAM_BACK_RIGHT",car_name,"right_rear")
#     get_cal("camera",cam_front_ext,cam_front_int,cal_meta,"CAM_FRONT",car_name,"front")
    
#     get_cal("lidar",lidar_ext,None,cal_meta,"LIDAR_TOP",car_name,"lidar")

#     if dump_json:
#         with open(os.path.join(sensor_output_file,"sensor.json"), 'w') as sensor_json_file:
#             json.dump(sensor_meta, sensor_json_file,indent=2)


#         with open(os.path.join(cal_sensor_file,"calibrated_sensor.json"), 'w') as cal_json_file:
#             json.dump(cal_meta, cal_json_file,indent=2)

#     cal_dict = {}
#     # print(cal_meta)

#     for cal in cal_meta:
#         cal_dict[cal["token"]]= cal
#     return cal_dict



def create_dv_data(image_path,
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
    calibration_dict = get_calibration_data(data_path=calibration_data_path) 
    car_name = "zhaojun"
    infos = _fill_trainval_infos(car_name,
                            cam_front_dict,
                            cam_front_right_dict,
                            cam_front_left_dict,
                            cam_back_dict,
                            cam_back_right_dict,
                            cam_back_left_dict,ego_dict,gt_dict,calibration_dict,"","")
    train_infos ,val_infos = split_list(infos)
    train_data = {
        "infos":train_infos,
        "metadata":{
            "version":"dv-train"
        }
    }

    val_data = {
        "infos":val_infos,
        "metadata":{
            "version":"dv-val"
        }
    }
    info_prefix = "dvscenes"
    train_info_path = osp.join(out_path,
                             '{}_infos_temporal_train.pkl'.format(info_prefix))
                    
    val_info_path = osp.join(out_path,
                             '{}_infos_temporal_val.pkl'.format(info_prefix))
    
    mmcv.dump(train_data, train_info_path)
    mmcv.dump(val_data, val_info_path)

    # dump to json for debug
    # with open("output.json","w") as json_result:
    #     json.dump(train_infos,json_result)


    return infos


#处理单个scene
def _fill_trainval_infos(car_name,
                            cam_front_dict,
                           cam_front_right_dict,
                           cam_front_left_dict,
                           cam_back_dict ,
                           cam_back_right_dict,
                           cam_back_left_dict,
                           ego_dict,
                           gt_dict,
                           calibration_dict,
                           scene_name = "",
                           scene_token = ""):

    result = []
    #filter front camera
    # lidar 10hz,camera 20hz,一帧gt ,可以匹配到2帧数据,取时间差小的数据
    filted_cam_front_dict={}
    flag_gt_dict = {}
    for gt_timestamp in gt_dict.keys():
        candidated_cam_front_key ,candidated_frame = find_closest_key(cam_front_dict,gt_timestamp)
        if flag_gt_dict.__contains__(gt_timestamp):
            stored_frame = flag_gt_dict[gt_timestamp]
            stored_timestamp = stored_frame["header_time"]
            if abs(stored_timestamp-gt_timestamp) > abs(candidated_cam_front_key-gt_timestamp):
                filted_cam_front_dict[candidated_cam_front_key] = candidated_frame
        else:
            flag_gt_dict[gt_timestamp] = candidated_frame
            filted_cam_front_dict[candidated_cam_front_key] = candidated_frame
    #对每一组序列，降序排列，保证时间序列有效
    filted_cam_front_dict = OrderedDict(sorted(filted_cam_front_dict.items(), key=lambda x: x[0], reverse=False))
    # obtain 6 image's information per frame
    prev_token = ""
    current_token = ""
    next_token = ""

    sence_meta = []
    sample_meta = []
    sample_data_meta = []
    log_meta = []
    isinstance_meta = []
    ego_meta = []

    
    # 
    for index ,(cam_front_key,cam_front_frame) in enumerate(filted_cam_front_dict.items()):
            
        cam_front_right_timestamp ,cam_front_right_frame = find_closest_key(cam_front_right_dict,cam_front_key)
        cam_front_left_timestamp ,cam_front_left_frame = find_closest_key(cam_front_left_dict,cam_front_key)
        cam_back_timestamp ,cam_back_frame = find_closest_key(cam_back_dict,cam_front_key)
        cam_back_right_timestamp ,cam_back_right_frame = find_closest_key(cam_back_right_dict,cam_front_key)
        cam_back_left_timestamp ,cam_back_left_frame = find_closest_key(cam_back_left_dict,cam_front_key)
        
        gt_timestamp,gt_boxes = find_closest_key(gt_dict,cam_front_key)
        ego_timestamp,ego_info = find_closest_key(ego_dict,cam_front_key)

        lidar_token = hashlib.sha256((car_name+"LIDAR_TOP"+"cal").encode('utf-8')).hexdigest()
        lidar2ego_cal= calibration_dict[lidar_token]
        lidar_path = gt_boxes[0]["lidar_frame_num"] ,#属于同一帧pcd的gt_box，对应的pcd帧号相同，取第一个,
        gt_names = [gt["type"] for gt in gt_boxes]
        num_lidar_pts = [gt["lidar_pts"] for gt in gt_boxes]
        current_token = hashlib.sha256(str(cam_front_key).encode('utf-8')).hexdigest()
        # pack cams
        cams = {
                        #cam_dict,type,cal_dict,ego_dict
            # "CAM_FRONT":        _pack_cam(car_name,cam_front_frame,"CAM_FRONT",calibration_dict,ego_info),
            "CAM_FRONT_RIGHT":  _pack_cam(car_name,cam_front_right_frame,"CAM_FRONT_RIGHT",calibration_dict,ego_info),
            "CAM_FRONT_LEFT":   _pack_cam(car_name,cam_front_left_frame,"CAM_FRONT_LEFT",calibration_dict,ego_info),
            # "CAM_BACK":         _pack_cam(car_name,cam_back_frame,"CAM_BACK",calibration_dict,ego_info),  #缺少后视的标定
            "CAM_BACK_LEFT":    _pack_cam(car_name,cam_back_left_frame,"CAM_BACK_LEFT",calibration_dict,ego_info),
            "CAM_BACK_RIGHT":   _pack_cam(car_name,cam_back_right_frame,"CAM_BACK_RIGHT",calibration_dict,ego_info),
        }
        gt_boxes = [[gt["center_x"],gt["center_y"],gt["center_z"],gt["height"],gt["width"],gt["length"],gt["yaw"]] for gt in gt_boxes]
        info = {
            "lidar_path":lidar_path,
            "gt_timestamp":gt_timestamp,
            "token": current_token,
            "prev":prev_token,
            "next": next_token,
            "can_bus": np.array(_get_can_bus_info()),
            "frame_idx":index,  # temporal related info
            "sweeps": [],
            "cams": cams,
            "scene_token":"",  # temporal related info
            "lidar2ego_translation":lidar2ego_cal["translation"],
            "lidar2ego_rotation":lidar2ego_cal["rotation"],
            "ego2global_translation":ego_info["pose"],
            "ego2global_rotation": ego_info["rotation"],
            "timestamp": cam_front_key,
            "gt_boxes":np.array(gt_boxes),
            "gt_names":np.array(gt_names),
            "ego_info":ego_info,
            "num_lidar_pts":np.array(num_lidar_pts)
            }
        # print(info["gt_boxes"].shape)
        
        result.append(info)
    for index ,sample in enumerate(result):
        if index == 0:
            sample["next"] = hashlib.sha256(str(result[1]["timestamp"]).encode('utf-8')).hexdigest()
            continue
        elif index > 0 and index <len(result)-1:
            sample["prev"] = hashlib.sha256(str(result[index-1]["timestamp"]).encode('utf-8')).hexdigest()
            sample["next"] = hashlib.sha256(str(result[index+1]["timestamp"]).encode('utf-8')).hexdigest()
            continue
        elif index == len(result):
            sample["prev"] = hashlib.sha256(str(result[index-1]["timestamp"]).encode('utf-8')).hexdigest()
            continue

    return result
# _pack_cam(car_name,cam_front_frame,"CAM_FRONT",calibration_dict,ego_info),


def _pack_cam(car_name,cam_dict,sensor,cal_dict,ego_dict):
    data_path=os.path.join("/data/dataset/dv_bev/mini-1.0",sensor)
    cal_token = hashlib.sha256((car_name+sensor+"cal").encode('utf-8')).hexdigest()
    lidar_token = hashlib.sha256((car_name+"LIDAR_TOP"+"cal").encode('utf-8')).hexdigest()

    cam_cal = cal_dict[cal_token]
    lidar_cal = cal_dict[lidar_token]
    cam2ego = [cam_cal["translation"],cam_cal["rotation"]]
    lidar2ego = [lidar_cal["translation"],lidar_cal["rotation"]]
    cam2lidar = compute_sensor_to_lidar(cam2ego,lidar2ego)
    cam = {
        "data_path":os.path.join(data_path,cam_dict["file_name"]), 
        "type":sensor, 
        "sample_data_token" : "",
        "sensor2ego_translation":cam_cal["translation"],
        "sensor2ego_rotation":cam_cal["rotation"],
        "ego2global_translation":ego_dict["pose"],
        "ego2global_rotation":ego_dict["rotation"],
        "timestamp":cam_dict["header_time"], # 如果需要真实接近lidar的时间，需要换成cam_dict["measurement_time"]
        "sensor2lidar_rotation":cam2lidar[1],
        "sensor2lidar_translation":cam2lidar[0], 
        "cam_intrinsic":np.array(cam_cal["camera_intrinsic"])
    }
    return cam

def _get_can_bus_info():
    can_bus = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.]
    return can_bus

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
            x,y,z = llh_to_xyz(data["gnss"]["position"]["lon"],
                                data["gnss"]["position"]["lat"],
                                data["gnss"]["position"]["height"])

            rotation = to_quat(data["heading"]["heading"],data["heading"]["pitch"],data["heading"]["roll"]).tolist()
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
                "pose":[x,y,z],
                "rotation":rotation,
                "baseline_length":data["heading"]["baseline_length"],
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
                    "lidar_time_sec":float(stamp_sec),
                    "obj_stamp_sec":float(row["obj_stamp_sec"]),
                    "type": "pedestrian",#int(row["type"]),临时让程序可以跑通
	                "type_confidence": float(row["type_confidence"]),
                    "yaw": float(row["yaw"]),
                    "roll": float(row["roll"]),#障碍物朝向
	                "pitch":float(row["pitch"]),
                    "center_x":float(row["center.x"]),#障碍物的位置，车体坐标系 速腾原始数据是 前X，左Y
                    "center_y":float(row["center.y"]),
                    "center_z":float(row["center.z"]),
                    "height":float(row["height"]),#障碍物的真实尺寸
                    "length":float(row["length"]),
                    "width":float(row["width"]),
                    "lidar_frame_num":row["frame_num"],
                    "lidar_pts":5
                }
                gt_boxes.append(gt_box)
            gt_dict[float(stamp_sec)]=gt_boxes

    # return gt_dict,gt_dict_tmp
    return gt_dict

#生成 category.json
# output_data_path:输出json文件的路径
# instance_meta: 存放instance数据
# gt_dict:真值数据
def _get_instance(output_data_path,instance_meta,gt_dict):
    # 对真值按照时间戳进行排序
    gt_dict = OrderedDict(sorted(gt_dict.items(), key=lambda x: x[0], reverse=False))

def _get_category(output_data_path):
    import random
    category_meta = [
        {
            "token": "1fa93b757fc74fb197cdd60001ad8abf",
            "name": "human.pedestrian.adult",
            "description": "Adult subcategory."
        }]

    with open(os.path.join(output_data_path,"category.json"), 'w') as sensor_json_file:
        json.dump(category_meta, sensor_json_file,indent=2)
    return random.choice(category_meta)



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
    # image_path = ""
    # root_path = ""
    # out_path = ""
                
    # dataset = create_dv_data(image_path,root_path,out_path)
    # infos = dataset["infos"]
    # for info in infos:
    #     with open('0_debug.json', 'w') as file:
    #         json.dump(info, file)
    #     break

    # # # test 生成 sensor.json,calibration_sensor.json
    # cal_data_path = "data/dvscenes/icc/calibration"
    # sensor_output_file = "."
    # cal_sensor_file ="."
    
    # cal_dict = get_calibration_data(data_path=cal_data_path)
    # print(cal_dict)


    # 生成 sence.json,sample.json,sample_data.json
    image_path = ""
    root_path = ""
    out_path = ""

    infos = create_dv_data(image_path,root_path,out_path)
    for info in infos :
        print(info)
        break



