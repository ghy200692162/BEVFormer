
import json,yaml
import csv
import math,os
from numpy import long
import hashlib
from scipy.spatial.transform import Rotation

cam_front_path="data/dvscenes/sample/apollo_sensor_camera_front_narrow_image_compressed.txt"
cam_left_front_path = "data/dvscenes/sample/apollo_sensor_camera_left_front_image_compressed.txt"
cam_left_rear_path = "data/dvscenes/sample/apollo_sensor_camera_left_rear_image_compressed.txt"
cam_right_front_path = "data/dvscenes/sample/apollo_sensor_camera_right_front_image_compressed.txt"
cam_right_rear_path = "data/dvscenes/sample/apollo_sensor_camera_right_rear_image_compressed.txt"
cam_rear_path = "data/dvscenes/sample/apollo_sensor_camera_rear_image_compressed.txt"

ego_data_path = "data/dvscenes/sample/apollo_sensor_gnss_gpfpd.txt"
calibration_data_path = "data/dvscenes/icc/calibration"
gt_data_path = "data/dvscenes/sample/GT.csv"


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
    calibration_dict = {}#_parse_calibration_data(calibration_data_path)
    infos = _fill_trainval_infos(cam_front_dict,
                            cam_front_right_dict,
                            cam_front_left_dict,
                            cam_back_dict,
                            cam_back_right_dict,
                            cam_back_left_dict,ego_dict,gt_dict,calibration_dict,"","")
    data = {
        "infos":infos,
        "metadata":"dv-test"
    }
    # dump to json for debug
    with open("output.json","w") as json_result:
        json.dump(data,json_result)

    return data


#处理单个scene
def _fill_trainval_infos(cam_front_dict,
                           cam_front_right_dict,
                           cam_front_left_dict,
                           cam_back_dict ,
                           cam_back_right_dict,
                           cam_back_left_dict,
                           ego_dict,gt_dict,
                           calibration_dict,
                           scene_name = "",
                           scene_token = ""):

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
    filted_cam_front_dict = OrderedDict(sorted(filted_cam_front_dict.items(), key=lambda x: x[0], reverse=False))
    # obtain 6 image's information per frame
    scene_num=1
    prev_token = ""
    current_token = ""
    next_token = ""

    for index ,(cam_front_key,cam_front_frame) in enumerate(filted_cam_front_dict.items()):
            
        cam_front_right_timestamp ,cam_front_right_frame = _find_closest_key(cam_front_right_dict,cam_front_key)
        cam_front_left_timestamp ,cam_front_left_frame = _find_closest_key(cam_front_left_dict,cam_front_key)
        cam_back_timestamp ,cam_back_frame = _find_closest_key(cam_back_dict,cam_front_key)
        cam_back_right_timestamp ,cam_back_right_frame = _find_closest_key(cam_back_right_dict,cam_front_key)
        cam_back_left_timestamp ,cam_back_left_frame = _find_closest_key(cam_back_left_dict,cam_front_key)
        
        gt_timestamp,gt_boxes = _find_closest_key(gt_dict,cam_front_key)
        ego_timestamp,ego_info = _find_closest_key(ego_dict,cam_front_key)

        
        current_token = hashlib.sha256(cam_front_key)
        # pack cams
        cams = {
            "CAM_FRONT":_pack_cam(cam_front_frame,"CAM_FRONT"),
            "CAM_FRONT_RIGHT":_pack_cam(cam_front_right_frame,"CAM_FRONT_RIGHT"),
            "CAM_FRONT_LEFT":_pack_cam(cam_front_left_frame,"CAM_FRONT_LEFT"),
            "CAM_BACK":_pack_cam(cam_back_frame,"CAM_BACK"),
            "CAM_BACK_LEFT":_pack_cam(cam_back_left_frame,"CAM_BACK_LEFT"),
            "CAM_BACK_RIGHT":_pack_cam(cam_back_right_frame,"CAM_BACK_RIGHT"),
        }

        info = {
            "lidar_path":gt_boxes[0]["lidar_frame_num"] ,#属于同一帧pcd的gt_box，对应的pcd帧号相同，取第一个,
            "gt_timestamp":gt_timestamp,
            "token": current_token,
            "prev":prev_token,
            "next": next_token,
            "can_bus": "",
            "frame_idx": 0,  # temporal related info
            "sweeps": [],
            "cams": cams,
            "scene_token":"",  # temporal related info
            "lidar2ego_translation":[ 0.9138,0.0136,2.092797295600176], #cs_record['translation'] 'lidar2ego_translation': [0.985793, 0.0, 1.84019], 'lidar2ego_rotation': [0.706749235646644, -0.015300993788500868, 0.01739745181256607, -0.7070846669051719], 'ego2global_translation': [600.1202137947669, 1647.490776275174, 0.0], 'ego2global_rotation': [-0.968669701688471, -0.004043399262151301, -0.007666594265959211, 0.24820129589817977]
            "lidar2ego_rotation":[ 0.9988215998157758,0.0014161136772485944,0.049405229353998104,-0.003764120826211623] ,#cs_record['rotation']
            "ego2global_translation":ego_info["pose"],# pose_record['translation']""
            "ego2global_rotation": ego_info["rotation"],#pose_record['rotation']
            "timestamp": cam_front_key,
            "gt_boxes":gt_boxes,
            "ego_info":ego_info
            }
        
        result.append(info)
    for index ,sample in enumerate(result):
        if index == 0:
            sample["next"] = hashlib.sha256(result[1]["timestamp"]).hexdigest()
            continue
        elif index > 0 and index <len(result)-1:
            sample["prev"] = hashlib.sha256(result[index-1]["timestamp"]).hexdigest()
            sample["next"] = hashlib.sha256(result[index+1]["timestamp"]).hexdigest()
            continue
        elif index == len(result):
            sample["prev"] = hashlib.sha256(result[index-1]["timestamp"]).hexdigest()
            continue

    return result


def _pack_cam(cam_dict,type):
    cam = {
        "data_path":cam_dict["file_name"], 
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
            x,y,z = llh_to_xyz(data["gnss"]["position"]["lon"],
                                data["gnss"]["position"]["lat"],
                                data["gnss"]["position"]["height"])

            rotation = to_quat(data["heading"]["heading"],data["heading"]["pitch"],data["heading"]["roll"])
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
                "pos":[x,y,z],
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
                    "obj_stamp_sec":float(row["obj_stamp_sec"]),
                    "type": int(row["type"]),
	                "type_confidence": float(row["type_confidence"]),
                    "yaw": float(row["yaw"]),
                    "roll": float(row["roll"]),#障碍物朝向
	                "pitch":float(row["pitch"]),
                    "center_x":float(row["center.x"]),#障碍物的位置，车体坐标系
                    "center_y":float(row["center.y"]),
                    "center_z":float(row["center.z"]),
                    "height":float(row["height"]),#障碍物的真实尺寸
                    "length":float(row["length"]),
                    "width":float(row["width"]),
                    "lidar_frame_num":row["frame_num"]
                }
                gt_boxes.append(gt_box)
            gt_dict[float(stamp_sec)]=gt_boxes

    # return gt_dict,gt_dict_tmp
    return gt_dict
def _get_cal(modality,yaml_ext_data,yaml_int_data,result_array,sensor,car_name,tag_name):
        if modality == "camera":
            if tag_name == "front":
                transform = yaml_ext_data["transform"]
            else :
                transform = yaml_ext_data.get("header",{}).get(tag_name, {}).get("transform", {})
            print(transform)
            translation = [transform["translation"]['x'],transform["translation"]['y'],transform["translation"]['z']]
            rotation = [transform["rotation"]["x"],transform["rotation"]["y"],transform["rotation"]["z"],transform["rotation"]["w"]]
        elif modality == "lidar":
            lidar_calibration = yaml_ext_data.get('lidar', [])[0].get('lidar_config', {}).get('calibration', {})
            x = lidar_calibration.get('x')
            y = lidar_calibration.get('y')
            z = lidar_calibration.get('z')

            roll = lidar_calibration.get('roll')
            pitch = lidar_calibration.get('pitch')
            yaw = lidar_calibration.get('yaw')
            rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
            translation = [x,y,z]
            rotation = rotation.as_quat().tolist()

        if yaml_int_data is not None:
            camera_intrinsic = yaml_int_data['K']
            width =  yaml_int_data['width']
            height = yaml_int_data['height']
        else:
            camera_intrinsic = []
            width =  0
            height = 0
        result_array.append(
            {
                "token":hashlib.sha256((car_name+sensor+"cal").encode('utf-8')).hexdigest(),
                "sensor_token":hashlib.sha256((car_name+sensor).encode('utf-8')).hexdigest(),
                "translation": translation,
                "rotation":rotation,
                "height" : height,
                "width" : width,
                "camera_intrinsic":[camera_intrinsic[i:i+3] for i in range(0, len(camera_intrinsic), 3)]
            }
        )

def _get_yaml_data(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def _parse_calibration_data(data_path,sensor_output_file,cal_sensor_file):
    sensor_meta = []
    cal_meta = []

    modality_dict = {
        "camera":["CAM_FRONT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT"],
        "lidar":["LIDAR_TOP"]
    }

    cam_front_ext = _get_yaml_data(os.path.join(data_path,"front_narrow_extrinsics.yaml"))
    cam_front_int = _get_yaml_data(os.path.join(data_path,"front_narrow_intrinsics.yaml"))

    cam_front_left_int = _get_yaml_data(os.path.join(data_path,"left_front_intrinsics.yaml"))
    cam_front_right_int = _get_yaml_data(os.path.join(data_path,"right_front_intrinsics.yaml"))
    cam_back_left_int = _get_yaml_data(os.path.join(data_path,"left_rear_intrinsics.yaml"))
    cam_back_right_int = _get_yaml_data(os.path.join(data_path,"right_rear_intrinsics.yaml"))

    cam_around_ext = _get_yaml_data(os.path.join(data_path,"camera_around_extrinsics.yaml"))
    lidar_ext = _get_yaml_data(os.path.join(data_path,"car.yaml"))

    car_name = cam_front_ext['header']['car_name']
    print(car_name)
    #token = car+sensor
    for modality,sensors in modality_dict.items():
        for sensor in sensors:
            sensor_meta.append({
                        "token": hashlib.sha256((car_name+sensor).encode('utf-8')).hexdigest(),
                        "channel": sensor,
                        "modality": modality
                })
    # token = car+sensor+cal
    _get_cal("camera",cam_around_ext,cam_front_left_int,cal_meta,"CAM_FRONT_LEFT",car_name,"left_front")
    _get_cal("camera",cam_around_ext,cam_front_right_int,cal_meta,"CAM_FRONT_RIGHT",car_name,"right_front")
    _get_cal("camera",cam_around_ext,cam_back_left_int,cal_meta,"CAM_BACK_LEFT",car_name,"left_rear")
    _get_cal("camera",cam_around_ext,cam_back_right_int,cal_meta,"CAM_BACK_RIGHT",car_name,"right_rear")
    _get_cal("camera",cam_front_ext,cam_front_int,cal_meta,"CAM_FRONT",car_name,"front")
    _get_cal("lidar",lidar_ext,None,cal_meta,"LIDAR_TOP",car_name,"lidar")

    with open(os.path.join(sensor_output_file,"sensor.json"), 'w') as sensor_json_file:
        json.dump(sensor_meta, sensor_json_file,indent=2)


    with open(os.path.join(cal_sensor_file,"calibrated_sensor.json"), 'w') as cal_json_file:
        json.dump(cal_meta, cal_json_file,indent=2)


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
    cal_data_path = "data/dvscenes/icc/calibration"
    sensor_output_file = "."
    cal_sensor_file ="."
    _parse_calibration_data(cal_data_path,sensor_output_file,cal_sensor_file)