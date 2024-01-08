#生成训练用的pkl数据
#解析好的数据存放在2个位置
#image:/data/dataset/dv_bev/
#其他sensor数据：data/dvscenes/sample/
#标定相关数据：目前确实这部分数据
#annotation:缺失
#info data sheama 
            #    ifno ={'lidar_path': lidar_path,
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
# cams = {
#       'data_path': data_path,
        # 'type': sensor_type,
        # 'sample_data_token': sd_rec['token'],
        # 'sensor2ego_translation': cs_record['translation'],
        # 'sensor2ego_rotation': cs_record['rotation'],
        # 'ego2global_translation': pose_record['translation'],
        # 'ego2global_rotation': pose_record['rotation'],
        # 'timestamp': sd_rec['timestamp']
        # 'sensor2lidar_rotation':
        # 'sensor2lidar_translation':
        #
        #
        #
        #}
import json
import csv
from numpy import long
cam_front_path="data/dvscenes/sample/apollo_sensor_camera_front_narrow_image_compressed.txt"
cam_left_front_path = "data/dvscenes/sample/apollo_sensor_camera_left_front_image_compressed.txt"
cam_left_rear_path = "data/dvscenes/sample/apollo_sensor_camera_left_rear_image_compressed.txt"
cam_right_front_path = "data/dvscenes/sample/apollo_sensor_camera_right_front_image_compressed.txt"
cam_right_rear_path = "data/dvscenes/sample/apollo_sensor_camera_right_rear_image_compressed.txt"
cam_rear_path = "data/dvscenes/sample/apollo_sensor_camera_rear_image_compressed.txt"

ego_data_path = "data/dvscenes/sample/apollo_sensor_gnss_gpfpd.txt"
calibration_data_path = ""
gt_data_path = ""
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




def _fill_trainval_infos(cam_front_dict,
                           cam_front_right_dict,
                           cam_front_left_dict,
                           cam_back_dict ,
                           cam_back_right_dict,
                           cam_back_left_dict,
                           ego_dict,gt_dict,calibration_dict):
    # obtain 6 image's information per frame
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]

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


def _parse_gt_data(data_path):
    gt_dict = {}
    gt_dict_tmp= {}
    with open(data_path,"r") as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            stamp_sec = row["stamp_sec"]
            if gt_dict_tmp.__contains__(stamp_sec):
                gt_dict_tmp[stamp_sec].append(row)
            else:
                 gt_dict_tmp[stamp_sec]= [row]
        for stamp_sec, rows in gt_dict_tmp.items():
            gt_boxes = []
            gt_names = []
            gt_velocity = []
            for row in rows:
                gt_box = {
                    "obj_stamp_sec":row["obj_stamp_sec"],
                    "type": row["type"],
	                "type_confidence": row["type_confidence"],
                    "roll": row["roll"],
	                "pitch":row["pitch"],
	                "yaw": row["yaw"],
                    "center_x":row["center.x"],
                    "center_y":row["center.y"],
                    "center_z":row["center.z"],
                }
            gt_boxes.append(gt_box)
            gt_dict[stamp_sec]=gt_boxes
            
    return gt_dict,gt_dict_tmp

def _get_bbox():
    pass



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

    gt_data_path = "data/dvscenes/sample/GT.csv"
    _,_gt_tmp = _parse_gt_data(gt_data_path)
    print(len(_gt_tmp["1660892076.450032"]))
    # for gt in _gt_tmp:
    #     print(gt)
    #     break