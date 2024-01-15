import os,json,yaml
from scipy.spatial.transform import Rotation
import hashlib

def get_cal(modality,yaml_ext_data,yaml_int_data,result_array,sensor,car_name,tag_name):
        if modality == "camera":
            if tag_name == "front":
                transform = yaml_ext_data["transform"]
            else :
                transform = yaml_ext_data.get("header",{}).get(tag_name, {}).get("transform", {})
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

def get_yaml_data(yaml_path):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def get_calibration_data(data_path,sensor_output_file="",cal_sensor_file="",dump_json=False):
    sensor_meta = []
    cal_meta = []

    modality_dict = {
        "camera":["CAM_FRONT","CAM_BACK","CAM_BACK_LEFT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT"],
        "lidar":["LIDAR_TOP"]
    }

    cam_front_ext = get_yaml_data(os.path.join(data_path,"front_narrow_extrinsics.yaml"))
    cam_front_int = get_yaml_data(os.path.join(data_path,"front_narrow_intrinsics.yaml"))
    cam_front_left_int = get_yaml_data(os.path.join(data_path,"left_front_intrinsics.yaml"))
    cam_front_right_int = get_yaml_data(os.path.join(data_path,"right_front_intrinsics.yaml"))
    cam_back_left_int = get_yaml_data(os.path.join(data_path,"left_rear_intrinsics.yaml"))
    cam_back_right_int = get_yaml_data(os.path.join(data_path,"right_rear_intrinsics.yaml"))

    cam_around_ext = get_yaml_data(os.path.join(data_path,"camera_around_extrinsics.yaml"))
    lidar_ext = get_yaml_data(os.path.join(data_path,"car.yaml"))

    car_name = cam_front_ext['header']['car_name']
    #token = car+sensor
    for modality,sensors in modality_dict.items():
        for sensor in sensors:
            sensor_meta.append({
                        "token": hashlib.sha256((car_name+sensor).encode('utf-8')).hexdigest(),
                        "channel": sensor,
                        "modality": modality
                })
    # token = car+sensor+cal
    get_cal("camera",cam_around_ext,cam_front_left_int,cal_meta,"CAM_FRONT_LEFT",car_name,"left_front")
    get_cal("camera",cam_around_ext,cam_front_right_int,cal_meta,"CAM_FRONT_RIGHT",car_name,"right_front")
    get_cal("camera",cam_around_ext,cam_back_left_int,cal_meta,"CAM_BACK_LEFT",car_name,"left_rear")
    get_cal("camera",cam_around_ext,cam_back_right_int,cal_meta,"CAM_BACK_RIGHT",car_name,"right_rear")
    get_cal("camera",cam_front_ext,cam_front_int,cal_meta,"CAM_FRONT",car_name,"front")
    
    get_cal("lidar",lidar_ext,None,cal_meta,"LIDAR_TOP",car_name,"lidar")

    if dump_json:
        with open(os.path.join(sensor_output_file,"sensor.json"), 'w') as sensor_json_file:
            json.dump(sensor_meta, sensor_json_file,indent=2)


        with open(os.path.join(cal_sensor_file,"calibrated_sensor.json"), 'w') as cal_json_file:
            json.dump(cal_meta, cal_json_file,indent=2)

    cal_dict = {}
    # print(cal_meta)

    for cal in cal_meta:
        cal_dict[cal["token"]]= cal
    return cal_dict