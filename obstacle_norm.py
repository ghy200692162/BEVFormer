import csv
import sys
import os

DEBUG = False
# 真值原点：本车后轴中点

if DEBUG: sys.path.append('../')
from model.obstacle import Obstacle
from collections import OrderedDict, Counter
import numpy as np
from data_norm.lane_norm import dut_stamp
import copy
from scipy.interpolate import interp1d
import ujson
from pyquaternion import Quaternion
import math

def find_2_nearest_timeStamp(inputStampSeq,targetStamp):
    stampSeqToFind = copy.deepcopy(inputStampSeq)
    nearest_top1_idx = np.argmin(np.abs(np.array(stampSeqToFind) - np.array(targetStamp)))
    timeStamp1 = stampSeqToFind[nearest_top1_idx]
    stampSeqToFind.pop(nearest_top1_idx)
    nearest_top2_idx = np.argmin(np.abs(np.array(stampSeqToFind) - np.array(targetStamp)))
    timeStamp2 = stampSeqToFind[nearest_top2_idx]
    stamps2 = sorted([timeStamp1, timeStamp2])
    return stamps2

def interp_(inputXseq,inputYseq,queryX):
    interp = interp1d(x=inputXseq, y=inputYseq, fill_value='extrapolate', axis=0)
    return interp(queryX)

def obstacle_suteng_csv_normalization(obstacle_csv_path):
    """
    transform suteng zhenzhi csv to Standardized dataframe format
    :param obstacle_csv_path:
    :return: List[List[]]

    suteng car coordinates (vertical view): forward +x, left +y, above +z, roll / pitch / yaw
    didi car coordinates (vertical view): forward +y, right +x, above +z, pitch / roll / yaw

    """
    assert os.path.exists(obstacle_csv_path), '%s not exist!' % obstacle_csv_path
    # load data
    with open(obstacle_csv_path, 'r') as fns:
        reader = csv.reader(fns)
        all_rows = [row for row in reader]
        # ['stamp_sec', 'frame_num', 'id', 'track_id', 'center.x', 'center.y', 'center.z', 'velocity.x', 'velocity.y', 'velocity.z',
        # 'acceleration.x', 'acceleration.y', 'acceleration.z', 'angular_velocity', 'direction.x', 'direction.y', 'direction.z',
        # 'height', 'length', 'width', 'is_radar_matching', 'is_tracked', 'radar_velocity.x', 'radar_velocity.y', 'radar_velocity.z',
        # 'type', 'type_confidence', 'pose.pos.x', 'pose.pos.y', 'pose.pos.z', 'roll', 'pitch', 'yaw', 'car_twist'],
        '''
        used_keys = ['stamp_sec', 'frame_num', 'id', 'track_id', 'center.x', 'center.y', 'center.z', 'velocity.x',
                     'velocity.y', 'velocity.z',
                     'acceleration.x', 'acceleration.y', 'acceleration.z', 'angular_velocity', 'direction.x',
                     'direction.y', 'direction.z',
                     'height', 'length', 'width', 'is_tracked', 'type', 'type_confidence']
        for uk in used_keys:
            assert uk in csv_keys, 'data key %s not exist in csv file !' % uk
        '''
        
        csv_keys = all_rows[0]
        csv_value = all_rows[1:]
        # print(len(csv_value), 'obstacles')
        key_num, data_num = len(csv_keys), len(csv_value)
        csv_data = OrderedDict()
        for row_idx, row_data in enumerate(csv_value):
            if row_idx == 0:
                csv_data = {csv_keys[col_idx]: [row_data[col_idx]] for col_idx in range(key_num)}
            else:
                for col_idx in range(key_num):
                    csv_data[csv_keys[col_idx]].append(row_data[col_idx])

    # transformation
    gt_for_dataset = []

    frames_id_and_num = dict(Counter(csv_data['frame_num']))
    frames_num = len(frames_id_and_num)
    oidx = 0  # overall index
    for frame_idx in range(frames_num):
        gt_for_img = []
        frame_ids = list(frames_id_and_num.keys())
        frame_id = frame_ids[frame_idx]
        obstacles_num = frames_id_and_num[frame_id]
        for obs_idx in range(obstacles_num):
            obs = Obstacle()
            obs.sensor_id = None  # 传感器id
            # obs.frame_id = int(csv_data['frame_num'][oidx])  # 帧id
            obs.frame_id = int(frame_id)  # 帧id
            # print('overall index',oidx, 'frame idx',frame_idx, 'obj idx', obs_idx, 'obstacle num', obstacles_num,
            #       'frames num', frames_num, 'frame id',frame_id)
            
            if "obj_stamp_sec" in csv_data:
                obs.stamp = int(float(csv_data['obj_stamp_sec'][oidx]) * 1000) # 20220420 for new version of suteng zhenzhi system
            else:
                obs.stamp = int(float(csv_data['stamp_sec'][oidx]) * 1000)  # 帧时间戳

            obs.id = int(csv_data['id'][oidx])  # 障碍物id
            obs.track_id = int(csv_data['track_id'][oidx])  # 障碍物跟踪id
            obs.is_tracked = bool(csv_data['is_tracked'][oidx])  # 代表是否被跟踪
            obs.age = None  # 障碍物的生命周期
            obs.type = int(csv_data['type'][oidx])  # 障碍物类型 0 unknown?1 person?2 rider?3 small car?4 big car
            obs.sub_type = None
            obs.position = np.array([-1 * float(csv_data['center.y'][oidx]), float(csv_data['center.x'][oidx]),
                                     float(csv_data['center.z'][oidx])])  # 障碍物的相对坐标（车体坐标系）
            obs.point_pos = 0  # 障碍物测量点在车身位置（如车尾中心、车中心等）
            obs.image_pos = None  # 图像感知框的左上角和右下角坐标（图像坐标系）
            obs.size = np.array([float(csv_data['length'][oidx]), float(csv_data['width'][oidx]),
                                 float(csv_data['height'][oidx])])  # 障碍物的真实尺寸
            obs.move_status = int(csv_data["move_status"][oidx]) # 障碍物运动状态，1运动，0静止，-1停止

            # input is unit vector，converted to angle value
            # obs.rotation = np.array([float(csv_data['direction.y'][oidx]), float(csv_data['direction.x'][oidx]),
            #                          float(csv_data['direction.z'][oidx])])
            y, x, z = float(csv_data['direction.y'][oidx]), float(csv_data['direction.x'][oidx]), float(csv_data['direction.z'][oidx])

            # fix divide 0 error, suteng coord system
            # pitch, roll, yaw = np.arctan(z / x), np.arctan(-1 * z / y), np.arctan(y / x)
            # -pi <= angle <= pi
            if x != 0:
                pitch = np.arctan(z / x)
            else:
                if z > 0:
                    pitch = math.pi / 2
                else:
                    pitch = -1 * math.pi / 2
            
            if y != 0:
                roll = np.arctan(-1 * z / y)
            else:
                if z > 0:
                    roll = math.pi / 2
                else:
                    roll = -1 * math.pi / 2
            
            # 按速腾坐标系计算yaw角：direction.x是朝前，direction.y朝左。
            # https://cooper.didichuxing.com/docs/document/2199341930262
            # x
            # ^ 
            # | 象限1
            # ---> -y
            if x != 0:
                if x>= 0:
                    # 一、二象限不变，符合速腾朝向定义，偏右为负，偏左为正
                    yaw = np.arctan(y / x)
                else:
                    # 三、四象限特殊处理
                    if y >= 0:
                        yaw = math.pi + np.arctan(y / x)
                    else:
                        yaw = -math.pi + np.arctan(y / x)
            else:
                if y > 0:
                    yaw = math.pi / 2
                else:
                    yaw = -1 * math.pi / 2
            
            # print('yaw=',math.degrees(yaw))
            
            obs.rotation = np.array([float(pitch), float(roll), float(yaw)])  # 障碍物的相对朝向（车体坐标系）

            obs.linear_twist = np.array([-1 * float(csv_data['velocity.y'][oidx]), float(csv_data['velocity.x'][oidx]),
                                         float(csv_data['velocity.z'][oidx])])  # 障碍物的相对线速度（车体坐标系）
            obs.angular_twist = np.array(float(csv_data['angular_velocity'][oidx]))  # 障碍物的相对角速度（车体坐标系）
            obs.acceleration = np.array(
                [-1 * float(csv_data['acceleration.y'][oidx]), float(csv_data['acceleration.x'][oidx]),
                 float(csv_data['acceleration.z'][oidx])])  # 障碍物的相对加速度（车体坐标系）
            
            obs.cipv = False
            if 'cipv' in csv_data:
                if int(csv_data['cipv'][oidx]) == 1:
                    obs.cipv = True  # 目标是否为cipv
                
                if int(csv_data['cipv'][oidx]) == 5: # 速腾的真值 5对应 左一 （玉良定义的3）
                    obs.potential_cipv = 3
                elif int(csv_data['cipv'][oidx]) == 4: # 对应4
                    obs.potential_cipv = 4
                else:
                    obs.potential_cipv = int(csv_data['cipv'][oidx]) - 1
                    
                # print(csv_data['cipv'][oidx])
                # print('obs.potential_cipv',obs.potential_cipv)
            else:
                obs.potential_cipv = -1
                
            # 目标是否要cutin
            if 'cut_in' in csv_data:
                if int(csv_data['cut_in'][oidx]) == 1:
                    obs.cut_in_flag = True
                else:
                    obs.cut_in_flag = False
            obs.pose_cov = None  # 速度和角速度对角线方差
            obs.twist_cov = None  # 障碍物置信度
            obs.acceleration_cov = None
            obs.certainty = None  # 障碍物类别置信度
            obs.type_certainty = float(csv_data['type_confidence'][oidx])  # 障碍物类别置信度
            lane_attribute = int(csv_data['lane_id'][oidx])
            if lane_attribute != -9:
                lane_attribute = -lane_attribute
            obs.lane_attribute = lane_attribute
            
            """
            多属性字段，获取该字段的值，并以2进制表示，由低位到高位，b0为cut out驾驶行为定义，b1和b2是动静态（01静止,10运动,11停止),
            b3和b4是potential_cipv判断（01 potential_cipv，10unknow，11非potential_cipv）。
            解析时，请从低位到高位获取，从低位到高位获取可避免解析程序的失效。
            """
            obs.label = 0b00000000000000000000000000000000 # valid
            obs.valid_info = 0b00000001100000011111011010111110  # 该字段的值以2进制表示，由低位到高位，第0位表示上面定义的第一个属性是否有效，以此类推
            gt_for_img.append(obs)
            oidx += 1
        
        if DEBUG: print(f'{frame_idx} / {frames_num}')
        gt_for_dataset.append(gt_for_img)

        # sort the frames based on the timestamp
        gt_obstacles = sorted(gt_for_dataset, key=lambda data_list: data_list[0].stamp, reverse=False)

    return gt_obstacles

def dut_simulate_noise_obstacle_normalization(obstacle_csv_path):
    '''
    input： suteng zhenzhi csv file
    output: simulated list<list<obstacle>> and added noise
    '''

    gt_frames_obstalce_list = obstacle_suteng_csv_normalization(obstacle_csv_path)
    gt_timeStamp_seq = [frames[0].stamp for frames in gt_frames_obstalce_list]
    gt_timeStamp_obs = {frames[0].stamp: frames for frames in gt_frames_obstalce_list}
    ### 仿真时间戳
    dut_timeStamp_seq = dut_stamp(gt_timeStamp_seq)

    ### use interpolation to generate object for testing
    # dut_timeStamp_obs = copy.deepcopy(gt_timeStamp_obs)
    if DEBUG: print('dut time stamp', len(dut_timeStamp_seq))
    dut_timeStamp_obs = {}

    # generating dut obstalces...
    # for dut_timeStamp in tqdm(dut_timeStamp_seq[:10]):
    for stampIdx,dut_timeStamp in enumerate(dut_timeStamp_seq):
        # find nearest two stamps
        stamps2 = find_2_nearest_timeStamp(inputStampSeq=gt_timeStamp_seq,targetStamp=dut_timeStamp)
        startStamp, endStamp = stamps2
        # if DEBUG: print('%s/%s' % (stampIdx + 1, len(dut_timeStamp_seq)), startStamp, dut_timeStamp, endStamp)

        # generate source value list for interpolation
        # positions, sizes, rotations, linear_twists, angular_twists, accelerations = [],[],[],[],[],[]
        miu, sigma = 0.1, 0.05
        noise = np.random.normal(miu, sigma, (1, 1))[0]

        gt_trackId_obs1 = {ob.track_id: ob for ob in gt_timeStamp_obs[startStamp] if ob.is_tracked}
        gt_trackId_obs2 = {ob.track_id: ob for ob in gt_timeStamp_obs[endStamp] if ob.is_tracked}

        trackIds_to_interpolate = set(gt_trackId_obs1).intersection(gt_trackId_obs2)
        # obs_to_interpolate a= {ob.track_id:ob for ob in gt_timeStamp_obs[startStamp] if ob.track_id in trackIds_to_interpolate}
        dut_trackId_obs = {}
        for track_id in trackIds_to_interpolate:
            ob1, ob2 = gt_trackId_obs1[track_id], gt_trackId_obs2[track_id]
            ob_to_interpolate = copy.deepcopy(ob1)

            ob_to_interpolate.stamp = int(dut_timeStamp)
            ob_to_interpolate.frame_id = int(stampIdx)

            ob_to_interpolate.position = interp_(stamps2,[ob1.position, ob2.position],dut_timeStamp) + noise
            ob_to_interpolate.size = interp_(stamps2,[ob1.size, ob2.size],dut_timeStamp) + noise
            ob_to_interpolate.rotation = interp_(stamps2,[ob1.rotation, ob2.rotation],dut_timeStamp) + noise
            ob_to_interpolate.linear_twist = interp_(stamps2,[ob1.linear_twist, ob2.linear_twist],dut_timeStamp) + noise
            ob_to_interpolate.angular_twist = interp_(stamps2,[ob1.angular_twist, ob2.angular_twist],dut_timeStamp) + noise
            ob_to_interpolate.acceleration = interp_(stamps2,[ob1.acceleration, ob2.acceleration],dut_timeStamp) + noise

            dut_trackId_obs[track_id] = ob_to_interpolate
        dut_timeStamp_obs[dut_timeStamp] = dut_trackId_obs

    dut_obstacles = [[dut_ob for trackId, dut_ob in track_id_obs.items()] for dutStamp, track_id_obs in dut_timeStamp_obs.items()]
    if DEBUG: print('generated frames', len(dut_obstacles))
    return dut_obstacles

def obstacle_apollo_json_normalization(obstacle_json_path, odometry_json_path=None):
    ## CyberRT 转标准化数据
    """
    :param obstacle_json_path: json的文件地址
    :param stamp_seq: 时间戳，输入为空列表或者None，为None则不修改，为list则进行修改
    :return: obstacle 标准格式

    apollo car coordinates (vertical view): forward +y, right +x, above +z, pitch / roll / yaw
    same as didi car coordinates system
    """
    if DEBUG: print(obstacle_json_path,odometry_json_path)
    assert os.path.exists(obstacle_json_path), '%s not exist!' % obstacle_json_path
    with open(obstacle_json_path, 'r') as f:
        all_frames_data = f.readlines()
        perception_result = ujson.loads(all_frames_data[0])
        # print(perception_result)

    # transform perception results to data frame format
    data_frame_result = []
    obstacle_keys = {'id':1, 'position':1, 'theta':1, 'velocity':1, 'length':1, 'width':1, 'height':1, 'tracking_time':1, 'type':0, 'timestamp':0,
            'acceleration':1, 'anchor_point':0, 'bbox2d':1, 'sub_type':1, 'height_above_ground':0, 'position_covariance':1,
            'velocity_covariance':1, 'acceleration_covariance':0}
    frame_keys = {'timestamp_sec':0, 'module_name':0, 'sequence_num':1, 'lidar_timestamp':0, 'camera_timestamp':1, 'frame_id':0}
    

    # current json type: (parent_type, child_type) which it belongs
    type_map = {'ST_UNKNOWN':('unknown','unknown'), 'ST_UNKNOWN_MOVABLE':('unknown','unknown_movable'),
                      'ST_UNKNOWN_UNMOVABLE':('unknown','unknown_unmovable'), 'ST_CAR':('small_car','car'),
                     'ST_VAN':('small_car','van'), 'ST_TRUCK':('big_car','truck'), 'ST_BUS':('big_car','bus'),
                    'ST_CYCLIST':('rider','cyclist'), 'ST_MOTORCYCLIST':('rider','motorcyclist'),
                      'ST_TRICYCLIST':('small_car','tricyclist'),'ST_PEDESTRIAN':('person',None),'ST_TRAFFICCONE':('trafficcone',None),
                     'ST_LIGHT_TRUCK':('small_car','light_truck'), 'ST_TWO_WHEELED':('two_wheeled',None), 'ST_RIDER':('rider',None),
                    'ST_SPECIAL_VEHICLE':('special_vehicle',None)}
    type_id = {'unknown':0, 'person':1, 'rider':2, 'small_car':3, 'big_car':4, 'trifficcone': 5, 'two_wheeled':6, 'special_vehicle':7}
    sub_type_id = {"unknown": {'unknown':0, 'unknown_movable':1 ,'unknown_unmovable':2},
            'person':None, 'rider':{'motocyclist':0, 'cyclist':1,}, 'small_car':{'car':0, 'van':1, 'light_truck':2, 'tricyclist':3},
                   'big_car':{'truck':0, 'bus':1}, 'trafficcone':None, 'two_wheeled':None, 'special_vehicle': None}
    lane_attribute_map = {'THIRD_LEFT_LANE':-3, 'SECOND_LEFT_LANE':-2, 'LEFT_LANE':-1, 'HOST_LANE':0,
        'RIGHT_LANE':1,'SECOND_RIGHT_LANE':2,'THIRD_RIGHT_LANE':3,'OTHER_LANE':4, 'UNKNOWN_LANE':-9}

    if DEBUG: print(type(perception_result), len(perception_result),perception_result[0])

    '''
    # random data validation:
    # random sample 5 frames for input json data validation
    obstacle_keys_need = [k for k,v in obstacle_keys.items() if v == 1]
    frame_keys_need = [k for k,v in frame_keys.items() if v == 1]
    sample_frame_num = 10
    if len(perception_result) < sample_frame_num:
        sample_frame_num = len(perception_result)
    some_frames = random.sample(perception_result,sample_frame_num)
    for one_frame in some_frames:
        for key_ in frame_keys_need:
            assert key_ in one_frame['header'].keys(),f'Data source Error! No Frame Header Key: {key_}'
        # for key_ in one_frame['header'].keys():
        #     assert key_ in frame_keys,f'Weird Header Key: {key_}'
        if 'perception_obstacle' not in one_frame:
            one_frame['perception_obstacle'] = []
        if len(one_frame['perception_obstacle']) == 0:
            continue
        else:
            some_obs = random.sample(one_frame['perception_obstacle'],1)
            for one_ob in some_obs:
                # check keys must have
                for key_ in obstacle_keys_need:
                    assert key_ in one_ob.keys(),f'Data source Error! No Obstacle Key: {key_}'
                # check all keys
                # for key_ in one_obstacle.keys():
                #     assert key_ in obstacle_keys,f'Weird Obstacle Key: {key_}'
    '''
    
    for frame_idx,frame_result in enumerate(perception_result):
        if 'perception_obstacle' not in frame_result:
            frame_result['perception_obstacle'] = []
        json_obstacles = frame_result['perception_obstacle']
        if len(json_obstacles) == 0:
            continue
        json_header = frame_result['header']
        frame_id = json_header['sequence_num'] # Note: not 'frame_id'
        if 'timestamp_sec' in json_header:
            frame_stamp = float(json_header['timestamp_sec']) * 1000 # s to ms
        else:
            frame_stamp = float(json_header['camera_timestamp']) / 1000000 # ns to ms
        
        cipv_index = -1
        frame_potential_cipv_index_list = []
        if 'cipv_info' in frame_result:
            if frame_result['cipv_info']['cipv_id'] >= 0: # != -1
                cipv_index = frame_result['cipv_info']['cipv_id']
            frame_potential_cipv_index_list = frame_result['cipv_info']['potential_cipv_id']
        
        frame_obstacles = []        
        for ob_idx,json_ob in enumerate(json_obstacles):
            ob = Obstacle()
            ob_label = ['0b','0','0','0','0','0','0','0','0',
                        '0','0', # -23 ~ -24 may change
                        '0',
                        '0','0','0','0','0','0','0', # -15 ~ -21 may change
                        '1',
                        '1','0','0','0','0','0','0', # -7 ~ -13 may change
                        '1',
                        '0','0', # -4 ~ -5 may change
                        '1','1','0']
            ob.sensor_id = None  # -1, 传感器id
            ob.frame_id = int(frame_id)  # -2, 帧id
            ob.stamp = int(frame_stamp)  # -3, 帧时间戳
            if 'id' in json_ob:
                ob.id = json_ob['id']  # -4, 障碍物id
                ob_label[-4] = '1'
                ob.track_id = json_ob['id']  # -5, 障碍物跟踪id
                ob_label[-5] = '1'
                
            ob.is_tracked = True  # -6, 代表是否被跟踪

            if 'associate_camera_id' in json_ob:
                ob.camera_track_id = json_ob['associate_camera_id']  # 障碍物在相机感知输出结果中的跟踪id
                ob_label[-7] = '1'
            if 'associate_radar_id' in json_ob:
                ob.radar_track_id = json_ob['associate_radar_id']  # 障碍物在毫米波雷达感知输出结果中的跟踪id
                ob_label[-8] = '1'
            if "obj_source" in json_ob:
                if json_ob["obj_source"] == "OS_UNKNOWN": # 障碍物感知来源, UNKNOWN=0，CAMERA=1，RADAR=2，FUSION=3，LIDAR=4，
                    ob.source = 0
                elif json_ob["obj_source"] == "OS_CAMERA":
                    ob.source = 1
                elif json_ob["obj_source"] == "OS_RADAR":
                    ob.source = 2
                elif json_ob["obj_source"] == "OS_FUSION":
                    ob.source = 3
                elif json_ob["obj_source"] == "OS_LIDAR":
                    ob.source = 4
                ob_label[-9] = '1'

            if 'tracking_time' in json_ob:
                ob.age = int(json_ob['tracking_time'])  # -10, 障碍物的生命周期
                ob_label[-10] = '1'
            
            child_type = None
            if 'sub_type' in json_ob:
                apollo_type = json_ob['sub_type']
                parent_type,child_type = type_map[apollo_type][0],type_map[apollo_type][1]
                # print(parent_type,child_type)
                ob.type = type_id[parent_type]   # -11, 障碍物类型
                ob_label[-11] = '1'
                
            if child_type:
                ob.sub_type = sub_type_id[parent_type][child_type]
                ob_label[-12] = '1'
            
            # Pred motion_state, UNKNOWN = 0, MOVING = 1, STATIONARY = 2, STOPPED = 3
            # GT, move_status, 障碍物运动状态，1运动，0静止，-1停止
            if "motion_state" in json_ob:
                motion_state = json_ob["motion_state"]
                if motion_state == "MS_UNKNOWN":
                    ob.move_status = 2
                elif motion_state == "MS_MOVING":
                    ob.move_status = 1
                elif motion_state == "MS_STATIONARY":
                    ob.move_status = 0
                elif motion_state == "MS_STOPPED":
                    ob.move_status = -1

            
            # convert None to 0
            if 'position' in json_ob:
                x,y,z = json_ob['position']['x'],json_ob['position']['y'],json_ob['position']['z']
                x,y,z = x if x else 0, y if y else 0, z if z else 0
                ob.position = np.array([float(x),float(y),float(z)])  # -13, 障碍物的相对坐标（车体坐标系）
                ob_label[-13] = '1'
            ob.point_pos = 0  # -14, 障碍物测量点在车身位置（如车尾中心、车中心等）
                            
            if 'bbox2d' in json_ob:
                xmin,ymin,xmax,ymax = json_ob['bbox2d']['xmin'],json_ob['bbox2d']['ymin'],json_ob['bbox2d']['xmax'],json_ob['bbox2d']['ymax']
                xmin,ymin,xmax,ymax = xmin if xmin else 0, ymin if ymin else 0, xmax if xmax else 0, ymax if ymax else 0
                ob.image_pos = np.array([int(xmin),int(ymin),int(xmax),int(ymax)])  # -15, 图像感知框的左上角和右下角坐标（图像坐标系）
                ob_label[-15] = '1'
            
            ob.lane_attribute = -9
            ob.lane_offset_ratio = 0.0
            if 'situation' in json_ob:
                if 'lane_attrbute' in json_ob['situation']:
                    dut_lane_attribute = json_ob['situation']['lane_attrbute']
                    ob.lane_attribute = lane_attribute_map.get(dut_lane_attribute,0)
                if 'lane_offset_ratio' in json_ob['situation']:
                    ob.lane_offset_ratio = json_ob['situation']['lane_offset_ratio']
                
                
            ob.ground_point_2d = [-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.] # 前视图接地点坐标和置信度
            ob.ground_point_3d = [-10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,0.,0.,0.,0.] # BEV 3D 接地点横纵位置、接地点是否存在
            if "measurements" in json_ob:
                for m_ in json_ob["measurements"]:
                    if 'ground_point' in m_:
                        fl,fr,bl,br = m_['ground_point']
                        flx,fly,frx,fry,blx,bly,brx,bry,vfl,vfr,vbl,vbr = -1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1. # 前视图接地点坐标和置信度
                        flx3d,fly3d,frx3d,fry3d,blx3d,bly3d,brx3d,bry3d,vfl3d,vfr3d,vbl3d,vbr3d = -10.,-10.,-10.,-10.,-10.,-10.,-10.,-10.,0.,0.,0.,0. # BEV 3D 接地点横纵位置、接地点是否存在
                        
                        # 左前轮
                        try:    
                            if fl['position']: 
                                flx,fly,vfl = fl['position']['x'],fl['position']['y'],fl['visibility']
                                if fl['position_3d']:
                                    flx3d,fly3d,vfl3d = fl['position_3d']['x'],fl['position_3d']['y'],fl['visibility_3d']
                            # 右前轮
                            if fr['position']: 
                                frx,fry,vfr = fr['position']['x'],fr['position']['y'],fr['visibility']
                                if fr['position_3d']:
                                    frx3d,fry3d,vfr3d = fr['position_3d']['x'],fr['position_3d']['y'],fr['visibility_3d']
                            # 左后轮
                            if bl['position']: 
                                blx,bly,vbl = bl['position']['x'],bl['position']['y'],bl['visibility']
                                if fl['position_3d']:
                                    blx3d,bly3d,vbl3d = bl['position_3d']['x'],bl['position_3d']['y'],bl['visibility_3d']
                            # 右后轮
                            if br['position']: 
                                brx,bry,vbr = br['position']['x'],br['position']['y'],br['visibility']
                                if br['position_3d']:
                                    brx3d,bry3d,vbr3d = br['position_3d']['x'],br['position_3d']['y'],br['visibility_3d']
                        except:
                            # 旧版本代码不兼容，输出报错内容
                            import traceback
                            traceback.print_exc()
                             
                        ob.ground_point_2d = [flx,fly,frx,fry,blx,bly,brx,bry,vfl,vfr,vbl,vbr]
                        ob.ground_point_3d = [flx3d,fly3d,frx3d,fry3d,blx3d,bly3d,brx3d,bry3d,vfl3d,vfr3d,vbl3d,vbr3d]
                        
                        if DEBUG:
                            print('ob.ground_point_2d',ob.ground_point_2d)
                            print('ob.ground_point_3d', ob.ground_point_3d)
                    
                    # if 'situation' in m_:
                    #     dut_lane_attribute =  m_['situation']['lane_attrbute']
                    #     dut_lane_attribute_map = {'UNKNOWN_LANE':-9,'LEFT_LANE':-1,'HOST_LANE':0,'RIGHT_LANE':1,'OTHER_LANE':4}
                    #     ob.lane_attribute = dut_lane_attribute_map.get(dut_lane_attribute,0)

                                 
            l,w,h = 0,0,0
            if 'length' in json_ob: 
                l = json_ob['length']
                ob_label[-16] = '1'
            if 'width' in json_ob: w = json_ob['width']
            if 'height' in json_ob: h = json_ob['height']            
            # l,w,h = l if l else 0, w if w else 0, h if h else 0
            ob.size = np.array([float(l), float(w),float(h)])  # -16, 障碍物的真实尺寸
            
            theta = 0
            if 'theta' in json_ob: 
                theta = json_ob['theta']
                ob_label[-17] = '1'
            # theta = theta if theta else 0
            ob.rotation = np.array([float(0), float(0), float(theta)])  # -17, 障碍物的相对朝向（车体坐标系），json_ob['theta']
            
            vx,vy,vz = 0,0,0
            if 'velocity' in json_ob:
                vx,vy,vz = json_ob['velocity']['x'],json_ob['velocity']['y'],json_ob['velocity']['z']
                ob_label[-18] = '1'
            # vx,vy,vz = vx if vx else 0, vy if vy else 0, vz if vz else 0
            ob.linear_twist = np.array([float(vx), float(vy), float(vz)])  # -18, 障碍物的相对线速度（车体坐标系）

            if 'angular_speed' in json_ob: # -19, 障碍物的相对角速度（车体坐标系）
                angx,angy,angz = json_ob['angular_speed']['x'],json_ob['angular_speed']['y'],json_ob['angular_speed']['z']
                angx,angy,angz = angx if angx else 0, angy if angy else 0, angz if angz else 0 
                ob.angular_twist = np.array([float(angx), float(angy),float(angz)])
                ob_label[-19] = '1'
            
            accx,accy,accz = 0,0,0
            if 'acceleration' in json_ob:
                accx,accy,accz = json_ob['acceleration']['x'],json_ob['acceleration']['y'],json_ob['acceleration']['z']
                accx,accy,accz = accx if accx else 0, accy if accy else 0, accz if accz else 0
                ob_label[-20] = '1'
            ob.acceleration = np.array([float(accx), float(accy),float(accz)])   # -20, 障碍物的相对加速度（车体坐标系）
            
            if cipv_index >= 0:
                ob_label[-21] = '1'
            if ob_idx == cipv_index:
                ob.cipv = True
            else:
                ob.cipv = False  # 目标是否为cipv
            
            if frame_potential_cipv_index_list and (ob_idx in frame_potential_cipv_index_list):
                ob.potential_cipv = frame_potential_cipv_index_list.index(ob_idx)
                # print(frame_potential_cipv_index_list, ob_idx,ob.potential_cipv)
            else:
                ob.potential_cipv = -1

            if 'forward_motion_state' in json_ob.keys():
                status = json_ob['forward_motion_state']
                if status == 'FMS_CUT_IN':
                    ob.cut_in_flag = True  # -22, 目标是否要cutin
                else:
                    ob.cut_in_flag = None
                if status == 'FMS_PRE_CUT_IN':
                    ob.pre_cutin_flag = True  # -23, 目标是否预测为cutin
                    ob.pre_cutin_prob = json_ob['forward_motion_state_confidence'] # -24, 目标预测为cutin的概率
                else:
                    ob.pre_cutin_flag = None 
                    ob.pre_cutin_prob = None
            else:
                ob.cut_in_flag = None
            if 'position_covariance' in json_ob: 
                # for i in json_ob['position_covariance']:
                #     print(type(i), i)
                ob.pose_cov = json_ob['position_covariance']  # -25, 位姿对角线方差
                ob_label[-23] = '1'
            if 'velocity_covariance' in json_ob: 
                ob.velocity_cov = json_ob['velocity_covariance'] # -26,
                ob_label[-24] = '1'
            ob.twist_cov = None  # -26, 速度和角速度对角线方差
            ob.acceleration_cov = None # -27
            ob.certainty = None  # -28, 障碍物置信度
            ob.type_certainty = None  # -30, 障碍物类别置信度


            """
            多属性字段，获取该字段的值，并以2进制表示，由低位到高位，b0为cut out驾驶行为定义，b1和b2是动静态（01静止,10运动,11停止),
            b3和b4是potential_cipv判断（01 potential_cipv，10unknow，11非potential_cipv）。
            解析时，请从低位到高位获取，从低位到高位获取可避免解析程序的失效。
            """
            ob.label = 0b00000000000000000000000000000000 # -30
            ob.valid_info = int(''.join(ob_label),2)
            
            '''
            if not ob.angular_twist:
                ob.valid_info = 0b00000001000000010111111111101110  # 该字段的值以2进制表示，由低位到高位，第0位表示上面定义的第一个属性是否有效，以此类推
            else:
                ob.valid_info = 0b00000001000000011111111111101110
            if cipv_index >= 0:
                ob.valid_info = ob.valid_info + 0b00000000000000100000000000000000
            '''
            
            frame_obstacles.append(ob)
        if DEBUG: print(f'Frame {frame_idx+1}/{len(perception_result)}', len(frame_obstacles), 'obstacles')
        data_frame_result.append(frame_obstacles)
    if DEBUG: print('got apollo json ', len(data_frame_result), 'frames')

    global_coord = input_file['apollo_obstacle_global']
    if global_coord:
        if DEBUG: print('Converting global to local...')
        assert os.path.exists(odometry_json_path), '%s not exist!' % odometry_json_path
        with open(odometry_json_path, 'r') as f:
            all_frames_data = f.readlines()
            odometry_list = ujson.loads(all_frames_data[0])
        data_frame_result = global_to_local(data_frame_result,odometry_list)
    
    # to keep var name consistent with SIL detail design: https://cooper.didichuxing.com/docs/document/2199515636945
    dut_obstacles = data_frame_result
    return dut_obstacles

def global_to_local(data_frame_list, odometry_list):
    """
    convert object property values from global absolute coordinate system to local relative coordinate system
    :return:
    """
    local_data_frame_list = []
    odomTimeSeq = [i['header']['timestamp_sec']*1000 for i in odometry_list]
    odomTimePose = {i['header']['timestamp_sec']*1000:i['pose'] for i in odometry_list}

    for global_frame in data_frame_list:
        local_frame = []
        globalTimeStamp = global_frame[0].stamp
        # find two_nerast odom time stamp
        stamps2 = find_2_nearest_timeStamp(inputStampSeq=odomTimeSeq,targetStamp=globalTimeStamp)
        startOdomTime, endOdomTime = stamps2

        odomYaw1, odomYaw2 = Quaternion([v for k,v in odomTimePose[startOdomTime]['orientation'].items()]).yaw_pitch_roll[0], \
                             Quaternion([v for k,v in odomTimePose[endOdomTime]['orientation'].items()]).yaw_pitch_roll[0]
        odomYaw = interp_(stamps2,[odomYaw1, odomYaw2],globalTimeStamp)

        odomAngular_twist1,odomAngular_twist2 = [v for k,v in odomTimePose[startOdomTime]['angular_velocity'].items()],\
                                                [v for k,v in odomTimePose[endOdomTime]['angular_velocity'].items()]
        odomAngular_twist = interp_(stamps2,[odomAngular_twist1, odomAngular_twist2],globalTimeStamp)

        odomT1 = [v for k,v in odomTimePose[startOdomTime]['position'].items()] # odomTx1, odomTy1, odomTz1
        odomT2 = [v for k,v in odomTimePose[endOdomTime]['position'].items()] # odomTx2, odomTy2, odomTz2
        odomTx,odomTy,odomTz = interp_(stamps2,[odomT1,odomT2],globalTimeStamp) # odomT,shift of position

        odomV1 = [v for k,v in odomTimePose[startOdomTime]['linear_velocity'].items()] # odomVx1, odomVy1, odomVz1
        odomV2 = [v for k,v in odomTimePose[endOdomTime]['linear_velocity'].items()] # odomVx2, odomVy2, odomVz2
        odomVx,odomVy,odomVz = interp_(stamps2,[odomV1,odomV2],globalTimeStamp) # odomV,shift of velocity

        MatR = np.array([[np.cos(-odomYaw),-np.sin(-odomYaw),0],[np.sin(-odomYaw),np.cos(-odomYaw),0], [0,0,1]]).reshape((3,3))
        positionT = np.array([odomTx,odomTy,odomTz]).reshape((3,1))
        velocityT = np.array([odomVx,odomVy,odomVz]).reshape((3,1))
        MatPosition4D = np.vstack((np.hstack((MatR,positionT)), np.array([0,0,0,1])))
        MatVelocity4D = np.vstack((np.hstack((MatR,velocityT)), np.array([0,0,0,1])))

        for global_obstacle in global_frame:
            local_obstacle = copy.deepcopy(global_obstacle)
            local_obstacle.position = np.linalg.pinv(MatPosition4D) * np.concatenate((global_obstacle.position,[1]))
            local_obstacle.position = local_obstacle.position[:3]
            local_obstacle.rotation = np.array([global_obstacle.position[0], global_obstacle.position[1],
                                                global_obstacle.position[2]-odomYaw])
            local_obstacle.linear_twist = np.linalg.pinv(MatVelocity4D) * np.concatenate((global_obstacle.linear_twist,[1]))
            local_obstacle.linear_twist = local_obstacle.linear_twist[:3]
            if global_obstacle.angular_twist != None:
                local_obstacle.angular_twist = np.array([global_obstacle.angular_twist[0], global_obstacle.angular_twist[1],
                                                global_obstacle.angular_twist[2]-odomAngular_twist])
            if 'acceleration' in odomTimePose[startOdomTime] and 'acceleration' in odomTimePose[endOdomTime]:
                odomA1 = odomTimePose[startOdomTime]['acceleration']  # odomVx1, odomVy1, odomVz1
                odomA2 = odomTimePose[endOdomTime]['acceleration']  # odomVx2, odomVy2, odomVz2
                odomAx, odomAy, odomAz = interp_(stamps2, [odomA1, odomA2], globalTimeStamp)  # odomV,shift of velocity

                MatAcceleration4D = np.array([[np.cos(-odomYaw), -np.sin(-odomYaw), 0, odomAx],
                                              [sin(-odomYaw), cos(-odomYaw), 0, odomAy],
                                              [0, 0, 1, odomAz], [0, 0, 0, 1]]).reshape((4, 4))
                local_obstacle.acceleration = np.linalg.pinv(MatAcceleration4D) * global_obstacle.acceleration

            if global_obstacle.pose_cov:
                odomPcov1 = odomTimePose[startOdomTime]['position_covariance']  
                odomPcov2 = odomTimePose[endOdomTime]['position_covariance']  
                odomPcov = interp_(stamps2, [odomPcov1, odomPcov2], globalTimeStamp) 

                PC = odomPcov.reshape((3,3))
                positionX,positionY,positionZ,pitch,roll,yaw = global_obstacle.pose_cov
                result = np.linalg.pinv(MatR) * PC *(np.linalg.pinv(MatR)).T
                local_obstacle.pose_cov = np.array([result[0,0], result[1,1], result[2,2], pitch,roll,yaw])

            if global_obstacle.twist_cov:
                odomTcov1 = odomTimePose[startOdomTime]['twist_covariance']  
                odomTcov2 = odomTimePose[endOdomTime]['twist_covariance']  
                odomTcov = interp_(stamps2, [odomTcov1, odomTcov2], globalTimeStamp) 

                TC = odomTcov.reshape((3, 3))
                linearVx,linearVy,linearVz,angularVx,angularVy,angularVz = global_obstacle.twist_cov
                result = np.linalg.pinv(MatR) * TC *(np.linalg.pinv(MatR)).T
                local_obstacle.twist_cov = np.array([result[0,0], result[1,1], result[2,2], angularVx,angularVy,angularVz])

            if global_obstacle.acceleration_cov:
                odomAcov1 = odomTimePose[startOdomTime]['acceleration_covariance']  
                odomAcov2 = odomTimePose[endOdomTime]['acceleration_covariance']  
                odomAcov = interp_(stamps2, [odomAcov1, odomAcov2], globalTimeStamp) 

                AC = odomAcov.reshape((3, 3))
                _,_,_,a,b,c = global_obstacle.acceleration_cov
                result = np.linalg.pinv(MatR) * AC *(np.linalg.pinv(MatR)).T
                local_obstacle.acceleration_cov = np.array([result[0,0], result[1,1], result[2,2], a, b, c])

            local_frame.append(local_obstacle)
        local_data_frame_list.append(local_frame)
    return local_data_frame_list






if __name__ == '__main__':
    obstacle_csv_path = "/nfs/adas_s3_dataset/adas_dataset_public/suteng_zhenzhi_bag_sample/211019_211021_shenzhen_collected/" \
                        "2021-10-20-16-20-00/result/GT.csv"
    # obstacle_csv_path = '/nfs/adas_s3_dataset/adas_dataset_public/suteng_zhenzhi_bag_sample/tmp/GT.csv'
    obstacle_csv_path = '/nfs/dataset-meta-data/bag_split/suteng/2022-08-29-16-23-48_1/gt/result/GT.csv'

    # transform suteng csv to standard data
    # out = obstacle_suteng_csv_normalization(obstacle_csv_path)

    # generate dut data
    # dut = dut_simulate_noise_obstacle_normalization(obstacle_csv_path)
    # print(len(dut))
    # for frame in dut:
    #     for ob in frame:
    #         print('frame id', ob.frame_id, 'time stamp', ob.stamp)

    # apollo_json_path = '/nfs/adas_s3_dataset/adas_s3_dataset/public/wangjunchu/tmp/20211130153041683777.json'
    odometry_json_path = '/nfs/dataset-adas_s3_dataset/adas_dataset_public/zijian_data_collected/tmp/odometry.json'
    # transform apollo json to standard data

    apollo_json_path = '/nfs/cache-902-1/wangjunchu/tmp/obstacle.json'
    # apollo_json_path = '/nfs/cache-902-1/wangjunchu/tmp/obstacle_polar_2.json'
    apollo_json_path = '/nfs/volume-902-16/wangjunchu/tmp/sil/dt_obstacles.json'
    apollo_json_path = '/nfs/volume-902-16/wangjunchu/data/sil/dt_obstacles_2022-06-24-08-50-32_4_gpt_bev.json'
    # apollo_json_path = '/nfs/ofs-902-1/object-detection/wangjunchu/data/sil/dt_obstacles.json'
    apollo_json_path = '/nfs/ofs-902-1/object-detection/wangjunchu/tmp/tmp/dt_obstacles.json'
    
    out = obstacle_apollo_json_normalization(apollo_json_path,odometry_json_path)
    print(out)
    print('done!')




