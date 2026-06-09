# Copyright (c) OpenMMLab. All rights reserved.
import pickle

import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

from tools.data_converter import nuscenes_converter as nuscenes_converter

from pyquaternion import Quaternion
import numpy as np

def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat.tolist()

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

VERSION= 'v1.0-trainval'
NUSCENES = 'nuscenes'
def get_gt(info):
    """Generate gt labels from info.

    Args:
        info(dict): Infos needed to generate gt labels.

    Returns:
        Tensor: GT bboxes.
        Tensor: GT labels.
    """
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT'][
        'ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes = list()
    gt_labels = list()
    for ann_info in info['ann_infos']:
        # Use ego coordinate.
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)



def add_ann_adj_info(extra_tag, with_lidar_seg=False):
    nuscenes_version = VERSION
    dataroot = f'./data/{NUSCENES}/'
    nuscenes = NuScenes(nuscenes_version, dataroot)

    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set), 'rb'))
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            ann_infos = list()
            for ann in sample['anns']:
                ann_info = nuscenes.get('sample_annotation', ann)
                velocity = nuscenes.box_velocity(ann_info['token'])
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros(3)
                ann_info['velocity'] = velocity
                ann_infos.append(ann_info)
            dataset['infos'][id]['ann_infos'] = ann_infos
            dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
            dataset['infos'][id]['scene_token'] = sample['scene_token']
            scene = nuscenes.get('scene',  sample['scene_token'])
            dataset['infos'][id]['scene_name'] = scene['name']
            dataset['infos'][id]['prev'] = sample['prev']
            # description = scene['description']
            if with_lidar_seg:
                lidar_sd_token = sample['data']['LIDAR_TOP']
                dataset['infos'][id]['lidarseg_filename'] =  nuscenes.get('lidarseg', lidar_sd_token)['filename']


            scene = nuscenes.get('scene', sample['scene_token'])
            dataset['infos'][id]['occ_path'] = \
                './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
        with open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)
            
def add_global_info(extra_tag, with_lidar_seg=False):
    nuscenes_version = VERSION
    dataroot = f'./data/{NUSCENES}/'
    nuscenes = NuScenes(nuscenes_version, dataroot)

    for set in ['train', 'val']:
        dataset = pickle.load(
            open('./data/%s/%s_infos_%s.pkl' % (NUSCENES, extra_tag, set), 'rb'))
        last_scene = '-1'
        last_token_idx = '-1'
        for id in range(len(dataset['infos'])):
            if id % 10 == 0:
                print('%d/%d' % (id, len(dataset['infos'])))
            info = dataset['infos'][id]
            # get sweep adjacent frame info
            sample = nuscenes.get('sample', info['token'])
            if last_scene != sample['scene_token']:
                scene = nuscenes.get('scene', info['scene_token'])
                key_frame_recs = [nuscenes.get('sample', scene['first_sample_token'])]
                key_frame_tokens = [scene['first_sample_token']]
                tran = nuscenes.get('ego_pose', key_frame_recs[-1]['data']['LIDAR_TOP'])['translation']
                rot = nuscenes.get('ego_pose', key_frame_recs[-1]['data']['LIDAR_TOP'])['rotation']
                x, y, _ = tran
                tm = rt2mat(tran, quaternion=rot)
                xs, ys, rots, trans, tms = [x], [y], [rot], [tran], [tm]
                while key_frame_recs[-1]['next'] != '':
                    key_frame_recs.append(nuscenes.get('sample', key_frame_recs[-1]['next']))
                    key_frame_tokens.append(key_frame_recs[-1]['token'])
                    tran = nuscenes.get('ego_pose', key_frame_recs[-1]['data']['LIDAR_TOP'])['translation']
                    rot = nuscenes.get('ego_pose', key_frame_recs[-1]['data']['LIDAR_TOP'])['rotation']
                    x, y, _ = tran
                    tm = rt2mat(tran, quaternion=rot)
                    xs.append(x)
                    ys.append(y)
                    rots.append(rot)
                    trans.append(tran)
                    tms.append(tm)
                global_size_x = (- min(xs) + max(xs) + 150) / 2
                global_size_y = (- min(ys) + max(ys) + 150) / 2
                global_x_center = (min(xs) + max(xs)) / 2
                global_y_center = (min(ys) + max(ys)) / 2
                global_range_xy = np.array([global_x_center - global_size_x, global_y_center - global_size_y, -1, 
                                            global_x_center + global_size_x, global_y_center + global_size_y, 5.4])
                aabb_min, aabb_max = np.split(global_range_xy, 2)
                global_size = aabb_max - aabb_min
                global_center = [global_x_center, global_y_center, 0]
                # reset last_scene
                last_scene = sample['scene_token']
                last_token_idx = -1
            curr_idx = key_frame_tokens.index(info['token'])
            assert curr_idx == last_token_idx + 1
            last_token_idx = curr_idx
            
            # add global infos
            dataset['infos'][id]['global_center'] = global_center
            dataset['infos'][id]['global_range_xy'] = global_range_xy
            dataset['infos'][id]['global_size'] = global_size
            dataset['infos'][id]['global_kf_token'] = key_frame_tokens
            dataset['infos'][id]['global_idx'] = curr_idx
            dataset['infos'][id]['global_tms'] = tms
            dataset['infos'][id]['global_rots'] = rots
            dataset['infos'][id]['global_trans'] = trans
            dataset['infos'][id]['curr_tm'] = tms[curr_idx]
            
        with open('./data/%s/%s_infos_global_%s.pkl' % (NUSCENES, extra_tag, set),
                  'wb') as fid:
            pickle.dump(dataset, fid)


if __name__ == '__main__':
    dataset = 'nuscenes'
    version = 'v1.0'
    train_version = VERSION
    root_path = f'./data/{NUSCENES}'
    extra_tag = 'bevdetv4-nuscenes'
    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=train_version,
        max_sweeps=0)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag)

    print('add_global_infos')
    add_global_info(extra_tag)