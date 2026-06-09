# mSTCV (temporal-consistency) evaluation config for ST-Occ.
#
# This config evaluates the same model/checkpoint as stocc-r50-256x704-36e.py
# but additionally measures the spatiotemporal classification variability of
# the predictions: per frame, the fraction of non-free voxels whose predicted
# class changed w.r.t. the aggregated past predictions at the same global
# location (STCV). The frame-averaged value is reported as `mSTCV` in the
# evaluation results (a fraction; x100 gives the percentage reported in the
# ST-Occ paper, arXiv:2508.04705). `flickering` (raw changed-voxel count) and
# `flickering_percentage` (normalized by all annotated voxels) are also logged.
#
# Usage:
#   bash tools/dist_test.sh occupancy_configs/st_occ/stocc-r50-256x704-36e_flicker_eval.py <checkpoint> <num_gpus>

_base_ = ['./stocc-r50-256x704-36e.py']

model = dict(
    # count the flickering of the prediction w.r.t. the global gt memory
    flickering_cnt=True,
    # use the precise 2x upscaled global coordinates when aggregating
    precise_upscale=True,
)

point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/gts'
num_cls = 19  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(type='LoadOccupancy', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path),
            dict(type='LoadOccVelocityMap', enlarge=1.4, z_enlarge_ratio=0.8),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs', 'gt_occupancy', 'visible_mask',
                                         'gt_speed_map'])
            ]
        )
]

data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

evaluation = dict(pipeline=test_pipeline)
