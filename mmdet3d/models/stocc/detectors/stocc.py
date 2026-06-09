# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS

from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
from mmdet3d.models.stocc.modules.occ_loss_utils import nusc_class_frequencies


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


# xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
# ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
# zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
# grid = torch.stack(
#     (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

@DETECTORS.register_module()
class STOcc(CenterPoint):

    def __init__(self,
                 # BEVDet components
                 forward_projection=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 # BEVFormer components
                 backward_projection=None,
                 # depth_net
                 depth_net=None,
                 # occupancy head
                 occupancy_head=None,
                 # other settings.
                 use_depth_supervision=False,
                 readd=False,
                 fix_void=False,
                 occupancy_save_path=None,
                 do_history=True,
                 history_cat_num=16,
                 single_bev_num_channels=80,
                 temporal_encoder=None,
                 temporal_pe=None,
                 uncertainty_config=None,
                 implicit_uncertainty=False,
                 relative_scale=False,
                 embed_feature_uncertainty=False,
                 w_au=False,
                 au_ef=False,
                 flickering_cnt=False,
                 precise_upscale=False,
                 disable_flow_init=False,
                 **kwargs):
        super(STOcc, self).__init__(**kwargs)
        self.fix_void = fix_void

        # BEVDet init
        self.forward_projection = builder.build_neck(forward_projection) if forward_projection else None
        self.img_bev_encoder_backbone = builder.build_backbone(
            img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None

        # BEVFormer init
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None

        # Depth Net
        self.depth_net = builder.build_head(depth_net) if depth_net else None

        # Occupancy Head
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        self.readd = readd  # fuse voxel features and bev features

        self.use_depth_supervision = use_depth_supervision

        self.occupancy_save_path = occupancy_save_path  # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5  # seconds between each frame
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0

        self.gt_flow = None

        if temporal_encoder is not None:
            self.temporal_encoder = build_transformer_layer_sequence(temporal_encoder)
            self.temporal_pe = build_positional_encoding(temporal_pe)
            self.temporal_encoder.init_weights()

            self.global_feats = None
            self.global_valid_masks = None
            self.scene_names = None

            self.global_vis_cnt = None
            self.global_pred_distribs = None

            self.global_pos = None
            self.global_flip_reverse_grid_norm = None
            self.global_frame_cnt = None

            self.global_gt = None
            self.global_mask_gt = None
            self.global_coord = None
            self.global_coord_norm = None

            self.grid_config = self.forward_projection.grid_config

            self.global_biases = None
            self.ego_shape_maxes = None
            self.ego_shape_min = torch.tensor([0, 0, 0]).cuda()
            self.local_shape = torch.tensor(
                ((self.grid_config['x'][1] - self.grid_config['x'][0]) / self.grid_config['x'][2],
                 (self.grid_config['y'][1] - self.grid_config['y'][0]) / self.grid_config['x'][2],
                 (self.grid_config['z'][1] - self.grid_config['z'][0]) / self.grid_config['z'][2])).cuda()

            assert self.grid_config['x'][2] == self.grid_config['y'][2] == self.grid_config['z'][2]
            self.voxel_size = self.grid_config['x'][2]

        self.uncertainty_config = uncertainty_config

        self.initial_distribution = torch.from_numpy(nusc_class_frequencies / np.sum(nusc_class_frequencies)).cuda()

        if self.temporal_encoder is not None and implicit_uncertainty:
            self.implicit_uncertainty = True
        else:
            self.implicit_uncertainty = False

        self.relative_scale = relative_scale

        self.gt_flow = None

        self.is_test = False

        self.embed_feature_uncertainty = embed_feature_uncertainty

        self.w_au = w_au

        self.au_ef = au_ef

        self.flickering_cnt = flickering_cnt
        if flickering_cnt:
            self.flickering_cnt_gt = {}
            self.flickering_cnt_pred = {}
            self.total_cnt_gt = {}
            self.total_non_free_gt = {}

            self.flickering_distri_gt = {}
            self.flickering_distri_pred = {}

        self.precise_upscale = precise_upscale

        self.disable_flow_init = disable_flow_init

    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        return x

    @force_fp32()
    def bev_encoder(self, x):
        if self.with_specific_component('img_bev_encoder_backbone'):
            x = self.img_bev_encoder_backbone(x)

        if self.with_specific_component('img_bev_encoder_neck'):
            x = self.img_bev_encoder_neck(x)

        if type(x) not in [list, tuple]:
            x = [x]

        return x

    @force_fp32()
    def register_scene_buffer(self, idx, img_metas, device=None):
        voxel_resolution = torch.ceil(img_metas[idx]['global_size'] / self.voxel_size).long()
        self.scene_names[idx] = img_metas[idx]['scene_name']
        self.global_feats[idx] = torch.zeros(1, self.single_bev_num_channels, voxel_resolution[0], voxel_resolution[1],
                                             voxel_resolution[2], device=device, dtype=torch.float32)
        self.global_valid_masks[idx] = torch.zeros([1, voxel_resolution[0], voxel_resolution[1], voxel_resolution[2]],
                                                   device=device, dtype=torch.long)
        xsize, ysize, zsize = self.global_feats[idx].shape[2:]
        # self.global_biases[idx] = torch.tensor([xsize/2, ysize/2, 2.5], device=device, dtype=torch.float32)
        self.global_biases[idx] = torch.tensor([xsize / 2, ysize / 2, 1.25], device=device,
                                               dtype=torch.float32)  # might be a bug
        self.ego_shape_maxes[idx] = torch.tensor([xsize - 1, ysize - 1, zsize - 1], device=device)

        self.global_vis_cnt[idx] = torch.zeros([1, voxel_resolution[0], voxel_resolution[1], voxel_resolution[2]],
                                               device=device, dtype=torch.long)
        self.global_pred_distribs[idx] = torch.zeros(
            [1, voxel_resolution[0], voxel_resolution[1], voxel_resolution[2], self.occupancy_head.out_channel - 1],
            device=device, dtype=torch.float32)
        self.global_frame_cnt = 0

        # used to aggregate gt for training
        self.global_gt[idx] = torch.zeros(
            [1, voxel_resolution[0] * 2, voxel_resolution[1] * 2, voxel_resolution[2] * 2], device=device,
            dtype=torch.int)
        self.global_mask_gt[idx] = torch.zeros(
            [1, voxel_resolution[0] * 2, voxel_resolution[1] * 2, voxel_resolution[2] * 2], device=device,
            dtype=torch.bool)
        
        self.global_pred[idx] = torch.zeros([1, voxel_resolution[0] * 2, voxel_resolution[1] * 2, voxel_resolution[2] * 2], dtype=torch.int).to(device=device, non_blocking=True)
        self.global_mask_pred[idx] = torch.zeros([1, voxel_resolution[0] * 2, voxel_resolution[1] * 2, voxel_resolution[2] * 2], dtype=torch.bool).to(device=device, non_blocking=True)
        
        self.global_au[idx] = torch.ones([1, voxel_resolution[0], voxel_resolution[1], voxel_resolution[2]], dtype=torch.float32).to(device=device, non_blocking=True)

    @force_fp32()
    def update_global(self, curr_bev, img_metas, bda, gt_flow):
        bs = curr_bev.shape[0]
        if self.scene_names is None:
            self.scene_names = [None] * bs
        if self.global_feats is None:
            self.global_feats = [None] * bs
            self.global_valid_masks = [None] * bs
            self.global_biases = [None] * bs
            self.ego_shape_maxes = [None] * bs
            self.global_vis_cnt = [None] * bs
            self.global_pred_distribs = [None] * bs

            # just for global pos temporary memory
            self.global_pos = [None] * bs
            self.global_flip_reverse_grid_norm = [None] * bs

            self.global_gt = [None] * bs
            self.global_mask_gt = [None] * bs
            self.global_coord = None
            self.global_coord_norm = None

            self.global_au = [None] * bs

            self.global_pred = [None] * bs
            self.global_mask_pred = [None] * bs

        for i, single_img_metas in enumerate(img_metas):
            if single_img_metas['scene_name'] != self.scene_names[i]:
                self.register_scene_buffer(i, img_metas, device=curr_bev.device)
                self.gt_flow = None

        # Generate grid
        curr_bev = curr_bev.permute(0, 1, 4, 2, 3)
        n, c_, z, h, w = curr_bev.shape
        # grid_xyz = gridcloud3d(1, z, h, w, device=curr_bev.device)[0]

        # Generate grid x-w y-h z-z
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)

        _xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w,
                                                                                                             z) / (
                          w - 1)
        _ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w,
                                                                                                             z) / (
                          h - 1)
        _zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w,
                                                                                                             z) / (
                          z - 1)
        _grid = torch.stack(
            (_xs, _ys, _zs, torch.ones_like(_xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
        
        xs2x = torch.linspace(0, 2*w - 1, 2*w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 2*w, 1).expand(2*h, 2*w, 2*z)
        ys2x = torch.linspace(0, 2*h - 1, 2*h, dtype=curr_bev.dtype, device=curr_bev.device).view(2*h, 1, 1).expand(2*h, 2*w, 2*z)
        zs2x = torch.linspace(0, 2*z - 1, 2*z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, 2*z).expand(2*h, 2*w, 2*z)
        grid2x = torch.stack(
            (xs2x, ys2x, zs2x, torch.ones_like(xs2x)), -1).view(1, 2*h, 2*w, 2*z, 4).expand(n, 2*h, 2*w, 2*z, 4).view(n, 2*h, 2*w, 2*z, 4, 1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4, 4), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.forward_projection.dx[0]
        feat2bev[1, 1] = self.forward_projection.dx[1]
        feat2bev[2, 2] = self.forward_projection.dx[2]
        feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1, 4, 4)
        
        feat2bev_2x = torch.clone(feat2bev)
        feat2bev_2x[0].diagonal()[:3] = feat2bev_2x[0].diagonal()[:3] / 2
        
        forward_augs = generate_forward_transformation_matrix(bda)
        global_tm = torch.stack([img_metas[i]['curr_tm'] for i in range(bs)], dim=0).to(feat2bev.device)
        global_centers = torch.stack([img_metas[i]['global_center'] for i in range(bs)], dim=0).to(feat2bev.device)
        global_biases = torch.stack([self.global_biases[i] for i in range(bs)], dim=0)
        ego_shape_maxes = torch.stack([self.ego_shape_maxes[i] for i in range(bs)], dim=0)
        global_bevxy_aug_flow = global_tm @ torch.inverse(forward_augs) @ feat2bev
        global_bevxy_flow = global_tm @ feat2bev
        global_bevxy_aug = global_bevxy_aug_flow.view(n, 1, 1, 1, 4, 4) @ grid
        global_bevxy = global_bevxy_flow.view(n, 1, 1, 1, 4, 4) @ grid
        global_coord_aug = (global_bevxy_aug[..., :3, :] - global_centers.view(n, 1, 1, 1, 3,
                                                                               1)) / self.voxel_size + global_biases.view(
            n, 1, 1, 1, 3, 1)
        global_coord = (global_bevxy[..., :3, :] - global_centers.view(n, 1, 1, 1, 3,
                                                                       1)) / self.voxel_size + global_biases.view(n, 1,
                                                                                                                  1, 1,
                                                                                                                  3, 1)
                                                                       
        global_bevxy_aug_flow_2x = global_tm @ torch.inverse(forward_augs) @ feat2bev_2x
        global_bevxy_aug_2x = global_bevxy_aug_flow_2x.view(n, 1, 1, 1, 4, 4) @ grid2x
        global_coord_aug_2x = (global_bevxy_aug_2x[..., :3, :] - global_centers.view(n, 1, 1, 1, 3, 1)) / (self.voxel_size / 2) + global_biases.view(n, 1, 1, 1, 3, 1) * 2

        global_distance_aug = [None] * bs
        if self.uncertainty_config.global_distance:
            distance_aug_flow = torch.inverse(forward_augs) @ feat2bev
            global_bevxy_distance_aug = distance_aug_flow.view(n, 1, 1, 1, 4, 4) @ grid
            global_bevxy_distance_aug_xyz = global_bevxy_distance_aug.squeeze(-1)[..., :3]
            global_distance_aug = torch.norm(global_bevxy_distance_aug_xyz, dim=-1, keepdim=False)

        # global_coord_norm_aug = global_coord_aug / (ego_shape_maxes + 1).view(n, 1, 1, 1, 3, 1) * 2.0 - 1.0
        global_coord_norm_aug = global_coord_aug / (ego_shape_maxes).view(n, 1, 1, 1, 3,
                                                                          1) * 2.0 - 1.0  # might be a bug

        reverse_flow = torch.inverse(feat2bev) @ forward_augs @ feat2bev
        reverse_grid = reverse_flow.view(n, 1, 1, 1, 4, 4) @ grid

        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        reverse_grid = reverse_grid[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        return_feats = []
        for idx in range(bs):
            self.global_feats[idx].detach_()
            return_feat = self.update_global_single(idx, global_coord[idx, ..., 0], global_coord_norm_aug[idx, ..., 0],
                                                    img_metas[idx], curr_bev[idx].unsqueeze(0), self.global_feats[idx],
                                                    _grid[idx, ..., :3, 0], reverse_grid[idx], forward_augs[idx],
                                                    global_distance_aug[idx],
                                                    self.gt_flow[idx] if self.gt_flow is not None else None)
            return_feats.append(return_feat)
        return_feats = torch.cat(return_feats, dim=0)

        self.global_frame_cnt += 1

        self.global_coord = global_coord
        self.global_coord_aug = global_coord_aug
        self.global_coord_aug_2x = global_coord_aug_2x
        self.gt_flow = gt_flow

        return return_feats

    @force_fp32()
    def update_global_single(self, idx, global_coord, global_coord_norm, img_meta, curr_bev, global_feat, grid,
                             reverse_grid, forward_aug, global_distance_aug, gt_flow):
        global_coord = torch.round(global_coord).long()
        global_coord = torch.clamp(global_coord, min=self.ego_shape_min, max=self.ego_shape_maxes[idx])

        local_valid_mask = torch.zeros_like(self.global_valid_masks[idx], dtype=torch.bool)
        local_valid_mask[:, global_coord[..., 0], global_coord[..., 1], global_coord[..., 2]] = True
        local_valid_mask_smooth_kernel = torch.ones((3, 3, 3), dtype=torch.float32, device=local_valid_mask.device)
        local_valid_mask_smooth_kernel[1, 1, 1] = 0
        neighbor_count = F.conv3d(local_valid_mask.float().unsqueeze(0),
                                  local_valid_mask_smooth_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)
        smoothed_local_valid_mask = (neighbor_count > 13).bool()
        _, global_pos_x, global_pos_y, global_pos_z = smoothed_local_valid_mask.nonzero(as_tuple=True)
        global_pos = torch.stack([global_pos_x, global_pos_y, global_pos_z], dim=1)
        # transfer global_pos back to local grid
        reverse_transformed_ego_grid = (global_pos - self.global_biases[idx]) * self.voxel_size + img_meta[
            'global_center'].to(global_coord.device)
        tm = img_meta['curr_tm'].to(global_coord.device)
        tm_inv = torch.inverse(tm.float())
        reverse_transformed_ego_grid_homogeneous = torch.cat([reverse_transformed_ego_grid,
                                                              torch.ones(reverse_transformed_ego_grid.shape[0], 1).to(
                                                                  global_coord.device)], dim=1)
        reverse_ego_grid = torch.matmul(forward_aug @ tm_inv,
                                        reverse_transformed_ego_grid_homogeneous.transpose(0, 1)).transpose(0, 1)[...,
                           :3]
        reverse_grid_xyz = reverse_ego_grid / self.voxel_size + torch.tensor([50, 50, 1.25]).to(global_coord.device)
        reverse_grid_xyz = torch.clamp(reverse_grid_xyz, min=self.ego_shape_min,
                                       max=self.local_shape)  # self.local_shape = 100 100 8
        reverse_grid_norm = reverse_grid_xyz / (self.local_shape) * 2.0 - 1.0

        # TSA update
        dtype = curr_bev.dtype
        n, c_, z, h, w = curr_bev.shape

        voxel_query = curr_bev.permute(0, 2, 3, 4, 1).flatten(1, 3).permute(1, 0, 2).to(dtype)
        voxel_mask = torch.zeros((1, z, h, w), device=voxel_query.device).to(dtype)
        voxel_pos = self.temporal_pe(voxel_mask).to(dtype)
        voxel_pos = voxel_pos.flatten(2).permute(2, 0, 1)

        ref_3d = grid.unsqueeze(0).permute(0, 3, 1, 2, 4).flatten(1, 3).unsqueeze(
            2)  ##### try to align with the FB-OCC coordinate
        shift_ref_3d = grid.unsqueeze(0).permute(0, 3, 1, 2, 4).flatten(1, 3).unsqueeze(
            2)  ##### try to align with the FB-OCC coordinate

        vis_mask = None
        mask = None
        au = None
        prev_weights = None  # the prev_weight should be the weight of prev_voxel ranging from 0 to 1 with shape (z*h*w, 1, 1), 0.5 means average history and current

        flip_global_coord_norm = global_coord_norm.permute(2, 0, 1, 3).reshape(-1,
                                                                               3)  ##### try to align with the FB-OCC coordinate
        global_coord_norm_shape = flip_global_coord_norm.shape[:-1]
        flip_global_coord_norm = flip_global_coord_norm.reshape(1, 1, 1, -1, 3)

        if not self.do_history or img_meta['start_of_sequence'] or torch.sum(
                self.global_valid_masks[idx][:, global_coord[..., 0], global_coord[..., 1],
                global_coord[..., 2]]) == 0:  # self attention for the first frame
            prev_voxel = curr_bev.permute(0, 2, 3, 4, 1).flatten(1, 3).permute(1, 0, 2).to(dtype).clone().detach()
        else:
            global_feat_reverse = global_feat.permute(0, 1, 4, 3, 2)

            prev_voxel = F.grid_sample(global_feat_reverse, flip_global_coord_norm,
                                       mode='bilinear', align_corners=True,
                                       padding_mode='border').reshape(global_feat_reverse.shape[1], -1).T.reshape(
                *global_coord_norm_shape,
                global_feat_reverse.shape[1]).unsqueeze(1).to(dtype)

        if self.uncertainty_config is not None and self.do_history:
            prev_weights = torch.ones((voxel_query.shape[0], 1, 1), device=curr_bev.device) * 0.5
            if self.uncertainty_config.global_vis_cnt:
                global_vis_cnt_reverse = self.global_vis_cnt[idx].permute(0, 3, 2, 1).unsqueeze(1)
                global_vis_time = F.grid_sample(global_vis_cnt_reverse.float(), flip_global_coord_norm,
                                                mode='bilinear', align_corners=True,
                                                padding_mode='border').reshape(global_vis_cnt_reverse.shape[1],
                                                                               -1).T.reshape(*global_coord_norm_shape,
                                                                                             global_vis_cnt_reverse.shape[
                                                                                                 1]).unsqueeze(1).to(
                    dtype)
                alpha = self.uncertainty_config.global_vis_cnt_alpha
                prev_weights = (global_vis_time) / (global_vis_time + alpha)

                if self.relative_scale:
                    global_vis_time = global_vis_time / (self.global_frame_cnt + alpha)

            if self.uncertainty_config.global_distance:
                global_distance_aug = global_distance_aug.permute(2, 0, 1).reshape(-1, 1, 1)
                distance_alpha = self.uncertainty_config.global_distance_alpha
                distnace_max = self.uncertainty_config.global_distance_max
                distance_certainty = torch.exp(-distance_alpha * global_distance_aug / distnace_max)
                prev_weights = prev_weights * (distance_certainty + 0.5)

                if self.relative_scale:
                    global_distance_aug = distance_certainty
                    
            if self.uncertainty_config.global_prev_distribution:
                global_prev_distri_reverse = self.global_pred_distribs[idx].permute(0, 4, 3, 2, 1)
                global_prev_distribution = F.grid_sample(global_prev_distri_reverse, flip_global_coord_norm,
                                                         mode='bilinear', align_corners=True,
                                                         padding_mode='border').reshape(
                    global_prev_distri_reverse.shape[1], -1).T.reshape(*global_coord_norm_shape,
                                                                       global_prev_distri_reverse.shape[1]).unsqueeze(
                    1).to(dtype)
                global_prev_distribution = torch.clamp(global_prev_distribution, min=0.0, max=1.0)

                if self.embed_feature_uncertainty:
                    uncertainty_input = torch.cat(
                        [global_vis_time, global_distance_aug, global_prev_distribution, prev_voxel, voxel_query],
                        dim=2).squeeze(1)
                else:
                    uncertainty_input = torch.cat([global_vis_time, global_distance_aug, global_prev_distribution],
                                                  dim=2).squeeze(1)

            if self.w_au:
                au = self.global_au[idx].permute(0,3,2,1).unsqueeze(1)
                au = F.grid_sample(au.float(), flip_global_coord_norm,
                                            mode='bilinear', align_corners=True, 
                                            padding_mode='border').reshape(au.shape[1], -1).T.reshape(*global_coord_norm_shape, 
                                                                                                                au.shape[1]).unsqueeze(1).to(dtype)
                
                if self.au_ef:
                    uncertainty_input = torch.cat([uncertainty_input, au.squeeze(2)], dim=1)
                    au=None
                    
        _gt_flow = None        
        if self.disable_flow_init:
            _gt_flow = gt_flow if self.do_history else None
        else:
            _gt_flow = gt_flow
        fused_feat = self.temporal_encoder(
            voxel_query,  # query: current local feature
            prev_voxel,  # key: RoI in global buffer
            prev_voxel,  # value
            query_key_padding_mask=mask,
            bev_z=z,
            bev_h=h,
            bev_w=w,
            bev_pos=voxel_pos,
            spatial_shapes=torch.tensor([[z, h, w]], device=voxel_query.device),
            level_start_index=torch.tensor([0], device=voxel_query.device),
            prev_bev=prev_voxel,
            ref_points=ref_3d,
            transformed_ref_points=shift_ref_3d,
            prev_weights=prev_weights if not self.implicit_uncertainty and self.do_history else None,
            uncertainty_embed=uncertainty_input if self.implicit_uncertainty and self.do_history else None,
            au=au if self.w_au else None,
            gt_flow = _gt_flow,
        )
        
        fused_feat = fused_feat.permute(1, 0, 2).reshape(1, z, h, w, c_).permute(0, 4, 3, 2, 1)  # 1 c_ w h z
        flip_reverse_grid_norm = reverse_grid_norm.flip((-1,))
        reverse_grid_shape = flip_reverse_grid_norm.shape[:-1]
        flip_reverse_grid_norm = flip_reverse_grid_norm.reshape(1, 1, 1, -1, 3)
        update_voxel_feature = F.grid_sample(fused_feat, flip_reverse_grid_norm,
                                             mode='bilinear', align_corners=True).reshape(fused_feat.shape[1],
                                                                                          -1).T.reshape(
            *reverse_grid_shape,
            fused_feat.shape[1]).unsqueeze(1).permute(1, 2, 0)
        global_feat[:, :, global_pos[..., 0], global_pos[..., 1], global_pos[..., 2]] = update_voxel_feature

        # update of memory
        self.global_vis_cnt[idx][:, global_pos[..., 0], global_pos[..., 1], global_pos[..., 2]] += 1
        self.global_pos[idx] = global_pos
        self.global_flip_reverse_grid_norm[idx] = flip_reverse_grid_norm

        self.global_valid_masks[idx][:, global_coord[..., 0], global_coord[..., 1], global_coord[..., 2]] += 1

        return fused_feat.permute(0, 1, 3, 2, 4)  ##### try to align with the FB-OCC coordinate: 1 c_ h w z

    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        return_map = {}

        context = self.image_encoder(img[0])
        cam_params = img[1:7]
        if self.with_specific_component('depth_net'):
            mlp_input = self.depth_net.get_mlp_input(*cam_params)
            context, depth = self.depth_net(context, mlp_input)
            return_map['depth'] = depth
            return_map['context'] = context
        else:
            context = None
            depth = None

        if self.with_specific_component('forward_projection'):
            bev_feat = self.forward_projection(cam_params, context, depth, **kwargs)
            return_map['cam_params'] = cam_params
        else:
            bev_feat = None

        bev_mask = None

        if self.with_specific_component('backward_projection'):

            bev_feat_refined = self.backward_projection([context],
                                                        img_metas,
                                                        lss_bev=bev_feat.mean(-1),
                                                        cam_params=cam_params,
                                                        bev_mask=bev_mask,
                                                        gt_bboxes_3d=None,  # debug
                                                        pred_img_depth=depth)

            if self.readd:
                bev_feat = bev_feat_refined[..., None] + bev_feat
            else:
                bev_feat = bev_feat_refined

        if 'gt_flow' in kwargs.keys():
            bev_feat = self.update_global(bev_feat, img_metas, img[6], kwargs['gt_flow'])
        else:
            bev_feat = self.update_global(bev_feat, img_metas, img[6], self.gt_flow)
            
        bev_feat = self.bev_encoder(bev_feat)
        return_map['img_bev_feat'] = bev_feat

        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        results = {}
        if img is not None and self.with_specific_component('image_encoder'):
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)

        return results

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if self.with_pts_bbox:
            losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)

        # update the ground truth of global memory
        coord_float = self.global_coord_aug[..., 0].permute(0, 4, 3, 1, 2) * 2
        gt_xsize, gt_ysize, gt_zsize = kwargs['gt_occupancy'].shape[1:]
        upscale_coord = F.interpolate(coord_float, size=(gt_zsize, gt_ysize, gt_xsize), mode='trilinear',
                                      align_corners=True).permute(0, 3, 4, 2, 1)
        upscale_coord = torch.round(upscale_coord).long()
        # clamp upscale_coord according to the global memory size for each sample
        upscale_coord = torch.clamp(upscale_coord, min=0)
        upscale_coord = torch.clamp(upscale_coord,
                                    max=((torch.stack(self.ego_shape_maxes) + 1) * 2 - 1)[:, None, None, None, :])
        if self.precise_upscale:
            upscale_coord = self.global_coord_aug_2x[...,0].long()
            upscale_coord = torch.clamp(upscale_coord, min=0)
            upscale_coord = torch.clamp(upscale_coord, max=((torch.stack(self.ego_shape_maxes)+1) * 2 - 1)[:,None, None, None, :])
        occ_gt = kwargs['gt_occupancy']
        # occ_gt_flow
        condition_mask = (0 < occ_gt) & (occ_gt <= 18)
        speed_gt = None
        if 'gt_speed_map' in kwargs:
            speed_gt = kwargs['gt_speed_map'] # 200 200 16 2(vx, vy)
            # get the absolute speed of each voxel by sqrt(vx^2 + vy^2)
            absolute_speed = torch.sqrt(speed_gt[..., 0] ** 2 + speed_gt[..., 1] ** 2)
            dynamic_mask = absolute_speed > 0.5
            
            condition_mask = condition_mask & ~dynamic_mask
            free_mask = occ_gt == 255
        for i in range(upscale_coord.shape[0]):
            update_mask = (self.global_mask_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]],
                           upscale_coord[i][..., 1][condition_mask[i]],
                           upscale_coord[i][..., 2][condition_mask[i]]] == False).squeeze(0)

            # update every frame
            update_mask = torch.ones_like(update_mask).bool()
            
            if self.flickering_cnt:
                previous = self.global_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask], upscale_coord[i][..., 1][condition_mask[i]][update_mask], upscale_coord[i][..., 2][condition_mask[i]][update_mask]]
                previous_mask = previous != 0
                flickering_mask = previous_mask & (previous != occ_gt[i][condition_mask[i]][update_mask].int())
                flickering_cnt = torch.sum(flickering_mask)
                scene_name, frame_token = img_metas[i]['scene_name'], img_metas[i]['sample_idx']
                if scene_name not in self.flickering_cnt_gt:
                    self.flickering_cnt_gt[scene_name] = {}
                self.flickering_cnt_gt[scene_name][frame_token] = flickering_cnt.item()

            self.global_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask],
            upscale_coord[i][..., 1][condition_mask[i]][update_mask],
            upscale_coord[i][..., 2][condition_mask[i]][update_mask]] = occ_gt[i][condition_mask[i]][update_mask].int()
            self.global_mask_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]],
            upscale_coord[i][..., 1][condition_mask[i]], upscale_coord[i][..., 2][condition_mask[i]]] = True

        if self.with_specific_component('occupancy_head'):
            if self.w_au:
                losses_occupancy, occ_pred, au = self.occupancy_head.forward_train(results['img_bev_feat'], results=results,
                                                                            gt_occupancy=kwargs['gt_occupancy'],
                                                                            gt_occupancy_flow=kwargs['gt_flow'])
            else:
                losses_occupancy, occ_pred = self.occupancy_head.forward_train(results['img_bev_feat'], results=results,
                                                                            gt_occupancy=kwargs['gt_occupancy'],
                                                                            gt_occupancy_flow=kwargs['gt_flow'])
            losses.update(losses_occupancy)

        if self.use_depth_supervision and self.with_specific_component('depth_net'):
            loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)

        # used for uncertainty memory
        occ_res = occ_pred[:, 1:, ...].softmax(dim=1)  # logits 200 200 16 18
        occ_res_pred = occ_res.argmax(dim=1)

        bs = occ_res.shape[0]
        for i in range(bs):
            occ_distrib = occ_res[i].unsqueeze(0).permute(0, 1, 3, 2, 4)
            reverse_grid_shape = self.global_flip_reverse_grid_norm[i].shape[-2:-1]
            distrib = F.grid_sample(occ_distrib, self.global_flip_reverse_grid_norm[i],
                                    mode='bilinear', align_corners=True).reshape(occ_distrib.shape[1], -1).T.reshape(
                *reverse_grid_shape,
                occ_distrib.shape[1]).unsqueeze(1).permute(1, 2, 0)
            self.global_pred_distribs[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1],
            self.global_pos[i][..., 2]] *= 1 / self.uncertainty_config.global_distribu_decay
            # t=0 d_0
            # t=1 d_0 * 1/2 + d_1
            # t=2 d_0 * 1/4 + d_1 * 1/2 + d_2
            # t=3 d_0 * 1/8 + d_1 * 1/4 + d_2 * 1/2 + d_3
            self.global_pred_distribs[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1],
            self.global_pos[i][..., 2]] += distrib.permute(0, 2, 1) * (
                        1 - 1 / self.uncertainty_config.global_distribu_decay)
            # additional softmax
            self.global_pred_distribs[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1],
            self.global_pos[i][..., 2]] = self.global_pred_distribs[i][:, self.global_pos[i][..., 0],
                                          self.global_pos[i][..., 1], self.global_pos[i][..., 2]].softmax(dim=2)
            
            if self.w_au:
                au_distrib = au[i].unsqueeze(0).unsqueeze(0).permute(0,1,3,2,4)
                au_distrib = F.grid_sample(au_distrib, self.global_flip_reverse_grid_norm[i], 
                                                mode='bilinear', align_corners=True).reshape(au_distrib.shape[1], -1).T.reshape(*reverse_grid_shape, 
                                                                                                                                au_distrib.shape[1]).unsqueeze(1).permute(1,2,0)
                self.global_au[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1], self.global_pos[i][..., 2]] = au_distrib.permute(0,2,1).squeeze(2)

            if self.flickering_cnt:
                previous = self.global_pred[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask], upscale_coord[i][..., 1][condition_mask[i]][update_mask], upscale_coord[i][..., 2][condition_mask[i]][update_mask]]
                previous_mask = previous != 0
                flickering_mask = previous_mask & (previous != occ_res_pred[i][condition_mask[i]][update_mask].int())
                flickering_cnt = torch.sum(flickering_mask)
                scene_name, frame_token = img_metas[i]['scene_name'], img_metas[i]['sample_idx']
                # print("Flickering cnt for", scene_name, frame_token, img_metas[i]['global_idx'], ": pred", flickering_cnt.item(), " gt:" , self.flickering_cnt_gt[scene_name][frame_token])
                if scene_name not in self.flickering_cnt_pred:
                    self.flickering_cnt_pred[scene_name] = {}
                self.flickering_cnt_pred[scene_name][frame_token] = flickering_cnt.item()
            
                self.global_pred[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask], upscale_coord[i][..., 1][condition_mask[i]][update_mask], upscale_coord[i][..., 2][condition_mask[i]][update_mask]] = occ_res_pred[i][condition_mask[i]][update_mask].int()
                self.global_mask_pred[i][:, upscale_coord[i][..., 0][condition_mask[i]], upscale_coord[i][..., 1][condition_mask[i]], upscale_coord[i][..., 2][condition_mask[i]]] = True

        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        self.do_history = True
        self.is_test = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                              (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))
            num_augs = len(img_inputs)
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))
            if num_augs == 1 and not img_metas[0].data[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                return self.simple_test(points[0], img_metas[0].data[0], img_inputs[0],
                                        **kwargs)
            else:
                return self.aug_test(points, img_metas, img_inputs, **kwargs)

        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)

    def aug_test(self, points,
                 img_metas,
                 img_inputs=None,
                 visible_mask=[None],
                 **kwargs):
        """Test function without augmentaiton."""
        assert False
        return None

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        bbox_list = [dict() for _ in range(len(img_metas))]

        if self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]

        if self.with_specific_component('occupancy_head'):
            # update the ground truth of global memory
            coord_float = self.global_coord_aug[..., 0].permute(0, 4, 3, 1, 2) * 2
            gt_xsize, gt_ysize, gt_zsize = kwargs['gt_occupancy'][0].shape[1:]
            upscale_coord = F.interpolate(coord_float, size=(gt_zsize, gt_ysize, gt_xsize), mode='trilinear',
                                          align_corners=True).permute(0, 3, 4, 2, 1)
            upscale_coord = torch.round(upscale_coord).long()
            # clamp upscale_coord according to the global memory size for each sample
            upscale_coord = torch.clamp(upscale_coord, min=0)
            upscale_coord = torch.clamp(upscale_coord,
                                        max=((torch.stack(self.ego_shape_maxes) + 1) * 2 - 1)[:, None, None, None, :])
            if self.precise_upscale:
                upscale_coord = self.global_coord_aug_2x[...,0].long()
                upscale_coord = torch.clamp(upscale_coord, min=0)
                upscale_coord = torch.clamp(upscale_coord, max=((torch.stack(self.ego_shape_maxes)+1) * 2 - 1)[:,None, None, None, :])
            occ_gt = kwargs['gt_occupancy'][0]
            # occ_gt_flow
            condition_mask = (0 < occ_gt) & (occ_gt <= 18)
            speed_gt = None
            if 'gt_speed_map' in kwargs:
                speed_gt = kwargs['gt_speed_map'][0] # 200 200 16 2(vx, vy)
                # get the absolute speed of each voxel by sqrt(vx^2 + vy^2)
                absolute_speed = torch.sqrt(speed_gt[..., 0] ** 2 + speed_gt[..., 1] ** 2)
                dynamic_mask = absolute_speed > 0.5
                
                condition_mask = condition_mask & ~dynamic_mask
                free_mask = occ_gt == 255
                
            for i in range(upscale_coord.shape[0]):
                update_mask = (self.global_mask_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]],
                               upscale_coord[i][..., 1][condition_mask[i]],
                               upscale_coord[i][..., 2][condition_mask[i]]] == False).squeeze(0)

                # update every frame
                update_mask = torch.ones_like(update_mask).bool()
                
                if self.flickering_cnt:
                    previous = self.global_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask], upscale_coord[i][..., 1][condition_mask[i]][update_mask], upscale_coord[i][..., 2][condition_mask[i]][update_mask]]
                    previous_mask = previous != 0
                    flickering_mask = previous_mask & (previous != occ_gt[i][condition_mask[i]][update_mask].int())
                    flickering_cnt = torch.sum(flickering_mask)
                    flickering_class = occ_gt[i][condition_mask[i]][update_mask][flickering_mask.squeeze(0)].int()
                    # get the count for each class
                    flickering_cnt_class = torch.zeros(19, device=flickering_cnt.device)
                    for j in range(19):
                        flickering_cnt_class[j] = torch.sum(flickering_class == j)
                    total_voxels = torch.sum(occ_gt[i] != 255)
                    non_free_voxels = torch.sum(occ_gt[i][occ_gt[i] != 255] != 18)
                    
                    scene_name, frame_token = img_metas[i]['scene_name'], img_metas[i]['sample_idx']
                    if scene_name not in self.flickering_cnt_gt:
                        self.flickering_cnt_gt[scene_name] = {}
                        self.total_cnt_gt[scene_name] = {}
                        self.total_non_free_gt[scene_name] = {}
                        self.flickering_distri_gt[scene_name] = {}
                        self.flickering_distri_pred[scene_name] = {}
                        
                    self.flickering_cnt_gt[scene_name][frame_token] = flickering_cnt.item()
                    self.total_cnt_gt[scene_name][frame_token] = total_voxels.item()
                    self.total_non_free_gt[scene_name][frame_token] = non_free_voxels.item()
                    self.flickering_distri_gt[scene_name][frame_token] = flickering_cnt_class.cpu().numpy()

                self.global_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask],
                upscale_coord[i][..., 1][condition_mask[i]][update_mask],
                upscale_coord[i][..., 2][condition_mask[i]][update_mask]] = occ_gt[i][condition_mask[i]][
                    update_mask].int()
                self.global_mask_gt[i][:, upscale_coord[i][..., 0][condition_mask[i]],
                upscale_coord[i][..., 1][condition_mask[i]], upscale_coord[i][..., 2][condition_mask[i]]] = True

            occ_head_result = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)
            pred_occupancy = occ_head_result['output_voxels'][0]
            self.gt_flow = occ_head_result['output_flow'][0].permute(0, 2, 3, 4, 1)
            # ... #!!!!! comment the above line will result in inference using gt flow if  "gt_flow" key is collect3d in test_pipeline
            if self.w_au:
                au = occ_head_result['output_uncertainties']

            # used for uncertainty memory
            occ_res = pred_occupancy[:, 1:, ...].softmax(dim=1)
            occ_res_pred = occ_res.argmax(dim=1)

            bs = occ_res.shape[0]
            for i in range(bs):
                occ_distrib = occ_res[i].unsqueeze(0).permute(0, 1, 3, 2, 4)
                reverse_grid_shape = self.global_flip_reverse_grid_norm[i].shape[-2:-1]
                distrib = F.grid_sample(occ_distrib, self.global_flip_reverse_grid_norm[i],
                                        mode='bilinear', align_corners=True).reshape(occ_distrib.shape[1],
                                                                                     -1).T.reshape(*reverse_grid_shape,
                                                                                                   occ_distrib.shape[
                                                                                                       1]).unsqueeze(
                    1).permute(1, 2, 0)
                self.global_pred_distribs[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1],
                self.global_pos[i][..., 2]] *= 1 / self.uncertainty_config.global_distribu_decay
                self.global_pred_distribs[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1],
                self.global_pos[i][..., 2]] += distrib.permute(0, 2, 1) * (
                            1 - 1 / self.uncertainty_config.global_distribu_decay)
                # additional softmax
                self.global_pred_distribs[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1],
                self.global_pos[i][..., 2]] = self.global_pred_distribs[i][:, self.global_pos[i][..., 0],
                                            self.global_pos[i][..., 1], self.global_pos[i][..., 2]].softmax(dim=2)
                if self.w_au:
                    au_distrib = au[i].unsqueeze(0).unsqueeze(0).permute(0,1,3,2,4)
                    au_distrib = F.grid_sample(au_distrib, self.global_flip_reverse_grid_norm[i], 
                                                    mode='bilinear', align_corners=True).reshape(au_distrib.shape[1], -1).T.reshape(*reverse_grid_shape, 
                                                                                                                                    au_distrib.shape[1]).unsqueeze(1).permute(1,2,0)
                    self.global_au[i][:, self.global_pos[i][..., 0], self.global_pos[i][..., 1], self.global_pos[i][..., 2]] = au_distrib.permute(0,2,1).squeeze(2)

                if self.flickering_cnt:
                    previous = self.global_pred[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask], upscale_coord[i][..., 1][condition_mask[i]][update_mask], upscale_coord[i][..., 2][condition_mask[i]][update_mask]]
                    previous_mask = previous != 0
                    flickering_mask = previous_mask & (previous != occ_res_pred[i][condition_mask[i]][update_mask].int())
                    flickering_cnt = torch.sum(flickering_mask)
                    flickering_class = occ_gt[i][condition_mask[i]][update_mask][flickering_mask.squeeze(0)].int()
                    # get the count for each class
                    flickering_cnt_class = torch.zeros(19, device=flickering_cnt.device)
                    for j in range(19):
                        flickering_cnt_class[j] = torch.sum(flickering_class == j)
                    scene_name, frame_token = img_metas[i]['scene_name'], img_metas[i]['sample_idx']
                    # print("Flickering cnt for", scene_name, frame_token, img_metas[i]['global_idx'], ": pred", flickering_cnt.item(), " gt:" , self.flickering_cnt_gt[scene_name][frame_token])
                    if scene_name not in self.flickering_cnt_pred:
                        self.flickering_cnt_pred[scene_name] = {}
                    self.flickering_cnt_pred[scene_name][frame_token] = flickering_cnt.item()
                    self.flickering_distri_pred[scene_name][frame_token] = flickering_cnt_class.cpu().numpy()
                
                self.global_pred[i][:, upscale_coord[i][..., 0][condition_mask[i]][update_mask], upscale_coord[i][..., 1][condition_mask[i]][update_mask], upscale_coord[i][..., 2][condition_mask[i]][update_mask]] = occ_res_pred[i][condition_mask[i]][update_mask].int()
                self.global_mask_pred[i][:, upscale_coord[i][..., 0][condition_mask[i]], upscale_coord[i][..., 1][condition_mask[i]], upscale_coord[i][..., 2][condition_mask[i]]] = True

            pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
            if self.fix_void:
                pred_occupancy = pred_occupancy[..., 1:]
            pred_occupancy = pred_occupancy.softmax(-1)

            # convert to CVPR2023 Format
            pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
            pred_occupancy = torch.flip(pred_occupancy, [2])
            pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
            pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)

            if return_raw_occ:
                pred_occupancy_category = pred_occupancy
            else:
                pred_occupancy_category = pred_occupancy.argmax(-1)

            # For test server
            if self.occupancy_save_path is not None:
                scene_name = img_metas[0]['scene_name']
                sample_token = img_metas[0]['sample_idx']
                save_pred_occupancy = pred_occupancy.argmax(-1).cpu().numpy()
                save_path = os.path.join(self.occupancy_save_path, 'occupancy_pred', f'{sample_token}.npz')
                np.savez_compressed(save_path, save_pred_occupancy.astype(np.uint8))

            pred_occupancy_category = pred_occupancy_category.cpu().numpy()

        else:
            pred_occupancy_category = None

        iou = None

        assert len(img_metas) == 1
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
            if self.flickering_cnt:
                flickering = {}
                flickering['gt'] = self.flickering_cnt_gt
                flickering['pred'] = self.flickering_cnt_pred
                flickering['total'] = self.total_cnt_gt
                flickering['total_non_free'] = self.total_non_free_gt
                flickering['gt_dist'] = self.flickering_distri_gt
                flickering['pred_dist'] = self.flickering_distri_pred
                result_dict['flickering'] = flickering
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs