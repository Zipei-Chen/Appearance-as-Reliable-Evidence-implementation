import math
import os
import sys
import glob
import cv2
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import pickle as pkl
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from scene.cameras import Camera


import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
import trimesh
from os.path import join as pjoin
import pandas as pd
import json


class EMDBAddScene(Dataset):
    def __init__(self, cfg, split='train', is_coarse=False):
        super().__init__()
        self.cfg = cfg
        self.split = split

        self.root_dir = cfg.root_dir
        self.recording_name = cfg.recording_name
        self.subject = cfg.subject
        # self.train_frames = cfg.train_frames
        # self.val_frames = cfg.val_frames
        self.white_bg = cfg.white_background

        with open(os.path.join(self.root_dir, self.recording_name, "P0_10_outdoor_rom_data.pkl"), "rb") as f:
            camera_info = pkl.load(f)
        camera_ext = np.linalg.inv(camera_info["camera"]['extrinsics'])

        # with open(os.path.join(self.root_dir, self.subject, 'camera.pkl'), 'rb') as f:
        #     camera = pkl.load(f, encoding='latin1')
        world2cam = np.linalg.inv(camera_ext)

        self.K, self.R, self.T = self.get_KRTD(camera_info["camera"]['intrinsics'], world2cam)

        self.H, self.W = 1920, 1440
        self.h, self.w = cfg.img_hw

        self.faces = np.load('body_models/misc_smplx/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc_smplx/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc_smplx/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc_smplx/J_regressors.npz'))

        # self.faces = np.load('body_models/misc_smplx/faces.npz')['faces']
        # self.skinning_weights = dict(np.load('body_models/misc_smplx/skinning_weights_all.npz'))
        # self.posedirs = dict(np.load('body_models/misc_smplx/posedirs_all.npz'))
        # self.J_regressor = dict(np.load('body_models/misc_smplx/J_regressors.npz'))

        # if split == 'train':
        #     frames = self.train_frames
        # elif split == 'val':
        #     frames = self.val_frames
        # elif split == 'test':
        #     frames = self.cfg.test_frames[self.cfg.test_mode]
        # elif split == 'predict':
        #     frames = self.cfg.predict_frames
        # else:
        #     raise ValueError

        # start_frame, end_frame, sampling_rate = frames

        # if split == 'predict':
        #     predict_seqs = ['rotating_models',
        #                     'gLO_sBM_cAll_d14_mLO1_ch05_view1']
        #     predict_seq = self.cfg.get('predict_seq', 0)
        #     predict_seq = predict_seqs[predict_seq]
        #     model_files = sorted(glob.glob(os.path.join(subject_dir, predict_seq, '*.npz')))
        #     self.model_files = model_files
        #     frames = list(reversed(range(-len(model_files), 0)))
        #     if end_frame == 0:
        #         end_frame = len(model_files)
        #     frame_slice = slice(start_frame, end_frame, sampling_rate)
        #     model_files = model_files[frame_slice]
        #     frames = frames[frame_slice]
        # else:
        # frames = list(range(start_frame, end_frame, sampling_rate))
        # frame_slice = slice(start_frame, end_frame, sampling_rate)
        # model_files = [os.path.join(subject_dir, f'animnerf_models/{frame:06d}.npz') for frame in frames]
        # self.gender_block = self.gender_dict[self.recording_name]
        # with open(pjoin(self.root_dir, "kinect_color", self.recording_name, "gender.txt"), 'r') as file:
        #     self.gender_unblock = file.read()
        #
        # self.gender_block = "neutral"
        # self.gender_unblock = "neutral"
        # self.gender_dict[self.recording_name] = "neutral"

        self.gender_gt = camera_info['gender']

        with open(pjoin(self.root_dir, self.recording_name, "rohm_result", "pose_info.pkl"), 'rb') as f:
        # with open(pjoin(self.root_dir, "recordings", self.recording_name, "pose_info_refine0.pkl"), 'rb') as f:
            model_files = pkl.load(f)

        self.model_files = model_files

        self.frames = list(range(0, math.floor(model_files["transform_list"].shape[0]), 1))
        self.frame_slice = slice(0, math.floor(model_files["transform_list"].shape[0]), 1)
        # self.frames = list(range(248, 249, 1))
        # self.frame_slice = slice(248, 249, 1)

        self.data = []
        img_dir = pjoin(self.root_dir, self.recording_name, "images")
        mask_dir = pjoin(self.root_dir, self.recording_name, "mask_predict_deeplab")
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        img_files = img_files[self.frame_slice]
        mask_files = mask_files[self.frame_slice]

        # assert model_files["transform_list"].shape[0] == len(img_files) == len(mask_files)
        assert len(self.frames) == len(img_files) == len(mask_files)

        if split == 'predict':
            for d_idx, f_idx in enumerate(self.frames):
                model_file = model_files[d_idx]
                # get dummy gt...
                img_file = img_files[0]
                mask_file = mask_files[0]

                self.data.append({
                    'data_idx': d_idx,
                    'frame_idx': f_idx,
                    'img_file': img_file,
                    'mask_file': mask_file,
                    "extrinsics": world2cam[f_idx],
                    'model_file': model_file,
                    })
        else:
            for d_idx, f_idx in enumerate(self.frames):
                img_file = img_files[d_idx]
                mask_file = mask_files[d_idx]
                # model_file = model_files[d_idx]

                self.data.append({
                    'data_idx': d_idx,
                    'frame_idx': f_idx,
                    'img_file': img_file,
                    'mask_file': mask_file,
                    "extrinsics": world2cam[f_idx],
                    # 'model_file': model_file,
                })

        # self.frames = frames
        # self.model_files_list = model_files

        self.metadata = self.get_metadata(self.model_files, self.gender_gt)

        self.preload = cfg.get('preload', True)
        if self.preload:
            self.cameras = [self.getitem(idx) for idx in range(len(self))]

    @staticmethod
    def get_KRTD(camera_ins, camera_ext):
        K = np.zeros([3, 3], dtype=np.float32)
        K[0, 0] = camera_ins[0, 0]
        K[1, 1] = camera_ins[1, 1]
        K[:2, 2] = camera_ins[:2, 2]
        K[2, 2] = 1
        # R = np.eye(3, dtype=np.float32)
        # T = np.zeros([3, 1], dtype=np.float32)
        R = camera_ext[:, :3, :3]
        T = camera_ext[:, :3, 3]

        return K, R, T

    def get_metadata(self, model_files, gender):
        # data_paths = self.model_files
        # data_path = data_paths[0]

        cano_data = self.get_cano_smpl_verts(model_files, gender)
        if self.split != 'train':
            metadata = cano_data
            return metadata

        # start, end, step = self.train_frames
        frames = list(range(model_files["transform_list"].shape[0]))
        # if end == 0:
        #     end = len(frames)
        # frame_slice = slice(start, end, step)
        frame_slice = self.frame_slice
        frames = frames[frame_slice]

        frame_dict = {
            frame: i for i, frame in enumerate(frames)
        }

        metadata = {
            'faces': self.faces,
            'posedirs': self.posedirs,
            'J_regressor': self.J_regressor,
            'cameras_extent': 3.469298553466797, # hardcoded, used to scale the threshold for scaling/image-space gradient
            'frame_dict': frame_dict,
        }
        metadata.update(cano_data)
        if self.cfg.train_smpl:
            metadata.update(self.get_smpl_data(model_files))

        return metadata

    def get_cano_smpl_verts(self, model_files_block, gender):
        '''
            Compute star-posed SMPL body vertices.
            To get a consistent canonical space,
            we do not add pose blend shape
        '''
        # compute scale from SMPL body
        # model_dict = np.load(data_path)
        # gender = 'female' if 'female' in self.subject else 'male'

        # gender_block = self.gender_block
        # gender_unblock = self.gender_unblock

        # 3D models and points
        minimal_shape_block = model_files_block['minimal_shape']
        # minimal_shape_unblock = model_files_unblock['minimal_shape']
        # Break symmetry if given in float16:
        if minimal_shape_block.dtype == np.float16:
            minimal_shape_block = minimal_shape_block.astype(np.float32)
            minimal_shape_block += 1e-4 * np.random.randn(*minimal_shape_block.shape)
            # minimal_shape_unblock = minimal_shape_unblock.astype(np.float32)
            # minimal_shape_unblock += 1e-4 * np.random.randn(*minimal_shape_unblock.shape)
        else:
            minimal_shape_block = minimal_shape_block.astype(np.float32)
            # minimal_shape_unblock = minimal_shape_unblock.astype(np.float32)

        # Minimally clothed shape
        # J_regressor_block = self.J_regressor[gender][:55, :]
        J_regressor_block = self.J_regressor[gender]
        # J_regressor_unblock = self.J_regressor[gender_unblock][:55, :]
        # J_regressor_unblock = self.J_regressor[gender_unblock]

        Jtr_block = np.dot(J_regressor_block, minimal_shape_block)
        # Jtr_unblock = np.dot(J_regressor_unblock, minimal_shape_unblock)

        # skinning_weights_block = self.skinning_weights[gender][:, :55]
        skinning_weights_block = self.skinning_weights[gender]
        # skinning_weights_unblock = self.skinning_weights[gender_unblock][:, :55]
        # skinning_weights_unblock = self.skinning_weights[gender_unblock]

        # Get bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v_block = get_02v_bone_transforms(Jtr_block)
        # bone_transforms_02v_unblock = get_02v_bone_transforms(Jtr_unblock)

        T_block = np.matmul(skinning_weights_block, bone_transforms_02v_block.reshape([-1, 16])).reshape([-1, 4, 4])
        # T_unblock = np.matmul(skinning_weights_unblock, bone_transforms_02v_unblock.reshape([-1, 16])).reshape([-1, 4, 4])

        vertices_block = np.matmul(T_block[:, :3, :3], minimal_shape_block[..., np.newaxis]).squeeze(-1) + T_block[:, :3, -1]
        # vertices_unblock = np.matmul(T_unblock[:, :3, :3], minimal_shape_unblock[..., np.newaxis]).squeeze(-1) + T_unblock[:, :3, -1]

        coord_max = np.max(vertices_block, axis=0)
        coord_min = np.min(vertices_block, axis=0)
        padding_ratio = self.cfg.padding
        padding_ratio = np.array(padding_ratio, dtype=np.float)
        padding = (coord_max - coord_min) * padding_ratio
        coord_max += padding
        coord_min -= padding

        cano_mesh_block = trimesh.Trimesh(vertices=vertices_block.astype(np.float32), faces=self.faces)
        # cano_mesh_unblock = trimesh.Trimesh(vertices=vertices_unblock.astype(np.float32), faces=self.faces)

        return {
            'gender': gender,
            'smpl_verts': vertices_block.astype(np.float32),
            'minimal_shape': minimal_shape_block,
            'Jtr': Jtr_block,
            'skinning_weights': skinning_weights_block.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v_block,
            'cano_mesh': cano_mesh_block,

            # 'gender_block': gender_block,
            # 'smpl_verts_block': vertices_block.astype(np.float32),
            # 'minimal_shape_block': minimal_shape_block,
            # 'Jtr_block': Jtr_block,
            # 'skinning_weights_block': skinning_weights_block.astype(np.float32),
            # 'bone_transforms_02v_block': bone_transforms_02v_block,
            # 'cano_mesh_block': cano_mesh_block,
            #
            # 'gender_unblock': gender_unblock,
            # 'smpl_verts_unblock': vertices_unblock.astype(np.float32),
            # 'minimal_shape_unblock': minimal_shape_unblock,
            # 'Jtr_unblock': Jtr_unblock,
            # 'skinning_weights_unblock': skinning_weights_unblock.astype(np.float32),
            # 'bone_transforms_02v_unblock': bone_transforms_02v_unblock,
            # 'cano_mesh_unblock': cano_mesh_unblock,

            'coord_min': coord_min,
            'coord_max': coord_max,
            'aabb': AABB(coord_max, coord_min),
        }

    def get_smpl_data(self, model_files):
        # load all smpl parameters of the training sequence
        if self.split != 'train':
            return {}

        from collections import defaultdict
        smpl_data = defaultdict(list)

        model_dict_block = model_files

        for idx, (frame) in enumerate(zip(self.frames)):
            # if idx == 0:
            smpl_data['betas'].append(model_dict_block['beta'][frame].astype(np.float32))
                # smpl_data['betas_unblock'] = model_dict_unblock['beta'][frame].astype(np.float32)

            smpl_data['root_orient'].append(model_dict_block['root_orient'][frame].astype(np.float32))
            smpl_data['pose_body'].append(model_dict_block['pose_body'][frame].astype(np.float32))
            smpl_data['trans'].append(model_dict_block['transl'][frame].astype(np.float32))

            # smpl_data['root_orient_unblock'].append(model_dict_unblock['root_orient'][frame].astype(np.float32))
            # smpl_data['pose_body_unblock'].append(model_dict_unblock['pose_body'][frame].astype(np.float32))
            # smpl_data['trans_unblock'].append(model_dict_unblock['transl'][frame].astype(np.float32))
            # smpl_data['pose_hand'].append(model_dict['pose_hand'].astype(np.float32))
            smpl_data['frames'].append(frame)
            smpl_data['pose_hand'].append(np.zeros(6).astype(np.float32))

        return smpl_data

    def __len__(self):
        return len(self.data)

    def getitem(self, idx):
        print("current idx:{} total:{}", idx, len(self.data))
        data_dict = self.data[idx]
        data_idx = data_dict['data_idx']
        frame_idx = data_dict['frame_idx']
        img_file = data_dict['img_file']
        mask_file = data_dict['mask_file']
        self.R = data_dict['extrinsics'][:3, :3]
        self.T = data_dict['extrinsics'][:3, 3]
        # model_file = data_dict['model_file']
        model_file = self.model_files

        K = self.K.copy()
        R = self.R.copy()
        T = self.T.copy()

        # note that in ZJUMoCap the camera center does not align perfectly in the intrinsic
        # here we try to offset it by modifying the extrinsic...
        M = np.eye(3)
        M[0, 2] = (K[0, 2] - self.W / 2) / K[0, 0]
        M[1, 2] = (K[1, 2] - self.H / 2) / K[1, 1]
        K[0, 2] = self.W / 2
        K[1, 2] = self.H / 2
        R = M @ R
        T = M @ T

        R = np.transpose(R)
        # T = T[:, 0]

        image = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        # image = cv2.undistort(image, K, dist, None)
        # mask = cv2.undistort(mask, K, dist, None)

        # # prox数据集需要翻转，不清楚原因
        # image = cv2.flip(image, 1)
        # mask = cv2.flip(mask, 1)

        lanczos = self.cfg.get('lanczos', False)
        interpolation = cv2.INTER_LANCZOS4 if lanczos else cv2.INTER_LINEAR

        image = cv2.resize(image, (self.w, self.h), interpolation=interpolation)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        # mask = mask != 0
        # mask[:15, :] = True
        # mask[-15:, :] = True
        # mask[:, :15] = True
        # mask[:, -15:] = True

        mask = ~mask
        # image[~mask] = 255. if self.white_bg else 0.
        image = image / 255.

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        # update camera parameters
        K[0, :] *= (self.w / self.W)
        K[1, :] *= (self.h / self.H)

        focal_length_x = K[0, 0]
        focal_length_y = K[1, 1]
        FovY = focal2fov(focal_length_y, self.h)
        FovX = focal2fov(focal_length_x, self.w)

        # Compute posed SMPL body
        minimal_shape = self.metadata['minimal_shape']
        gender = self.metadata['gender']

        # model_dict = np.load(model_file)
        model_dict = model_file

        n_smpl_points = minimal_shape.shape[0]
        trans_block = model_dict['transl'][frame_idx].astype(np.float32)
        # bone_transforms_block = (model_dict_block['transform_list'][frame_idx]).cpu().numpy().astype(np.float32)
        bone_transforms_block = (model_dict['transform_list'][frame_idx]).astype(np.float32)
        root_orient_block = model_dict['root_orient'][frame_idx].astype(np.float32)
        pose_body_block = model_dict['pose_body'][frame_idx].astype(np.float32)

        # pose_hand = model_dict['pose_hand'].astype(np.float32)
        pose_hand = np.zeros(6).astype(np.float32)
        # Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)

        pose_block = np.concatenate([root_orient_block, pose_body_block], axis=-1)
        pose_block = Rotation.from_rotvec(pose_block.reshape([-1, 3]))
        pose_mat_full_block = pose_block.as_matrix()  # 24 x 3 x 3
        pose_mat_block = pose_mat_full_block[1:, ...].copy()  # 23 x 3 x 3
        pose_rot_block = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose_mat_block], axis=0).reshape([-1, 9])  # 24 x 9, root rotation is set to identity
        pose_rot_full_block = pose_mat_full_block.reshape([-1, 9])  # 24 x 9, including root rotation

        # Minimally clothed shape
        # posedir = self.posedirs[gender]
        Jtr_block = self.metadata['Jtr']

        # canonical SMPL vertices without pose correction, to normalize joints
        center_block = np.mean(minimal_shape, axis=0)
        minimal_shape_centered_block = minimal_shape - center_block
        cano_max_block = minimal_shape_centered_block.max()
        cano_min_block = minimal_shape_centered_block.min()
        padding_block = (cano_max_block - cano_min_block) * 0.05

        # compute pose condition
        Jtr_norm_block = Jtr_block - center_block
        Jtr_norm_block = (Jtr_norm_block - cano_min_block + padding_block) / (cano_max_block - cano_min_block) / 1.1
        Jtr_norm_block -= 0.5
        Jtr_norm_block *= 2.

        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh
        # without global translation
        bone_transforms_02v_block = self.metadata['bone_transforms_02v']
        bone_transforms_block = bone_transforms_block @ np.linalg.inv(bone_transforms_02v_block)
        bone_transforms_block = bone_transforms_block.astype(np.float32)
        # bone_transforms_block[:, :3, 3] += trans_block  # add global offset



        return Camera(
            frame_id=frame_idx,
            K=K, R=R, T=T,
            FoVx=FovX,
            FoVy=FovY,
            image=image,
            mask=mask,
            gt_alpha_mask=None,
            image_name=f"{frame_idx:05d}",
            data_device=self.cfg.data_device,
            # human params
            rots=torch.from_numpy(pose_rot_block).float().unsqueeze(0),
            Jtrs=torch.from_numpy(Jtr_norm_block).float().unsqueeze(0),
            bone_transforms=torch.from_numpy(bone_transforms_block),

        )

    def __getitem__(self, idx):
        if self.preload:
            return self.cameras[idx]
        else:
            return self.getitem(idx)

    def readPointCloud(self, is_scene=False, is_block=False):
        if self.cfg.get('random_init', False):
            ply_path = os.path.join(self.root_dir, self.subject, 'random_pc.ply')

            aabb = self.metadata['aabb']
            coord_min = aabb.coord_min.unsqueeze(0).numpy()
            coord_max = aabb.coord_max.unsqueeze(0).numpy()
            n_points = 50_000

            xyz_norm = np.random.rand(n_points, 3)
            xyz = xyz_norm * coord_min + (1. - xyz_norm) * coord_max
            rgb = np.ones_like(xyz) * 255
            storePly(ply_path, xyz, rgb)

            pcd = fetchPly(ply_path)
        else:
            if is_scene:
                ply_path = os.path.join(self.root_dir, self.recording_name, "tmp_result_scene", "50000", "deformed_pose_add_color.ply")
                pcd = fetchPly(ply_path)
            else:
                ply_path = os.path.join(self.root_dir, self.recording_name, 'cano_smpl_block.ply')
                # try:
                #     pcd = fetchPly(ply_path)
                # except:
                verts = self.metadata['smpl_verts']
                faces = self.faces
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
                n_points = 50_000
                # n_points = 10475

                # xyz = mesh.sample(n_points)
                xyz = verts
                rgb = np.ones_like(xyz) * 255
                storePly(ply_path, xyz, rgb)

                pcd = fetchPly(ply_path)

        return pcd
