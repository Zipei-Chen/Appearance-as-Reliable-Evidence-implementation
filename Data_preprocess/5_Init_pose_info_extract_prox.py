import os.path
import pickle

import numpy as np
import torch

# from data_loaders.motion_representation import *
from utils_other import dist_util
from utils_other.vis_util import *
from utils_other.render_util import *
from utils_other.other_utils import update_globalRT_for_smplx
import smplx
import pandas as pd
from tqdm import tqdm
import configargparse
import cv2
import PIL.Image as pil_img
import pyrender
from os.path import join as pjoin
import pickle as pkl

# def points_coord_trans(xyz_source_coord, trans_mtx):
#     # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
#     xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose(0, 1))  # [N, 3]
#     xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
#     return xyz_target_coord


arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
group = configargparse.ArgParser(formatter_class=arg_formatter,
                                 prog='')

group.add_argument("--device", default=0, type=int, help="Device id to use.")
group.add_argument('--body_model_path', type=str, default='data/body_models/smplx_model', help='path to smplx model')
group.add_argument('--dataset', type=str, default='prox')
group.add_argument('--dataset_root', type=str, default='datasets/PROX', help='path to dataset')
group.add_argument('--saved_data_dir', type=str,
                   default='test_results/results_prox_rgb',  #
                   help='path to saved test results')


args = group.parse_args()
dist_util.setup_dist(args.device)


if __name__ == "__main__":

    test_recording_name_list = \
        ['MPH1Library_00034_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01',
         'N0Sofa_00145_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01',
         'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01',
         'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00150_01',
         'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']
        # test_recording_name_list = ["N0Sofa_00034_02"]

    ################################# evaluate metrics
    skating_list = {}
    acc_list = {}
    acc_error_list = {}
    ground_pene_dist_list = {}
    ground_pene_freq_list = {}
    gmpjpe_list = {}
    mpjpe_list = {}
    mpjpe_list_vis = {}
    mpjpe_list_occ = {}
    joint_mask_list = {}
    for recording_name in test_recording_name_list:

        image_data_root = pjoin(pjoin(args.dataset_root, "recordings"), recording_name)

        with open(pjoin(image_data_root, "gender.txt"), 'r') as file:
            gender = file.read()
            print(gender)

        smplx_neutral = smplx.create(model_path=args.body_model_path, model_type="smplx",
                                     gender=gender, flat_hand_mean=True, use_pca=False).to(dist_util.dev())


        cam2world_dir = os.path.join(args.dataset_root, 'cam2world')
        scene_name = recording_name.split("_")[0]
        with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
            cam2world = np.array(json.load(f))

        calibration_dir = os.path.join(args.dataset_root, 'calibration')
        with open(os.path.join(calibration_dir, "Color" + '.json'), 'r') as f:
            calibration = json.load(f)
            camera_c = calibration["c"]
            camera_f = calibration["f"]
            camera_k = calibration["k"]
            camera_mtx = calibration["camera_mtx"]

        ################################# read test results data
        saved_data_path = '{}/{}.pkl'.format(args.saved_data_dir, recording_name)
        with open(saved_data_path, 'rb') as f:
            saved_data = pickle.load(f)

        repr_name_list = saved_data['repr_name_list']
        repr_dim_dict = saved_data['repr_dim_dict']
        frame_name_list = saved_data['frame_name_list'] if args.dataset == 'egobody' else None
        # rec_ric_data_noisy_list = saved_data['rec_ric_data_noisy_list']
        joints_gt_scene_coord_list = saved_data['joints_gt_scene_coord_list'] if args.dataset == 'egobody' else None
        # rec_ric_data_rec_list_from_smpl = saved_data['rec_ric_data_rec_list_from_smpl']
        # joints_input_scene_coord_list = saved_data['joints_input_scene_coord_list']
        motion_repr_rec_list = saved_data['motion_repr_rec_list']
        # motion_repr_noisy_list = saved_data['motion_repr_noisy_list']
        mask_joint_vis_list = saved_data['mask_joint_vis_list']  # [n_clip, 143, 22]
        trans_scene2cano_list = saved_data['trans_scene2cano_list']
        n_seq = len(motion_repr_rec_list)
        clip_len_rec = motion_repr_rec_list.shape[1]
        print('n_seq:', n_seq)
        print('clip_len_rec:', clip_len_rec)
        joints_gt_scene_coord_list = joints_gt_scene_coord_list[:, 0:clip_len_rec] if args.dataset == 'egobody' else None

        ################ get contact lbls
        contact_lbl_rec_list = motion_repr_rec_list[:, :, -4:]  # np, [n_seq, clip_len, 4]
        contact_lbl_rec_list[contact_lbl_rec_list > 0.5] = 1.0
        contact_lbl_rec_list[contact_lbl_rec_list <= 0.5] = 0.0

        # smplx_init_model = smplx_neutral.forward()
        cur_total_dim = 0
        repr_dict_rec = {}
        for repr_name in repr_name_list:
            repr_dict_rec[repr_name] = motion_repr_rec_list[0:1, ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
            repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
            cur_total_dim += repr_dim_dict[repr_name]
        betas_init = repr_dict_rec["smplx_betas"][:, 0, :]
        smplx_init_model = smplx_neutral.forward(betas=betas_init)

        foot_contact_list = contact_lbl_rec_list
        minimal_shape = smplx_init_model.vertices.cpu().detach().numpy()
        mask_joint_vis_list = mask_joint_vis_list
        beta_list = []
        transl_list = []
        root_orient_list = []
        pose_body_list = []
        transform_list = []

        world2cam = np.linalg.inv(cam2world)
        with (torch.no_grad()):
            for idx in tqdm(range(n_seq)):
                cur_total_dim = 0
                repr_dict_rec = {}

                for repr_name in repr_name_list:
                    repr_dict_rec[repr_name] = motion_repr_rec_list[idx:(idx + 1), ..., cur_total_dim:(cur_total_dim + repr_dim_dict[repr_name])]
                    repr_dict_rec[repr_name] = torch.from_numpy(repr_dict_rec[repr_name]).to(dist_util.dev())
                    cur_total_dim += repr_dim_dict[repr_name]

                # joints_rec, smpl_verts_rec = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_neutral, return_verts=True, return_full_joints=True)

                # length = repr_dict_input["smplx_rot_6d"].shape[1]

                pose_rotation_6d = torch.cat([repr_dict_rec["smplx_rot_6d"], repr_dict_rec["smplx_body_pose_6d"]], dim=-1) .squeeze()
                pose_rotation_6d = rot6d_to_rotmat(pose_rotation_6d.reshape(clip_len_rec, -1, 6)).reshape(clip_len_rec, 22, 3, 3)
                pose_rotation = rotation_matrix_to_angle_axis(pose_rotation_6d)

                cano_joints = smplx_neutral(
                    global_orient=pose_rotation[:, 0, :].cuda(),
                    transl=repr_dict_rec["smplx_trans"].squeeze().cuda(),
                    body_pose=pose_rotation[:, 1:, :].cuda(),
                    jaw_pose=torch.zeros(len(pose_rotation), 3).to(pose_rotation.device),
                    leye_pose=torch.zeros(len(pose_rotation), 3).to(pose_rotation.device),
                    reye_pose=torch.zeros(len(pose_rotation), 3).to(pose_rotation.device),
                    left_hand_pose=torch.zeros(len(pose_rotation), 45).to(pose_rotation.device),
                    right_hand_pose=torch.zeros(len(pose_rotation), 45).to(pose_rotation.device),
                    expression=torch.zeros(len(pose_rotation), 10).to(pose_rotation.device),
                    # betas=betas.repeat(len(pose_rotation), 1).to(pose_rotation.device)
                    betas=repr_dict_rec["smplx_betas"].squeeze().cuda(),
                ).joints

                cano2scene = torch.from_numpy(np.linalg.inv(trans_scene2cano_list[idx]))

                delta_T = cano_joints[:, 0].detach().cpu().numpy() - repr_dict_rec["smplx_trans"].squeeze().cpu().numpy()

                smplx_params_dict = {
                    "global_orient" : pose_rotation[:, 0, :].cpu(),
                    "transl": repr_dict_rec["smplx_trans"].squeeze().cpu(),
                    "body_pose": pose_rotation[:, 1:, :].cpu(),
                    "betas": repr_dict_rec["smplx_betas"].squeeze().cpu()
                }

                smplx_params_dict_scene = update_globalRT_for_smplx(smplx_params_dict, cano2scene.detach().cpu().numpy(),
                                                                    delta_T=delta_T)

                transl = smplx_params_dict_scene["transl"]
                root_orient = torch.from_numpy(smplx_params_dict_scene["global_orient"].astype(np.float32))
                pose_body = smplx_params_dict_scene["body_pose"]
                betas = smplx_params_dict_scene["betas"]

                transform_matrix = smplx_neutral.forward(
                    global_orient=root_orient.cuda(),
                    transl=torch.from_numpy(transl).cuda(),
                    body_pose=pose_rotation[:, 1:],
                    jaw_pose=torch.zeros(len(pose_rotation), 3).to(pose_rotation.device),
                    leye_pose=torch.zeros(len(pose_rotation), 3).to(pose_rotation.device),
                    reye_pose=torch.zeros(len(pose_rotation), 3).to(pose_rotation.device),
                    left_hand_pose=torch.zeros(len(pose_rotation), 45).to(pose_rotation.device),
                    right_hand_pose=torch.zeros(len(pose_rotation), 45).to(pose_rotation.device),
                    expression=torch.zeros(len(pose_rotation), 10).to(pose_rotation.device),
                    # betas=torch.from_numpy(betas.numpy()).cuda(),
                    betas=betas_init.repeat(len(pose_rotation), 1).to(pose_rotation.device)
                ).A

                # transform_matrix = transform_matrix[:, :22].cpu()
                transform_matrix = transform_matrix.cpu()

                beta_list.append(betas)
                transl_list.append(transl)
                root_orient_list.append(root_orient)
                pose_body_list.append(pose_body)
                transform_list.append(transform_matrix.cpu())

        # camera
        camera_info = {
            "camera_c": camera_c,
            "camera_f": camera_f,
            "camera_k": camera_k,
            "camera_mtx": camera_mtx,
            "camera_r": world2cam[:3, :3],
            "camera_t": world2cam[:3, 3]
        }

        pose_info = {
            "foot_contact": foot_contact_list.reshape(-1, 4),
            "minimal_shape": minimal_shape.squeeze(),
            "mask_joint_vis": mask_joint_vis_list.reshape(-1, 22),
            "beta": np.concatenate(beta_list, axis=0).reshape(-1, 10),
            "transl": np.concatenate(transl_list, axis=0).reshape(-1, 3),
            "root_orient": np.concatenate(root_orient_list, axis=0).reshape(-1, 3),
            "pose_body": np.concatenate(pose_body_list, axis=0).reshape(-1, 63),
            "transform_list": np.concatenate(transform_list, axis=0),
        }

        pkl_path = os.path.join(image_data_root, 'camera_info.pkl')
        with open(pkl_path, 'wb') as result_file:
            pickle.dump(camera_info, result_file, protocol=2)

        pkl_path = os.path.join(image_data_root, 'pose_info.pkl')
        with open(pkl_path, 'wb') as result_file:
            pickle.dump(pose_info, result_file, protocol=2)
