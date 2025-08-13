import csv
import os.path
import pickle

import numpy as np
import torch

from utils_other.render_util import *
import smplx
import pandas as pd
from tqdm import tqdm
import cv2
import PIL.Image as pil_img
from PIL import ImageDraw
import pyrender

from plyfile import PlyData

from plyfile import PlyData, PlyElement

from pose2transformation import get_rigid_transformation
from os.path import join as pjoin
import pickle as pkl
import glob
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import math

skip_num = 4
dataset_name = "prox"

# OpenPose 关键点的连接线定义（用于绘制骨架）
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9],
    [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
    [0, 15], [15, 17], [2, 16], [5, 17]
]

# 颜色定义 (BGR格式)
colors = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
    [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
    [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
    [255, 0, 255], [255, 0, 170], [255, 0, 85]
]

openpose_list = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65], dtype=np.int32)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def concat_images_horizontally(img1, img2):
    # 获取两张图片的尺寸
    width1, height1 = img1.size
    width2, height2 = img2.size

    # 拼接后的图片宽度为两张图片宽度之和，高度取两张图片中的较大值
    new_width = width1 + width2
    new_height = max(height1, height2)

    # 创建一个新的空白图片，背景默认为白色
    new_img = pil_img.new('RGB', (new_width, new_height), (255, 255, 255))

    # 将第一张图片粘贴到新图片的左边
    new_img.paste(img1, (0, 0))
    # 将第二张图片粘贴到新图片的右边
    new_img.paste(img2, (width1, 0))

    return new_img


def draw_skeleton(image, keypoints, confidence_threshold=0.2):
    """
    在给定的图像上绘制 OpenPose 的骨架结构。

    参数:
    image (numpy.ndarray): 输入图像。
    keypoints (list of lists): OpenPose 检测到的关键点，形状为 (25, 3)，每个关键点包含 (x, y, confidence)。
    confidence_threshold (float): 置信度阈值，低于该值的关键点不会被绘制。

    返回:
    annotated_image (numpy.ndarray): 绘制了骨架结构的图像。
    """
    # 将 PIL 图像转换为 OpenCV 格式 (numpy.ndarray)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 创建图像的副本以避免修改原始图像
    annotated_image = image.copy()

    for i in range(len(keypoints)):
        # 如果置信度低于阈值，则跳过该关键点
        if keypoints[i][2] <= confidence_threshold:
            continue

        if openpose_list[i] >= 22:
            continue

        # 绘制关键点
        cv2.circle(annotated_image, (int(keypoints[i][0]), int(keypoints[i][1])), 5, colors[i % len(colors)], -1)

    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    annotated_pil_image = pil_img.fromarray(annotated_image_rgb)
    return annotated_pil_image


def perspective_project_point_to_image(world_point, rotation_matrix, translation_vector, intrinsic_matrix):
    """
    将世界坐标系中的3维点透视投影到相机平面上的2维点。

    :param world_point: 世界坐标系中的3维点 (X, Y, Z)
    :param rotation_matrix: 旋转矩阵 R (3x3)
    :param translation_vector: 平移向量 t (3x1)
    :param intrinsic_matrix: 内参矩阵 K (3x3)
    :return: 图像平面上的2维点 (u, v)
    """
    # 将世界坐标系中的点转换到相机坐标系
    camera_point = (np.matmul(rotation_matrix, world_point.unsqueeze(-1)) + translation_vector[:, None]).squeeze()

    # 透视投影到图像平面
    homogeneous_camera_point = np.concatenate([camera_point[:, [0]], camera_point[:, [1]], camera_point[:, [2]]], axis=-1)
    image_point_homogeneous = np.matmul(intrinsic_matrix, homogeneous_camera_point[:, :, None]).squeeze()

    # 归一化
    u = image_point_homogeneous[:, 0] / image_point_homogeneous[:, 2]
    v = image_point_homogeneous[:, 1] / image_point_homogeneous[:, 2]

    return np.concatenate([u[:, None], v[:, None]], axis=-1)


# scene_list = ['MPH1Library_00034_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01',
#                  'N0Sofa_00145_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01',
#                  'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01',
#                  'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00150_01',
#                  'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']
scene_list = ["N0Sofa_00034_02"]

output_dir = "C:/RoHM/datasets/PROX/recordings/"
root_dir = "C:/RoHM/datasets/PROX/"

fps = 30
thresh_vel = 0.1
thresh_height = 0.1

r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1.0)

for scene in scene_list:

    print("process {}".format(scene))

    # lemo_result_list = os.listdir(os.path.join("C:/PROX_temporal/PROXD_temp/", scene, "results"))

    output_dir_scene = pjoin(output_dir, scene, "total_result")
    # output_dir_scene = pjoin(output_dir, scene, "tmp")
    os.makedirs(output_dir_scene, exist_ok=True)

    # file_list = glob.glob(pjoin(output_dir_scene, '*.jpg'))
    # if len(file_list) > 10:
    #     continue
    img_dir = pjoin(output_dir, scene, "Color")
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))

    camera_ins_info_path = pjoin(root_dir, 'calibration', 'Color.json')
    camera_exter_info_path = "C:/RoHM/datasets/PROX/cam2world/{}.json".format(scene.split("_")[0])

    with open(camera_exter_info_path, 'r') as f:
        cam2world = np.asarray(json.load(f))

    with open(pjoin(root_dir, "recordings", scene, "gender.txt"), 'r', encoding='utf-8') as file:
        # 读取文件内容
        gender_gt = file.read()

    gender_gt = "neutral"
    original_smplx_parameter_path = pjoin(output_dir, scene, "pose_info.pkl")

    with open(original_smplx_parameter_path, 'rb') as f:
        original_smplx_parameter = pkl.load(f)

    smplx_neutral = smplx.create(model_path="./body_models/smplx_model", model_type="smplx", gender=gender_gt, flat_hand_mean=True, use_pca=False).cuda()

    experiment_result_path = "./exp_total/prox_{}-direct-mlp_field-identity-shallow_mlp-default/ckpt100000.pth".format(scene)

    experiment_result_smplx_parameter = torch.load(experiment_result_path)

    # with open(camera_exter_info_path, 'r') as f:
    #     camera_exter_info = np.array(json.load(f)["trans"])
    camera_exter_info = cam2world
    with open(camera_ins_info_path, 'rb') as f:
        camera_ins_info = json.load(f)

    # with open(os.path.join(root_dir, 'kinect_cam_params', 'kinect_{}'.format(view_dict[scene]), 'Color.json'), 'r') as f:
    #     camera_ins_info = json.load(f)
    [f_x, f_y] = camera_ins_info['f']
    [c_x, c_y] = camera_ins_info['c']
    camera, camera_pose, light = create_render_cam(cam_x=c_x, cam_y=c_y, fx=f_x, fy=f_y)

    slices = slice(0, math.floor(len(original_smplx_parameter["beta"])), skip_num)
    # slices = slice(0, math.floor(len(modified_smplx_parameter[1]["pose_correction.root_orients.weight"])), skip_num)
    # slices = slice(0, math.floor(len(modified_smplx_parameter_refine["foot_contact"])), skip_num)
    # slices = slice(0, math.floor(len(vposer_smplx_parameter["pose_body"])), skip_num)
    img_files = img_files[slices]
    # lemo_result_list = lemo_result_list[slices]

    # joint_mask_path = os.path.join("C:/RoHM/datasets/PROX", 'mask_joint', scene, 'mask_joint.npy')
    # joint_mask = np.load(joint_mask_path)
    # # joint_mask = joint_mask[slices][:, :22]
    # joint_mask = joint_mask[:, :22]

    # todo
    joint_mask = original_smplx_parameter["mask_joint_vis"]

    # pass
    length = len(img_files)
    # modified_smplx_parameter
    modified_smplx_result = smplx_neutral.forward(
        global_orient=experiment_result_smplx_parameter[2]["pose_correction.root_orients.weight"][slices][:len(img_files)],
        transl=experiment_result_smplx_parameter[2]["pose_correction.trans.weight"][slices][:len(img_files)],
        body_pose=experiment_result_smplx_parameter[2]["pose_correction.pose_bodys.weight"][:, :63][slices][:len(img_files)],
        jaw_pose=experiment_result_smplx_parameter[2]["pose_correction.pose_bodys.weight"][:, 63:66][slices][:len(img_files)],
        leye_pose=experiment_result_smplx_parameter[2]["pose_correction.pose_bodys.weight"][:, 66:69][slices][:len(img_files)],
        reye_pose=experiment_result_smplx_parameter[2]["pose_correction.pose_bodys.weight"][:, 69:72][slices][:len(img_files)],
        left_hand_pose=experiment_result_smplx_parameter[2]["pose_correction.pose_bodys.weight"][:, 72:117][slices][:len(img_files)],
        right_hand_pose=experiment_result_smplx_parameter[2]["pose_correction.pose_bodys.weight"][:, 117:162][slices][:len(img_files)],
        expression=torch.zeros(length, 10).cuda(),
        betas=experiment_result_smplx_parameter[2]["pose_correction.betas.weight"][slices][:len(img_files)]
        # betas=torch.from_numpy(original_smplx_parameter["beta"][slices]).cuda()
    )
    modified_joints = modified_smplx_result.joints[:, :22, :].detach().cpu().squeeze()
    modified_vertic = modified_smplx_result.vertices.detach().cpu().squeeze()

    os.makedirs(os.path.join(root_dir, scene, "human_mesh"), exist_ok=True)


    # modified_smplx_parameter_refine
    length = len(img_files)

    # original_smplx_parameter
    original_smplx_result = smplx_neutral.forward(
        global_orient=torch.from_numpy(original_smplx_parameter["root_orient"][slices]).cuda(),
        transl=torch.from_numpy(original_smplx_parameter["transl"][slices]).cuda(),
        body_pose=torch.from_numpy(original_smplx_parameter["pose_body"][slices]).cuda(),
        jaw_pose=torch.zeros(len(img_files), 3).cuda(),
        leye_pose=torch.zeros(len(img_files), 3).cuda(),
        reye_pose=torch.zeros(len(img_files), 3).cuda(),
        left_hand_pose=torch.zeros(len(img_files), 45).cuda(),
        right_hand_pose=torch.zeros(len(img_files), 45).cuda(),
        expression=torch.zeros(len(img_files), 10).cuda(),
        # betas=modified_smplx_parameter[1]["pose_correction.betas"].unsqueeze(0).repeat(len(img_files), 1)
        betas=torch.from_numpy(original_smplx_parameter["beta"][slices]).cuda()
    )
    # original_joints = original_smplx_result.joints[:, :22, :].detach().cpu().squeeze()
    # original_vertic = original_smplx_result.vertices.detach().cpu().squeeze()
    original_joints = original_smplx_result.joints[:, :22, :].detach().cpu()
    original_vertic = original_smplx_result.vertices.detach().cpu()

    world2cam = np.linalg.inv(cam2world)
    original_joints2D = perspective_project_point_to_image(original_joints[0], world2cam[:3, :3], world2cam[:3, 3], np.array(camera_ins_info['camera_mtx']))

    foot_joint_index_list = [7, 10, 8, 11]  # contact lbl dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe
    joints_foot_rec = original_joints[:, foot_joint_index_list, :]
    modified_joints_foot_rec = modified_joints[:, foot_joint_index_list, :]
    # modified_joints_foot_rec_refine = modified_joints_refine[:, foot_joint_index_list, :]

    ground_height = prox_floor_height[scene.split("_")[0]]
    pene_dist = original_joints[:, [10, 11], 2] - ground_height
    modified_pene_dist = modified_joints[:, [10, 11], 2] - ground_height
    # modified_pene_dist_refine = modified_joints_refine[:, [10, 11], 2] - ground_height

    joints_feet_horizon_vel_rec = np.linalg.norm(joints_foot_rec[1:, :, [0, 1]] - joints_foot_rec[:-1, :, [0, 1]], axis=-1) * fps  # [n_seq, clip_len, 2]
    joints_feet_height_rec = joints_foot_rec[0:-1, :, 2].detach().cpu().numpy()  # [n_seq, clip_len, 2]

    modified_joints_feet_horizon_vel_rec = np.linalg.norm(modified_joints_foot_rec[1:, :, [0, 1]] - modified_joints_foot_rec[:-1, :, [0, 1]], axis=-1) * fps  # [n_seq, clip_len, 2]
    modified_joints_feet_height_rec = modified_joints_foot_rec[0:-1, :, 2].detach().cpu().numpy()  # [n_seq, clip_len, 2]

        # modified_joints_feet_horizon_vel_rec_refine = np.linalg.norm(modified_joints_foot_rec_refine[1:, :, [0, 1]] - modified_joints_foot_rec_refine[:-1, :, [0, 1]], axis=-1) * fps  # [n_seq, clip_len, 2]
        # modified_joints_feet_height_rec_refine = modified_joints_foot_rec_refine[0:-1, :, 2].detach().cpu().numpy()  # [n_seq, clip_len, 2]

    #################################################################################### under the ground detected info
    pene_freq = pene_dist < -0.05
    # modified_pene_freq = modified_pene_dist < -0.05
    # modified_pene_freq_refine = modified_pene_dist_refine < -0.05

    #################################################################################### foot skatting detected info
    joints_feet_height_rec = joints_feet_height_rec - ground_height
    skating_rec_left = (
                (joints_feet_horizon_vel_rec[:, 0] > thresh_vel) * (joints_feet_horizon_vel_rec[:, 1] > thresh_vel) * (
                    joints_feet_height_rec[:, 0] < (thresh_height + 0.05)) * (
                        joints_feet_height_rec[:, 1] < thresh_height))
    skating_rec_right = (joints_feet_horizon_vel_rec[:, 2] > thresh_vel) * (
                joints_feet_horizon_vel_rec[:, 3] > thresh_vel) * (
                                    joints_feet_height_rec[:, 2] < (thresh_height + 0.05)) * (
                                joints_feet_height_rec[:, 3] < thresh_height)

    skating_label = skating_rec_left * skating_rec_right
    # skating_label = np.concatenate([skating_rec_left[None, :], skating_rec_right[None, :]], axis=0).transpose()

    modified_joints_feet_height_rec = modified_joints_feet_height_rec - ground_height
    modified_skating_rec_left = ((modified_joints_feet_horizon_vel_rec[:, 0] > thresh_vel) * (modified_joints_feet_horizon_vel_rec[:, 1] > thresh_vel) * \
                        (modified_joints_feet_height_rec[:, 0] < (thresh_height + 0.05)) * (modified_joints_feet_height_rec[:, 1] < thresh_height))
    modified_skating_rec_right = (modified_joints_feet_horizon_vel_rec[:, 2] > thresh_vel) * (modified_joints_feet_horizon_vel_rec[:, 3] > thresh_vel) * \
                        (modified_joints_feet_height_rec[:, 2] < (thresh_height + 0.05)) * (modified_joints_feet_height_rec[:, 3] < thresh_height)

    modified_skating_label = modified_skating_rec_left * modified_skating_rec_right

    length = original_joints.shape[0]

    original_iou_true = []
    modify_iou_true = []

    original_spr = []
    modify_spr = []

    # image_scene = cv2.cvtColor(cv2.imread("C:/RoHM/datasets/PROX/recordings/N0Sofa_00034_02/Background_predicted_histogram/background.jpg"), cv2.COLOR_BGR2RGB)
    # if dataset_name == "prox":
    #     image_scene = cv2.undistort(image_scene, np.asarray(camera_ins_info['camera_mtx']), np.asarray(camera_ins_info['k']))
    #     image_scene = cv2.flip(image_scene, 1)
    #
    # image_scene = pil_img.fromarray((image_scene).astype(np.uint8))

    for t in range(length):

        # t = 298

        print("process {}-th", t)

        image = cv2.cvtColor(cv2.imread(img_files[t]), cv2.COLOR_BGR2RGB)

        image = cv2.undistort(image, np.asarray(camera_ins_info['camera_mtx']), np.asarray(camera_ins_info['k']))
        image = cv2.flip(image, 1)

        skeleton_mesh_rec_list_modify_2 = create_pyrender_skel(joints=modified_joints[t].numpy(),
                                                               # add_trans=None,
                                                               add_trans=np.linalg.inv(camera_exter_info),
                                                               mask_scheme='video', mask_joint_id=np.array(range(22)),
                                                               add_contact=False,
                                                               # under_ground_list=modified_skating_label[t-1] if t >= 1 else None
                                                               # gaussian_gradient=gaussian_gradient[t]
                                                               )
        body_mesh_rec_modify_2 = create_pyrender_mesh(verts=modified_vertic[t], faces=smplx_neutral.faces,
                                                    trans=camera_exter_info, material=material_body_rec_occ)

        skeleton_mesh_rec_list_modify_1 = create_pyrender_skel(joints=original_joints[t].numpy(),
                                                               # add_trans=None,
                                                               add_trans=np.linalg.inv(camera_exter_info),
                                                               mask_scheme='video', mask_joint_id=[],
                                                               add_contact=False
                                                               # , under_ground_list=skating_label[t-1] if t >= 1 else None
                                                               )

        body_mesh_rec_modify_1 = create_pyrender_mesh(verts=original_vertic[t], faces=smplx_neutral.faces,
                                                    trans=camera_exter_info, material=material_body_rec_vis)



        scene_rec_skel_modify_2 = create_pyrender_scene(camera, camera_pose, light)
        scene_rec_mesh_modify_2 = create_pyrender_scene(camera, camera_pose, light)
        scene_rec_skel_modify_1 = create_pyrender_scene(camera, camera_pose, light)
        scene_rec_mesh_modify_1 = create_pyrender_scene(camera, camera_pose, light)

        # for mesh in skeleton_mesh_rec_list_modify_3:
        #     scene_rec_skel_modify_3.add(mesh, 'pred_joint')
        # scene_rec_mesh_modify_3.add(body_mesh_rec_modify_3, "mesh")

        for mesh in skeleton_mesh_rec_list_modify_2:
            scene_rec_skel_modify_2.add(mesh, 'pred_joint')
        scene_rec_mesh_modify_2.add(body_mesh_rec_modify_2, "mesh")

        for mesh in skeleton_mesh_rec_list_modify_1:
            scene_rec_skel_modify_1.add(mesh, 'pred_joint')
        scene_rec_mesh_modify_1.add(body_mesh_rec_modify_1, "mesh")


        # img_rec_skel_modify_3 = render_img(r, scene_rec_skel_modify_3, alpha=1.0)
        # img_rec_mesh_modify_3 = render_img(r, scene_rec_mesh_modify_3, alpha=0.9)

        img_rec_skel_modify_2 = render_img(r, scene_rec_skel_modify_2, alpha=1.0)
        img_rec_mesh_modify_2 = render_img(r, scene_rec_mesh_modify_2, alpha=0.7)

        img_rec_skel_modify_1 = render_img(r, scene_rec_skel_modify_1, alpha=1.0)
        img_rec_mesh_modify_1 = render_img(r, scene_rec_mesh_modify_1, alpha=1.0)

        mesh_original = (cv2.cvtColor(np.array(img_rec_mesh_modify_1)[:, :, :3], cv2.COLOR_RGB2BGR).mean(-1) > 0)
        mesh_modify = (cv2.cvtColor(np.array(img_rec_mesh_modify_2)[:, :, :3], cv2.COLOR_RGB2BGR).mean(-1) > 0)

        render_img_rec = pil_img.fromarray((image).astype(np.uint8))
        render_img_rec_modify = pil_img.fromarray((image).astype(np.uint8))
        # render_img_rec.paste(img_rec_skel_modify_1, (0, 0), img_rec_skel_modify_1)
        # render_img_rec.paste(img_rec_skel_modify_2, (0, 0), img_rec_skel_modify_2)
        # render_img_rec.paste(img_rec_skel_modify_3, (0, 0), img_rec_skel_modify_3)
        # render_img_rec.paste(img_rec_mesh_modify_2, (0, 0), img_rec_mesh_modify_2)

        # a = pil_img.new('RGB', (1920, 1080), (255, 255, 255))
        # a.paste(img_rec_skel_modify_0, (0, 0), img_rec_skel_modify_0)
        # a.save("C:/sam2-main/test.jpg")
        # render_img_rec.paste(img_rec_mesh_modify_2, (0, 0), img_rec_mesh_modify_2)
        render_img_rec.paste(img_rec_mesh_modify_1, (0, 0), img_rec_mesh_modify_1)
        render_img_rec_modify.paste(img_rec_mesh_modify_2, (0, 0), img_rec_mesh_modify_2)

        # if t == 300 or t == 100 or t == 200 or t == 400 or t == 500 or t == 600:
        #     image_scene.paste(img_rec_mesh_modify_2, (0, 0), img_rec_mesh_modify_2)

        # render_img_rec.paste(img_rec_skel_modify_1, (0, 0), img_rec_skel_modify_1)
        # render_img_rec.paste(img_rec_mesh_modify_2, (0, 0), img_rec_mesh_modify_2)
        # render_img_rec.paste(img_rec_mesh_modify_2_new, (0, 0), img_rec_mesh_modify_2_new)

        # render_img_rec = draw_skeleton(render_img_rec, keypoint_openpose[t])
        # render_img_rec.paste(img_rec_mesh_modify_3, (0, 0), img_rec_mesh_modify_3)

        # diff_angle= np.round(diff_angle[t], 2)
        # gaussian_gradient = np.round(gaussian_gradient.astype(np.float64), 2)

        # joints2d_obs_tmp = np.load("C:/humor/joints2d_obs_tmp.npy")[0, t, :22, :]
        # joints2d_pred_tmp = np.load("C:/humor/joints2d_pred_tmp.npy")[0, t, :22, :]
        #
        # render_img_rec = draw_points_on_image(render_img_rec, joints2d_pred_tmp)
        # render_img_rec = draw_points_on_image(render_img_rec, original_joints2D, color=(0, 255, 0))
        render_img_rec_new = concat_images_horizontally(render_img_rec, render_img_rec_modify)
        render_img_rec_new.save(pjoin(output_dir_scene, img_files[t].split("\\")[-1]))