import os.path
import pickle

import numpy as np
import torch

from utils_other.vis_util import *
from utils_other.render_util import *
import smplx
import pandas as pd
from tqdm import tqdm
import configargparse
import cv2
import PIL.Image as pil_img
import pyrender

from plyfile import PlyData

from plyfile import PlyData, PlyElement

from pose2transformation import get_rigid_transformation
from os.path import join as pjoin
import pickle as pkl
import glob


def create_pyrender_mesh(verts, faces, trans, material=None, vertex_colors=None, is_refine=False, ):
    with open('C:/3dgs-avatar-release/smplx_parts_segm.pkl', 'rb') as file:
        smplx_parts_segm = pkl.load(file, encoding='latin1')
    # segment = np.where((smplx_parts_segm["segm"] == 15) | (smplx_parts_segm["segm"] == 12) | (smplx_parts_segm["segm"] == 9) | (smplx_parts_segm["segm"] == 3) | (smplx_parts_segm["segm"] == 6) | (smplx_parts_segm["segm"] == 22) | (smplx_parts_segm["segm"] == 23) | (smplx_parts_segm["segm"] == 24))[0]
    segment = np.where((smplx_parts_segm["segm"] == 7) | (smplx_parts_segm["segm"] == 8) | (smplx_parts_segm["segm"] == 10) | (smplx_parts_segm["segm"] == 11))[0]

    body = trimesh.Trimesh(verts, faces, vertex_colors=vertex_colors, process=False)
    # body = body.submesh([segment.tolist()], append=True)
    body.apply_transform(np.linalg.inv(trans))
    if not is_refine:
        body_mesh = pyrender.Mesh.from_trimesh(body, material=material)
    else:
        body_mesh = pyrender.Mesh.from_trimesh(body, material=material_contact_1)
    return body_mesh


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


scene_list = ['MPH1Library_00034_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01',
                 'N0Sofa_00145_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01',
                 'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01',
                 'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00150_01',
                 'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']


r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1.0)

output_dir = "C:/RoHM/datasets/PROX/recordings/"

for scene in scene_list:
    print("process {}".format(scene))

    with open(pjoin(output_dir, scene, "gender.txt"), 'r') as file:
        gender_gt = file.read()
    smplx_neutral = smplx.create(model_path="./body_models/smplx_model", model_type="smplx", gender=gender_gt,
                                 flat_hand_mean=True, use_pca=False).cuda()

    img_dir = pjoin("C:/RoHM/datasets/PROX/recordings", scene, "Color")
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    # slices = slice(0, math.floor(len(img_files)), 4)

    camera_ins_info_path = pjoin("C:/RoHM/datasets/PROX/", 'calibration', 'Color.json')
    camera_exter_info_path = "C:/RoHM/datasets/PROX/cam2world/{}.json".format(scene.split("_")[0])
    original_smplx_parameter_path = pjoin("C:/RoHM/datasets/PROX/recordings", scene, "pose_info.pkl")
    # original_smplx_parameter_path = pjoin("C:/RoHM/datasets/PROX/recordings", scene, "coarse_pose_info_rgbd.pkl")

    with open(camera_exter_info_path, 'r') as f:
        camera_exter_info = np.array(json.load(f))

    with open(camera_ins_info_path, 'rb') as f:
        camera_ins_info = json.load(f)
    [f_x, f_y] = camera_ins_info['f']
    [c_x, c_y] = camera_ins_info['c']

    with open(original_smplx_parameter_path, 'rb') as f:
        original_smplx_parameter = pkl.load(f)

    img_files = img_files[:len(original_smplx_parameter["foot_contact"])]
    slices = slice(0, len(img_files), 5)
    img_files = img_files[slices]

    for i, image_name in enumerate(img_files):
        camera, camera_pose, light = create_render_cam(cam_x=c_x, cam_y=c_y, fx=f_x, fy=f_y)

        img_files = image_name

        # original_smplx_parameter
        original_smplx_result = smplx_neutral.forward(
            global_orient=torch.from_numpy(original_smplx_parameter["root_orient"][slices][i]).unsqueeze(0).cuda(),
            transl=torch.from_numpy(original_smplx_parameter["transl"][slices][i]).unsqueeze(0).cuda(),
            body_pose=torch.from_numpy(original_smplx_parameter["pose_body"][slices][i]).unsqueeze(0).cuda(),
            jaw_pose=torch.zeros(1, 3).cuda(),
            leye_pose=torch.zeros(1, 3).cuda(),
            reye_pose=torch.zeros(1, 3).cuda(),
            left_hand_pose=torch.zeros(1, 45).cuda(),
            right_hand_pose=torch.zeros(1, 45).cuda(),
            expression=torch.from_numpy(original_smplx_parameter["beta"][slices][i]).unsqueeze(0).cuda()
        )
        original_vertic = original_smplx_result.vertices.detach().cpu().squeeze()

        rgb = np.ones_like(original_vertic) * 255
        # os.makedirs(pjoin("C:/RoHM/datasets/PROX/recordings", scene, "human_point_cloud"), exist_ok=True)
        # if i % 20 == 0:
        #     storePly(pjoin("C:/RoHM/datasets/PROX/recordings", scene, "human_point_cloud", "first_human_{}.ply".format(i)), original_vertic, rgb)

        print("process {}-th", i)

        body_mesh_rec_modify_1 = create_pyrender_mesh(verts=original_vertic[:, :], faces=smplx_neutral.faces,
                                                    trans=camera_exter_info, material=material_body_rec_vis)

        scene_rec_mesh_modify_1 = create_pyrender_scene(camera, camera_pose, light)

        scene_rec_mesh_modify_1.add(body_mesh_rec_modify_1, "mesh")

        # img_rec_skel_modify_1 = render_img(r, scene_rec_skel_modify_1, alpha=1.0)
        depth = render_img_depth(r, scene_rec_mesh_modify_1, alpha=1)

        os.makedirs(pjoin("C:/RoHM/datasets/PROX/recordings", scene, "human_human"), exist_ok=True)
        np.save(pjoin("C:/RoHM/datasets/PROX/recordings", scene, "human_human", "depth_{}.npy".format(i)), depth)
