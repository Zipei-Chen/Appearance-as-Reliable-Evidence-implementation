import argparse
import copy

import cv2
import glob
import matplotlib
import numpy
import numpy as np
import os
import torch
from os.path import join as pjoin
from PIL import Image
import open3d as o3d
import torchvision.transforms as T
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2
import json
import pandas as pd
import cvxpy as cp
from sklearn.cluster import KMeans
import torch.optim as optim
import matplotlib.pyplot as plt


def row(A):
    return A.reshape((1, -1))


def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord


def kmeans_visualize(data):
    cmap = plt.get_cmap('tab10', 10)
    plt.figure(figsize=(12, 6))
    im = plt.imshow(data, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ticks=range(10))  # 添加 colorbar，并设置刻度为 0-9
    plt.title("Discrete Color Visualization for Values 0-9")
    plt.show()


print(torch.__version__)

test_list = ['MPH1Library_00034_01', 'N0Sofa_00034_01', 'N0Sofa_00034_02', 'N0Sofa_00141_01',
             'N0Sofa_00145_01', 'N3Library_00157_01', 'N3Library_00157_02', 'N3Library_03301_01',
             'N3Library_03301_02', 'N3Library_03375_01', 'N3Library_03375_02', 'N3Library_03403_01',
             'N3Library_03403_02', 'N3Office_00034_01', 'N3Office_00139_01', 'N3Office_00150_01',
             'N3Office_00153_01', 'N3Office_00159_01', 'N3Office_03301_01']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='C:/RoHM/datasets/PROX/recordings/')
    # parser.add_argument('--outdir', type=str, default='C:/RoHM/datasets/EgoBody/kinect_color/')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str,
                        default='C:/Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)

    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()
    args.pred_only = True
    args.grayscale = False

    args.outdir = 'C:/RoHM/datasets/PROX/recordings/'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    root_dir = 'C:/RoHM/datasets/PROX/'
    color_dir = pjoin(root_dir, "recordings")
    img_dir = pjoin(pjoin(color_dir, "{}"), "Color")

    list = {}
    depth_lists = {}
    cam2world = {}
    scenes = []

    # scene_lists = os.listdir(color_dir)
    scene_lists = test_list

    for scene in scene_lists:
        scene_list = []
        depth_list = []
        scenes.append(scene)
        for sample in os.listdir(img_dir.format(scene)):
            scene_list.append(pjoin(img_dir.format(scene), sample))
        depth_list = glob.glob(pjoin(pjoin(pjoin(color_dir, scene), "human_human"), "*.npy"))
        depth_list.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[-1]))
        # scene_list = scene_list[:len(depth_list)]

        with open(pjoin(pjoin(root_dir, 'cam2world'), scene.split("_")[0] + "_log.json"), 'r') as f:
            cam2world[scene] = json.load(f)["trans_final"]

        with open(pjoin(root_dir, "calibration", "Color.json")) as f:
            K = json.load(f)

        list[scene] = scene_list
        depth_lists[scene] = depth_list

    matrix = np.eye(3)
    matrix[0, 0] = K["f"][0]
    matrix[0, 2] = K["c"][0]
    matrix[1, 1] = K["f"][1]
    matrix[1, 2] = K["c"][1]
    K_tmp = copy.deepcopy(K)
    K_tmp["f"][0] = K["f"][0] * (512 / 1920)
    K_tmp["c"][0] = K["c"][0] * (512 / 1920)
    K_tmp["f"][1] = K["f"][1] * (424 / 1080)
    K_tmp["c"][1] = K["c"][1] * (424 / 1080)

    for scene in scene_lists:
        print("process: {}".format(scene))

        scene_cam2world = np.array(cam2world[scene])

        Rt = np.eye(4)
        Rt[:3, :3] = scene_cam2world[:3, :3]
        Rt[:3, 3] = scene_cam2world[:3, 3]

        scene_rgb_ori = Image.open(pjoin(args.outdir, scene, "Background_predicted_histogram", "background.jpg")).convert('RGB')
        scene_rgb_ori = np.array(scene_rgb_ori)
        scene_rgb_ori = cv2.cvtColor(scene_rgb_ori, cv2.COLOR_RGB2BGR)

        scene_rgb_ori = cv2.undistort(scene_rgb_ori.copy(), np.asarray(matrix), np.asarray(K['k']))
        scene_rgb_ori = cv2.flip(scene_rgb_ori, 1)

        scene_rgb_ori = cv2.cvtColor(scene_rgb_ori, cv2.COLOR_BGR2RGB)
        scene_rgb_ori = Image.fromarray(scene_rgb_ori)

        w, h = scene_rgb_ori.size

        scene_rgb_ori = np.array(scene_rgb_ori)[:, :, ::-1]

        # depth = depth_anything.infer_image(rgb, rgb.shape[0])
        depth_scene = depth_anything.infer_image(scene_rgb_ori, args.input_size)
        # depth_scene_resize = cv2.resize(depth_scene, (512, 424), interpolation=cv2.INTER_LINEAR)
        depth_kmeans_input = depth_scene.reshape(-1, 1)

        class_num = 10
        kmeans_model = KMeans(n_clusters=class_num, random_state=4)
        kmeans_model.fit(depth_kmeans_input)
        centroids = kmeans_model.cluster_centers_.squeeze()
        key_points = np.array(sorted(np.concatenate([[depth_kmeans_input.min()-1e-4], centroids, [depth_kmeans_input.max()+01e-4]], axis=0)))
        labels = torch.bucketize(torch.from_numpy(depth_scene), torch.from_numpy(key_points))
        weights = torch.from_numpy((depth_scene - key_points[labels - 1]) / (key_points[labels] - key_points[labels - 1]))

        # kmeans_visualize(labels)
        # scale_parameter = torch.nn.Parameter(torch.ones_like(torch.from_numpy(key_points)))
        # key_points = torch.from_numpy(key_points)
        scale_parameter = torch.nn.Parameter(torch.from_numpy(key_points))

        lr = 0.001
        optimizer = torch.optim.Adam(params=[scale_parameter], lr=lr, eps=1e-15)
        # optimizer = optim.Adam([scale_parameter], lr=0.01)

        human_mask_deeplab_list = sorted(glob.glob(pjoin(args.outdir, scene, "mask_predict_deeplab", "*.png")))[:len(list[scene])]

        frame_slice = slice(0, len(list[scene]), 5)

        for j in range(2):
            if j == 1:
                optimizer.param_groups[0]["lr"] *= 0.5
            for i, (image_name, depth_name, human_mask_deeplab_name) in tqdm(enumerate(zip(list[scene][frame_slice][:len(depth_lists[scene])], depth_lists[scene], human_mask_deeplab_list[frame_slice][:len(depth_lists[scene])]))):

                rgb_ori = Image.open(image_name).convert('RGB')
                depth_human = np.load(depth_name)
                human_mask = depth_human > 0
                human_mask_deeplab = 1 - cv2.imread(human_mask_deeplab_name).mean(-1) > 0

                current_numpy_image = np.array(rgb_ori)
                # 将 RGB 图像转换为 BGR 图像
                rgb_ori = cv2.cvtColor(current_numpy_image, cv2.COLOR_RGB2BGR)

                rgb_ori = cv2.undistort(rgb_ori.copy(), np.asarray(matrix), np.asarray(K['k']))
                rgb_ori = cv2.flip(rgb_ori, 1)
                human_mask_deeplab = cv2.undistort(human_mask_deeplab.copy() * 255.0, np.asarray(matrix), np.asarray(K['k']))
                human_mask_deeplab = cv2.flip(human_mask_deeplab, 1)
                human_mask_deeplab = human_mask_deeplab > 0

                rgb_ori = cv2.cvtColor(rgb_ori, cv2.COLOR_BGR2RGB)
                rgb_ori = Image.fromarray(rgb_ori)
                rgb_ori = np.array(rgb_ori)[:, :, ::-1]

                depth_current = depth_anything.infer_image(rgb_ori, args.input_size)
                # if i == 0:
                #     depth_tmp = depth_current

                if (human_mask * human_mask_deeplab).sum() == 0:
                    continue
                else:
                    # def closure():
                    optimizer.zero_grad()

                    depth_current_modify = (scale_parameter[labels - 1]) + weights * ((scale_parameter[labels]) - (scale_parameter[labels-1]))
                    # depth_current_modify = (scale_parameter[labels - 1] * key_points[labels-1]) + weights * ((scale_parameter[labels] * key_points[labels]) - (scale_parameter[labels-1] * key_points[labels]))

                    loss = ((torch.from_numpy(human_mask) * torch.from_numpy(human_mask_deeplab) * torch.abs(depth_current_modify - torch.from_numpy(depth_human))).sum()
                            / (human_mask * human_mask_deeplab).sum())
                    loss.backward()

                        # return loss
                    optimizer.step()
                    print(loss.item())

        depth_scene_modify = ((scale_parameter[labels - 1]) + weights * ((scale_parameter[labels]) - (scale_parameter[labels-1]))).detach().cpu().numpy()
        depth_scene_modify = cv2.resize(depth_scene_modify, (512, 424), interpolation=cv2.INTER_LINEAR)

        x, y = np.meshgrid(np.arange(512), np.arange(424))
        x = (x - K_tmp["c"][0]) / K_tmp["f"][0]
        y = (y - K_tmp["c"][1]) / K_tmp["f"][1]
        z = np.array(depth_scene_modify)

        scene_rgb_ori = cv2.resize(scene_rgb_ori, (512, 424), interpolation=cv2.INTER_LINEAR)

        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(scene_rgb_ori).reshape(-1, 3) / 255.0
        point_world = points_coord_trans(points, Rt)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_world)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(pjoin(pjoin(args.outdir, scene, "point_cloud"), "scene_init_cloud_points.ply"), pcd)