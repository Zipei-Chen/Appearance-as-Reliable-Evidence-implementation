import os
import time

os.environ["WANDB_MODE"] = "offline"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from gaussian_renderer.gaussian_render_scene import render_add_scene
from PIL import Image, ImageDraw
# import faulthandler
#
# with open("fault_log.txt", "w") as f:
#     faulthandler.enable(file=f)
#
# faulthandler.dump_traceback_later(timeout=28)
import trimesh

from scene import Scene
from scene.scene_new import SceneNew
from scene import GaussianModel
from scene import GaussianModelBackground
from utils.general_utils import fix_random, Evaluator, PSEvaluator
from tqdm import tqdm
from utils.loss_utils import full_aiap_loss

import hydra
from omegaconf import OmegaConf
import wandb
import lpips
from submodules import smplx

from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from utils.render_util import create_pyrender_skel, create_render_cam, create_pyrender_scene, render_img
import json
import pyrender
import pandas as pd
import pickle as pkl
from scipy.optimize import minimize, Bounds, minimize_scalar
import glob
from ot import sinkhorn2, sinkhorn, emd  # 计算Sinkhorn距离
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

smplx_neutral = smplx.create(model_path="./body_models/smplx_model", model_type="smplx", gender="male",
                             flat_hand_mean=True, use_pca=False).cuda()

is_coarse = False
is_directly = True
mask_loss = False
mse_mask_loss = False

down_half_body_index = [0, 1, 3, 4, 6, 7, 9, 10]
down_half_body_index_expand = [x * 3 + j for x in down_half_body_index for j in range(1, 4)]


def cagrad(grads, alpha=0.5, rescale=0):
    g1 = grads[:, 0]
    g2 = grads[:, 1]

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    # g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12)
    g0_norm = np.sqrt(g11)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = alpha * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        # return (coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-8) +
        #         0.5 * x * (g11 + g22 - 2 * g12) + (0.5 + x) * (g12 - g22) + g22)
        return (coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-8) +
                x * (g11 - g12) + g12)

    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    x = res.x

    gw_norm = np.sqrt(x ** 2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-8)
    lmbda = coef / (gw_norm + 1e-8)
    # g = (0.5 + lmbda * x) * g1 + (0.5 + lmbda * (1 - x)) * g2  # g0 + lmbda*gw
    g = (1 + lmbda * x) * g1 + (lmbda * (1 - x)) * g2  # g0 + lmbda*gw

    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def cagrad_weighted(grads, weight, alpha=0.5, rescale=0):
    g1 = grads[:, 0]
    g2 = grads[:, 1]

    weight_no = (1 - weight).astype(np.float32)

    g11 = g1.dot(g1).item()
    g12 = g1.dot(g2).item()
    g22 = g2.dot(g2).item()

    w11g11 = (g1 * weight).dot(g1 * weight)
    w22g22 = (g2 * weight_no).dot(g2 * weight_no)
    w12g12 = (g1 * weight).dot(g2 * weight_no)
    g1w1g1 = g1.dot(g1 * weight)
    g1w2g2 = g1.dot(g2 * weight_no)
    g2w1g1 = g2.dot(g1 * weight)
    g2w2g2 = g2.dot(g2 * weight_no)

    # g0_norm = 0.5 * np.sqrt(g11 + g22 + 2 * g12)
    g0_norm = np.sqrt(w11g11 + w22g22 + 2 * w12g12)

    # want to minimize g_w^Tg_0 + c*||g_0||*||g_w||
    coef = alpha * g0_norm

    def obj(x):
        # g_w^T g_0: x*0.5*(g11+g22-2g12)+(0.5+x)*(g12-g22)+g22
        # g_w^T g_w: x^2*(g11+g22-2g12)+2*x*(g12-g22)+g22
        # return (coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-8) +
        #         0.5 * x * (g11 + g22 - 2 * g12) + (0.5 + x) * (g12 - g22) + g22)
        # return (coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-8) +
        #         x * (g11 - g12) + g12)
        return (coef * np.sqrt(x ** 2 * (g11 + g22 - 2 * g12) + 2 * x * (g12 - g22) + g22 + 1e-8) +
                x * (g1w1g1 + g1w2g2 - g2w1g1 - g2w2g2) + g2w1g1 + g2w2g2)

    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    x = res.x

    gw_norm = np.sqrt(x ** 2 * g11 + (1 - x) ** 2 * g22 + 2 * x * (1 - x) * g12 + 1e-8)
    lmbda = coef / (gw_norm + 1e-8)
    # g = (0.5 + lmbda * x) * g1 + (0.5 + lmbda * (1 - x)) * g2  # g0 + lmbda*gw
    # g = (1 + lmbda * x) * g1 + (lmbda * (1 - x)) * g2  # g0 + lmbda*gw
    g = (g1 * weight) + (g2 * weight_no) + (lmbda * x) * g1 + (lmbda * (1 - x)) * g2  # g0 + lmbda*gw

    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha ** 2)
    else:
        return g / (1 + alpha)


def calculate_angle(vector_a, vector_b):
    """
    计算两个3维向量之间的夹角（以弧度为单位）。

    参数:
    vector_a (list or numpy.ndarray): 第一个3维向量。
    vector_b (list or numpy.ndarray): 第二个3维向量。

    返回:
    float: 两个向量之间的夹角（弧度）。
    """
    # 将输入向量转换为numpy数组
    # vector_a = np.array(vector_a)
    # vector_b = np.array(vector_b)

    # 计算向量的点积
    dot_product = np.dot(vector_a, vector_b)

    # 计算向量的模长
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 计算夹角的余弦值
    cos_theta = dot_product / (norm_a * norm_b)

    # 计算夹角（弧度）
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return angle


def global_depth_map(points_3d, intrinsic_matrix, rotation_matrix, translation_vector):
    w = 800  # 图像宽度
    h = 450  # 图像高度
    sample_num = points_3d.shape[0]
    points_3d = points_3d.reshape(-1, 3)

    # 将世界坐标系中的点转换到相机坐标系
    points_3d = (np.matmul(rotation_matrix, points_3d[:, :, None]) + translation_vector[:, None]).squeeze()
    image_point_homogeneous = np.matmul(intrinsic_matrix, points_3d[:, :, None]).squeeze()

    # 初始化深度图
    depth_map = np.full((h, w), np.inf, dtype=np.float32)

    # X = points_3d[:, 0]
    # Y = points_3d[:, 1]
    # Z = points_3d[:, 2]

    # 计算 u, v 坐标
    # u = (intrinsic_matrix[0, 0] * X / Z + intrinsic_matrix[0, 2]).astype(int).reshape(sample_num, 24)
    # v = (intrinsic_matrix[1, 1] * Y / Z + intrinsic_matrix[1, 2]).astype(int).reshape(sample_num, 24)
    # Z = Z.reshape(sample_num, 24)
    u = (image_point_homogeneous[:, 0] / image_point_homogeneous[:, 2]).numpy().astype(np.uint32)
    v = (image_point_homogeneous[:, 1] / image_point_homogeneous[:, 2]).numpy().astype(np.uint32)
    Z = (image_point_homogeneous[:, 2]).numpy()

    # 过滤掉不在图像范围内的点
    valid_mask = (0 <= u) & (u < w) & (0 <= v) & (v < h)
    u = u[valid_mask]
    v = v[valid_mask]
    Z = Z[valid_mask]
    # u = u[valid_mask]
    # v = v[valid_mask]
    # Z = Z[valid_mask]

    # index = range(sample_num)

    # 创建索引数组
    # indices = np.arange(sample_num)[valid_mask]
    #
    # # 更新深度图
    # depth_map[indices, v, u] = np.minimum(depth_map[indices, v, u], Z)

    # 创建一个范围数组，表示 depth_map 的第一个维度
    # i_indices = np.arange(len(depth_map))[:, None]  # 形状变为 (1124, 1)

    # 使用广播机制扩展 i_indices 以匹配 u 和 v 的形状
    # i_indices = np.repeat(i_indices, u.shape[1], axis=1)  # 形状变为 (1124, 24)

    # 使用高级索引获取 depth_map 中对应位置的值
    current_depths = depth_map[v, u]

    # 计算最小值
    new_depths = np.minimum(current_depths, Z)

    # 更新 depth_map
    depth_map[v, u] = new_depths

    # # 更新深度图
    # for i in range(len(depth_map)):
    #     depth_map[i, u[i], v[i]] = np.minimum(depth_map[i, u[i], v[i]], Z[i])

    # 处理周围邻域的点
    for du in range(-1, 1):
        for dv in range(-1, 1):
            uu = u + du
            vv = v + dv
            valid_mask = (0 <= uu) & (uu < w) & (0 <= vv) & (vv < h)
            uu = uu[valid_mask]
            vv = vv[valid_mask]
            # depth_map[i_indices, vv, uu] = np.minimum(depth_map[i_indices, vv, uu], Z[valid_mask].reshape(sample_num, 24))
            depth_map[vv, uu] = np.minimum(depth_map[vv, uu], Z[valid_mask])

    # 将深度图中的无穷大值替换为最大深度值
    max_depth = np.max(depth_map[depth_map != np.inf])
    depth_map[depth_map == np.inf] = max_depth

    return depth_map


def perspective_project_point_to_image(world_point, rotation_matrix, translation_vector, intrinsic_matrix, depth_map=None, mask_given=None):
    """
    将世界坐标系中的3维点透视投影到相机平面上的2维点。

    :param world_point: 世界坐标系中的3维点 (X, Y, Z)
    :param rotation_matrix: 旋转矩阵 R (3x3)
    :param translation_vector: 平移向量 t (3x1)
    :param intrinsic_matrix: 内参矩阵 K (3x3)
    :return: 图像平面上的2维点 (u, v)
    """
    if mask_given is None:
        sample_num = world_point.shape[0]
        world_point = world_point.reshape(-1, 3, 1)

        # 将世界坐标系中的点转换到相机坐标系
        camera_point = (np.matmul(rotation_matrix, world_point) + translation_vector[:, None]).squeeze()

        # 透视投影到图像平面
        homogeneous_camera_point = np.concatenate([camera_point[:, [0]], camera_point[:, [1]], camera_point[:, [2]]], axis=-1)
        image_point_homogeneous = np.matmul(intrinsic_matrix, homogeneous_camera_point[:, :, None]).squeeze()

        # 归一化
        u = image_point_homogeneous[:, 0] / image_point_homogeneous[:, 2]
        v = image_point_homogeneous[:, 1] / image_point_homogeneous[:, 2]

        # i_indices = np.repeat(np.arange(sample_num)[:, None], 24, axis=1)  # 形状变为 (1124, 1)

        # 创建一个布尔掩码来标记哪些索引是有效的

        W = depth_map.shape[1]
        H = depth_map.shape[0]

        valid_indices = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        # 将超出范围的 u, v 设置为合法范围内的值，以避免索引错误
        u_safe = np.clip(u, 0, W - 1).astype(np.uint32)
        v_safe = np.clip(v, 0, H - 1).astype(np.uint32)

        # 计算 block_mask
        block_mask = np.full(image_point_homogeneous.shape[0], False, dtype=bool)
        block_mask[valid_indices] = image_point_homogeneous[valid_indices, 2] < depth_map[(v_safe[valid_indices]), u_safe[valid_indices]] + 0.05
    else:
        block_mask = mask_given

    # block_mask = image_point_homogeneous[:, 2] < depth_map[v.astype(np.uint32), u.astype(np.uint32)] + 0.05

    expanded_array = np.repeat(block_mask[:, np.newaxis], 3, axis=1)

    flattened_array = expanded_array.flatten()

    # 步骤3：扩展为79个元素的数组，并用0填充多余的部分
    target_length = 79
    padded_array = np.pad(flattened_array, (0, target_length - len(flattened_array)), 'constant', constant_values=True)
    # if block_mask[0] == False:
    #     padded_array[66:66+3] = False

    return padded_array


def perspective_project(world_point, rotation_matrix, translation_vector, intrinsic_matrix):
    """
    将世界坐标系中的3维点透视投影到相机平面上的2维点。

    :param world_point: 世界坐标系中的3维点 (N x 3)
    :param rotation_matrix: 旋转矩阵 R (3x3)
    :param translation_vector: 平移向量 t (3)
    :param intrinsic_matrix: 内参矩阵 K (3x3)
    :return: 图像平面上的2维点 (N x 2)
    """
    sample_num = world_point.shape[0]
    # 确保world_point形状为(N, 3, 1)
    world_point = world_point.unsqueeze(-1)

    # 将世界坐标系中的点转换到相机坐标系
    camera_point = torch.matmul(rotation_matrix, world_point).squeeze(-1) + translation_vector

    # 透视投影到图像平面
    homogeneous_camera_point = torch.cat([camera_point[:, [0]], camera_point[:, [1]], camera_point[:, [2]]], dim=-1)
    image_point_homogeneous = torch.matmul(intrinsic_matrix, homogeneous_camera_point.unsqueeze(-1)).squeeze(-1)

    # 归一化
    u = image_point_homogeneous[:, 0] / image_point_homogeneous[:, 2]
    v = image_point_homogeneous[:, 1] / image_point_homogeneous[:, 2]

    return torch.stack([u, v], dim=-1)


def image_save(image, path, past_image=None, past_image2=None):
    tensor = image.mul(255).clamp(0, 255).byte()

    # 转换为 numpy 数组
    numpy_array = tensor.cpu().numpy()

    # 调整维度顺序以适应 PIL 的要求 (H, W, C)
    image_array = numpy_array.transpose((1, 2, 0))

    # 创建 PIL 图像对象
    image = Image.fromarray(image_array)

    if past_image is not None:
        image.paste(past_image, (0, 0), past_image)
    if past_image2 is not None:
        image.paste(past_image2, (0, 0), past_image2)

    image.save(path)


def C(iteration, value):
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = OmegaConf.to_container(value)
        if not isinstance(value, list):
            raise TypeError('Scalar specification only supports list, got', type(value))
        value_list = [0] + value
        i = 0
        current_step = iteration
        while i < len(value_list):
            if current_step >= value_list[i]:
                i += 2
            else:
                break
        value = value_list[i - 1]
    return value


def training(config):
    model = config.model
    dataset = config.dataset
    recording_name = dataset.recording_name
    opt = config.opt
    pipe = config.pipeline
    testing_iterations = config.test_iterations
    testing_interval = config.test_interval
    saving_iterations = config.save_iterations
    checkpoint_iterations = config.checkpoint_iterations
    checkpoint = config.start_checkpoint
    debug_from = config.debug_from

    diffusion_vel_loss = None
    diffusion_vel_loss_after = None

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda()  # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModel(model.gaussian)
    gaussians_scene = GaussianModelBackground(model.gaussian)
    scene = SceneNew(config, gaussians, gaussians_scene, config.exp_dir, is_coarse=is_coarse)
    scene.train()

    gaussians.training_setup(opt)
    gaussians_scene.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    data_stack = None
    ema_loss_for_log = 0.0
    # progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    progress_bar = tqdm(range(first_iter, 100000), desc="Training progress")
    first_iter += 1

    r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1.0)

    epoch_num = 0

    with open(os.path.join(dataset.root_dir, "recordings", recording_name, "gender.txt"), 'r', encoding='utf-8') as file:
        gender_gt = file.read()

    gender_gt = "neutral"
    smpl_neutral = smplx.create(model_path="./body_models/smplx_model", model_type="smplx", gender=gender_gt, use_pca=False).cuda()

    with open(os.path.join("C:/RoHM/datasets/PROX/", 'calibration', 'Color.json'), 'r') as f:
        color_cam = json.load(f)

    with open("C:/RoHM/datasets/PROX/cam2world/{}.json".format(recording_name.split("_")[0]), 'r') as f:
        cam2world = np.array(json.load(f))
    world2cam = np.linalg.inv(cam2world)

    intrinsic_mtx = np.array(color_cam["camera_mtx"])

    intrinsic_mtx[0, :] *= (800 / 1920)
    intrinsic_mtx[1, :] *= (450 / 1080)

    # visible_mask = 1 - np.load(os.path.join(dataset.root_dir, recording_name, "occlusion_mask.npy"))
    visible_mask_list = scene.train_dataset.model_files["mask_joint_vis"]

    loss_l1_mask_max = -1

    iteration = first_iter

    # scene.converter.pose_correction.root_orients.weight.requires_grad = False
    # scene.converter.pose_correction.trans.weight.requires_grad = False
    # scene.converter.pose_correction.pose_bodys.weight.requires_grad = False
    scene.converter.optimizer.param_groups[3]["lr"] = 2e-4

    # while iteration < opt.iterations + 1:
    lr_list = []
    while iteration < 100000 + 10:
        iteration += 1

        if iteration == 20000:
            for param_group in scene.converter.optimizer.param_groups:
                param_group['lr'] *= 0.5

        if iteration == 40000:
            for param_group in scene.converter.optimizer.param_groups:
                param_group['lr'] *= 0.25

        if iteration == 60000:
            for param_group in scene.converter.optimizer.param_groups:
                param_group['lr'] *= 0.25
                lr_list.append(param_group['lr'])

        if iteration == 80000:
            # for param_group in scene.converter.optimizer.param_groups:
            #     param_group['lr'] *= 0.25
            for i, param_group in enumerate(scene.converter.optimizer.param_groups):
                param_group['lr'] = max(param_group['lr'] - (lr_list[i] / 200000.0), 0.0)
        # if iteration == 50000:
        #     for param_group in scene.converter.optimizer.param_groups:
        #         param_group['lr'] *= 0.5

        # if iteration == 2000:
        #     gaussians.optimizer.param_groups[1]["lr"] *= 0.1
        #     gaussians.optimizer.param_groups[2]["lr"] *= 0.1
        #     scene.converter.optimizer.param_groups[4]["lr"] *= 0.1

        iter_start.record()

        # gaussians.update_learning_rate(iteration)
        # gaussians_scene.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            # gaussians_scene.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))

            if iteration >= model.pose_correction.delay:
                epoch_num += 1

            if epoch_num == 1 and epoch_num % 3 == 1:
                with open(os.path.join(dataset.root_dir, "recordings", recording_name, "pose_info.pkl"), 'rb') as f:
                    diffusion_result = pkl.load(f)

                diffusion_pose_body = torch.from_numpy(diffusion_result["pose_body"]).cuda()
                diffusion_transl = torch.from_numpy(diffusion_result["transl"]).cuda()
                diffusion_root_orient = torch.from_numpy(diffusion_result["root_orient"]).cuda()
                diffusion_betas = torch.from_numpy(diffusion_result["beta"]).cuda()

                diffusion_joint = smpl_neutral.forward(
                    global_orient=diffusion_root_orient,
                    transl=diffusion_transl,
                    body_pose=diffusion_pose_body,
                    jaw_pose=torch.zeros(len(diffusion_root_orient), 3).cuda(),
                    leye_pose=torch.zeros(len(diffusion_root_orient), 3).cuda(),
                    reye_pose=torch.zeros(len(diffusion_root_orient), 3).cuda(),
                    left_hand_pose=torch.zeros(len(diffusion_root_orient), 45).cuda(),
                    right_hand_pose=torch.zeros(len(diffusion_root_orient), 45).cuda(),
                    expression=torch.zeros(len(diffusion_root_orient), 10).cuda(),
                    betas=diffusion_betas,
                ).joints[:, :22, :]

                mask_block = diffusion_result["mask_joint_vis"]
                # new_col = mask_block[:, [20, 21]]
                # mask_block = np.hstack((mask_block, new_col))

        data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
        data = scene.train_dataset[data_idx]
        metadata = scene.train_dataset.metadata

        pre_data, after_data = None, None

        if data_idx > 0:
            pre_data_idx = data_idx - 1
            pre_data = scene.train_dataset[pre_data_idx]

        if data_idx < len(data_stack) - 1:
            after_data_idx = data_idx + 1
            after_data = scene.train_dataset[after_data_idx]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # lambda_mask = C(iteration, config.opt.lambda_mask)
        # use_mask = lambda_mask > 0.
        use_mask = 1
        render_pkg = render_add_scene(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask,
                            recording_name=recording_name, dataset_name=dataset.name, is_directly=is_directly)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg["opacity_render"] if use_mask else None
        scene_image = render_pkg["scene_rendered_image"]
        human_opacity_image = render_pkg["human_opacity_rendered_image"]
        visible_human_3d_point = render_pkg["visible_human_3d_point"]
        visible_human_2d_point = perspective_project(visible_human_3d_point, torch.from_numpy(data.data['R'].astype(np.float32)).cuda(), torch.from_numpy(data.data["T"].astype(np.float32)).cuda(), torch.from_numpy(data.data["K"].astype(np.float32)).cuda())
        # visible_num = render_pkg["visible_num"]

        # visible_mask_gs = (visible_num > 2)
        # visible_gs_points = (render_pkg["deformed_gaussian"].get_xyz)[visible_mask_gs[:10475]]
        # visible_gs_points_2d = perspective_project(visible_gs_points, torch.from_numpy(data.data['R'].astype(np.float32)).cuda(), torch.from_numpy(data.data["T"].astype(np.float32)).cuda(), torch.from_numpy(data.data["K"].astype(np.float32)).cuda())

        # if iteration == 1 and method_num == 5:
        #     scene_depth_map = global_depth_map(render_pkg["scene_gaussian"].get_xyz.clone().detach().cpu(), intrinsic_mtx, world2cam[:3, :3], world2cam[:3, 3])

        # Loss
        gt_image = data.original_image.cuda()
        human_mask = data.original_mask.cuda()
        # gt_image_mask = gt_image * human_mask + (1- human_mask > 0) * -1 * torch.ones_like(human_mask)

        if iteration >= 5000:
            lambda_skatting = 5
        else:
            lambda_skatting = 0.1

        loss_skatting = torch.tensor(0.).cuda()
        lambda_skatting = 0

        if pre_data is not None and lambda_skatting > 0:
            pre_render_pkg = render_add_scene(pre_data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask, recording_name=recording_name)
            render_diff = image - pre_render_pkg["render"]
            original_diff = gt_image - pre_data.original_image.cuda()
            loss_skatting += l1_loss(render_diff, original_diff)

        if after_data is not None and lambda_skatting > 0:
            after_render_pkg = render_add_scene(after_data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask, recording_name=recording_name)
            render_diff = after_render_pkg["render"] - image
            original_diff = after_data.original_image.cuda() - gt_image
            loss_skatting += l1_loss(render_diff, original_diff)

        if iteration >= 500:
            lambda_l1 = C(iteration, config.opt.lambda_l1)
        else:
            lambda_l1 = 0
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        loss_l1_mask = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image)
            if mask_loss and iteration >= model.pose_correction.delay:
                if mse_mask_loss:

                    # human_mask_cal = ((abs(gt_image - scene_image) * 255).mean(0) > 50) * 1
                    # loss_l1_mask = l1_loss(torch.sigmoid(((abs(image - scene_image) * 255).mean(0) - 10)), (human_mask.squeeze() > 1) * 1.0) * lambda_l1

                    human_mask_cal = human_opacity_image * ((abs(image - scene_image) * 255).mean(0) > 10)[:1]
                    loss_l1_mask = l1_loss(human_mask_cal, (human_mask.squeeze() > 1) * 1.0) * lambda_l1
                    # cv2.imwrite("c:/humor/test4.png", torch.sigmoid(((abs(image - scene_image) * 255).mean(0) - 10)).detach().cpu().numpy() * 255)

                    # if data_idx == 75:
                    #     loss_l1_mask.backward()
                    #     scene.optimize(iteration)
                    #     continue
                    #
                    # if loss_l1_mask.item() > loss_l1_mask_max:
                    #     loss_l1_mask_max = loss_l1_mask.item()
                    #     max_frame_id = data_idx
                else:
                    # # mask_pred = (((abs(image - scene_image) * 255).mean(0) > 10) * 1)[::5, ::5].reshape(-1)
                    # mask_pred = (human_opacity_image * ((abs(image - scene_image) * 255).mean(0) > 10))[:1].squeeze()[::5, ::5].reshape(-1)
                    # mask_pred = (mask_pred / mask_pred.sum()).unsqueeze(1)
                    #
                    # human_mask = ((human_mask.squeeze() > 1) * 1.0)[::5, ::5].reshape(-1)
                    # human_mask = (human_mask / human_mask.sum()).unsqueeze(0)

                    sample_indices = np.random.choice(visible_human_2d_point.shape[0], size=min(1000, visible_human_2d_point.shape[0]), replace=False)
                    sample_indices2 = np.random.choice(torch.nonzero(human_mask.squeeze().transpose(1, 0) > 1).shape[0], size=min(5000, torch.nonzero(human_mask.squeeze().transpose(1, 0) > 1).shape[0]), replace=False)

                    mask_pred = visible_human_2d_point[sample_indices].unsqueeze(1)
                    human_mask = (torch.nonzero(human_mask.squeeze().transpose(1, 0) > 1)[sample_indices2]).unsqueeze(0)

                    diff = mask_pred - human_mask
                    cost_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-16)
                    cost_matrix = (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())
                    with torch.no_grad():

                        a, b = torch.ones(mask_pred.shape[0]) / mask_pred.shape[0], torch.ones(human_mask.shape[1]) / human_mask.shape[1]
                        gamma = sinkhorn(a.cuda(), b.cuda(), cost_matrix, reg=0.1, numItermax=100)
                    loss_l1_mask = torch.sum(cost_matrix * gamma) * 0.2

                    del cost_matrix
                    del gamma
                    del diff
                    del a, b
                    torch.cuda.empty_cache()

                    # loss_l1_mask = l1_loss(((abs(image - scene_image) * 255).mean(0) > 1) * 1, human_mask.squeeze()) * 0.1
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        # loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim + lambda_skatting * loss_skatting
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim
        loss += loss_l1_mask

        # perceptual loss
        lambda_perceptual = 0
        # lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
        # if iteration >= 500:
        #     lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
        # else:
        #     lambda_perceptual = 0
        if lambda_perceptual > 0:
            # crop the foreground
            # mask = data.original_mask.cpu().numpy()
            # mask = np.where(mask)
            # y1, y2 = mask[1].min(), mask[1].max() + 1
            # x1, x2 = mask[2].min(), mask[2].max() + 1
            # fg_image = image[:, y1:y2, x1:x2]
            # gt_fg_image = gt_image[:, y1:y2, x1:x2]
            fg_image = image
            gt_fg_image = gt_image

            loss_perceptual = loss_fn_vgg(fg_image, gt_fg_image, normalize=True).mean()
            loss += lambda_perceptual * loss_perceptual
        else:
            loss_perceptual = torch.tensor(0.)

        # skinning loss
        lambda_skinning = C(iteration, config.opt.lambda_skinning)
        lambda_skinning = 0
        # skip skinning loss
        # lambda_skinning = 0
        if lambda_skinning > 0:
            loss_skinning = scene.get_skinning_loss()
            loss += lambda_skinning * loss_skinning
        else:
            loss_skinning = torch.tensor(0.).cuda()

        lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
            loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
        else:
            loss_aiap_xyz = torch.tensor(0.).cuda()
            loss_aiap_cov = torch.tensor(0.).cuda()
        loss += lambda_aiap_xyz * loss_aiap_xyz
        loss += lambda_aiap_cov * loss_aiap_cov

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss += lbd * value

        loss.backward(retain_graph=True)


        if epoch_num >= 1 and data_idx < len(diffusion_pose_body):

            diffusion_joint_loss_lambda = 5e-1
            diffusion_vel_loss_lambda = 10

            gs_joint = smpl_neutral.forward(
                global_orient=scene.converter.pose_correction.root_orients.weight[data_idx].unsqueeze(0),
                transl=scene.converter.pose_correction.trans.weight[data_idx].unsqueeze(0),
                body_pose=scene.converter.pose_correction.pose_bodys.weight[data_idx][:63].unsqueeze(0),
                jaw_pose=scene.converter.pose_correction.pose_bodys.weight[data_idx][63:63 + 3].unsqueeze(0),
                leye_pose=scene.converter.pose_correction.pose_bodys.weight[data_idx][63 + 3:63 + 6].unsqueeze(0),
                reye_pose=scene.converter.pose_correction.pose_bodys.weight[data_idx][63 + 6:63 + 9].unsqueeze(0),
                left_hand_pose=scene.converter.pose_correction.pose_bodys.weight[data_idx][63 + 9:63 + 54].unsqueeze(0),
                right_hand_pose=scene.converter.pose_correction.pose_bodys.weight[data_idx][63 + 54:63 + 99].unsqueeze(0),
                expression=torch.zeros(1, 10).cuda(),
                betas=scene.converter.pose_correction.betas.weight[data_idx].unsqueeze(0),
            ).joints[0, :22, :]

            # visible_mask = perspective_project_point_to_image(world_point=gs_joint.clone().detach().cpu(), rotation_matrix=world2cam[:3, :3], translation_vector=world2cam[:3, 3], intrinsic_matrix=intrinsic_mtx, depth_map=scene_depth_map, mask_given=visible_mask_list[data_idx].astype(np.bool8))
            visible_mask = perspective_project_point_to_image(world_point=gs_joint.clone().detach().cpu(), rotation_matrix=world2cam[:3, :3], translation_vector=world2cam[:3, 3], intrinsic_matrix=intrinsic_mtx, depth_map=None, mask_given=visible_mask_list[data_idx].astype(np.bool8))

            gaussian_root_orient_grad = scene.converter.pose_correction.root_orients.weight.grad.clone().detach().cpu().numpy()[data_idx].reshape(-1, 3)
            gaussian_pose_body_grad = scene.converter.pose_correction.pose_bodys.weight.grad.clone().detach().cpu().numpy()[data_idx][:63].reshape(-1, 63)
            gaussian_transl_grad = scene.converter.pose_correction.trans.weight.grad.clone().detach().cpu().numpy()[data_idx].reshape(-1, 3)
            gaussian_betas_grad = scene.converter.pose_correction.betas.weight.grad.clone().detach().cpu().numpy()[data_idx].reshape(-1, 10)

            gaussian_grad = np.concatenate([gaussian_root_orient_grad, gaussian_pose_body_grad, gaussian_transl_grad, gaussian_betas_grad], axis=1)
            # gaussian_grad = np.concatenate([gaussian_pose_body_grad, gaussian_betas_grad], axis=1)

            scene.converter.pose_correction.root_orients.weight.grad[data_idx] = 0
            scene.converter.pose_correction.pose_bodys.weight.grad[data_idx] = 0
            scene.converter.pose_correction.trans.weight.grad[data_idx] = 0
            scene.converter.pose_correction.betas.weight.grad[data_idx] = 0

            # diffusion_joint_loss = torch.nn.MSELoss()(diffusion_joint[data_idx], gs_joint) * diffusion_joint_loss_lambda
            diffusion_joint_loss = torch.sum((diffusion_joint[data_idx] - gs_joint) ** 2 * torch.from_numpy(1 - visible_mask_list[data_idx]).unsqueeze(1).cuda()) * diffusion_joint_loss_lambda

            if data_idx > 0:
                pre_data_idx_t = data_idx - 1
                pre_gs_joint = smpl_neutral.forward(
                    global_orient=scene.converter.pose_correction.root_orients.weight[pre_data_idx_t].unsqueeze(0),
                    transl=scene.converter.pose_correction.trans.weight[pre_data_idx_t].unsqueeze(0),
                    body_pose=scene.converter.pose_correction.pose_bodys.weight[pre_data_idx_t][:63].unsqueeze(0),
                    jaw_pose=scene.converter.pose_correction.pose_bodys.weight[pre_data_idx_t][63:63+3].unsqueeze(0),
                    leye_pose=scene.converter.pose_correction.pose_bodys.weight[pre_data_idx_t][63+3:63+6].unsqueeze(0),
                    reye_pose=scene.converter.pose_correction.pose_bodys.weight[pre_data_idx_t][63+6:63+9].unsqueeze(0),
                    left_hand_pose=scene.converter.pose_correction.pose_bodys.weight[pre_data_idx_t][63+9:63+54].unsqueeze(0),
                    right_hand_pose=scene.converter.pose_correction.pose_bodys.weight[pre_data_idx_t][63+54:63+99].unsqueeze(0),
                    expression=torch.zeros(1, 10).cuda(),
                    betas=scene.converter.pose_correction.betas.weight[pre_data_idx_t].unsqueeze(0),
                ).joints[0, :22, :]
                joint_diff = gs_joint - pre_gs_joint

                diffusion_joint_diff = diffusion_joint[data_idx] - diffusion_joint[pre_data_idx_t]

                # diffusion_vel_loss = torch.nn.L1Loss()(joint_diff, diffusion_joint_diff) * diffusion_vel_loss_lambda
                diffusion_vel_loss = torch.nn.MSELoss()(torch.norm(joint_diff, p=2, dim=1), torch.norm(diffusion_joint_diff, p=2, dim=1)) * diffusion_vel_loss_lambda
                # diffusion_vel_loss = torch.mean((torch.norm(joint_diff, p=2, dim=1) - torch.norm(diffusion_joint_diff, p=2, dim=1)) ** 2 * torch.from_numpy(visible_mask_list[data_idx] * 0.5 + (1 - visible_mask_list[data_idx]) * 1).unsqueeze(1).cuda()) * diffusion_joint_loss_lambda

                diffusion_joint_loss += diffusion_vel_loss

            if data_idx < len(diffusion_pose_body) - 1:
                after_data_idx_t = data_idx + 1
                after_gs_joint = smpl_neutral.forward(
                    global_orient=scene.converter.pose_correction.root_orients.weight[after_data_idx_t].unsqueeze(0),
                    transl=scene.converter.pose_correction.trans.weight[after_data_idx_t].unsqueeze(0),
                    body_pose=scene.converter.pose_correction.pose_bodys.weight[after_data_idx_t][:63].unsqueeze(0),
                    jaw_pose=scene.converter.pose_correction.pose_bodys.weight[after_data_idx_t][63:63 + 3].unsqueeze(0),
                    leye_pose=scene.converter.pose_correction.pose_bodys.weight[after_data_idx_t][63 + 3:63 + 6].unsqueeze(0),
                    reye_pose=scene.converter.pose_correction.pose_bodys.weight[after_data_idx_t][63 + 6:63 + 9].unsqueeze(0),
                    left_hand_pose=scene.converter.pose_correction.pose_bodys.weight[after_data_idx_t][63 + 9:63 + 54].unsqueeze(0),
                    right_hand_pose=scene.converter.pose_correction.pose_bodys.weight[after_data_idx_t][63 + 54:63 + 99].unsqueeze(0),
                    expression=torch.zeros(1, 10).cuda(),
                    betas=scene.converter.pose_correction.betas.weight[after_data_idx_t].unsqueeze(0),
                ).joints[0, :22, :]
                joint_diff = after_gs_joint - gs_joint

                diffusion_joint_diff = diffusion_joint[after_data_idx_t] - diffusion_joint[data_idx]

                # diffusion_vel_loss_after = torch.nn.L1Loss()(joint_diff, diffusion_joint_diff) * diffusion_vel_loss_lambda
                diffusion_vel_loss_after = torch.nn.MSELoss()(torch.norm(joint_diff, p=2, dim=1), torch.norm(diffusion_joint_diff, p=2, dim=1)) * diffusion_vel_loss_lambda

                diffusion_joint_loss += diffusion_vel_loss_after

            diffusion_joint_loss.backward(retain_graph=True)

            diffusion_root_orient_grad = scene.converter.pose_correction.root_orients.weight.grad.clone().detach().cpu().numpy()[data_idx].reshape(-1, 3)
            diffusion_pose_body_grad = scene.converter.pose_correction.pose_bodys.weight.grad.clone().detach().cpu().numpy()[data_idx][:63].reshape(-1, 63)
            diffusion_transl_grad = scene.converter.pose_correction.trans.weight.grad.clone().detach().cpu().numpy()[data_idx].reshape(-1, 3)
            diffusion_betas_grad = scene.converter.pose_correction.betas.weight.grad.clone().detach().cpu().numpy()[data_idx].reshape(-1, 10)

            diffusion_grad = np.concatenate([diffusion_root_orient_grad, diffusion_pose_body_grad, diffusion_transl_grad, diffusion_betas_grad], axis=1)

            fussion_grad = cagrad_weighted(torch.from_numpy(np.concatenate([gaussian_grad, diffusion_grad], axis=0)).transpose(1, 0), visible_mask, alpha=1, rescale=1)

            scene.converter.pose_correction.root_orients.weight.grad[data_idx] = fussion_grad[:3]
            scene.converter.pose_correction.pose_bodys.weight.grad[data_idx][:63] = fussion_grad[3:3+63]
            scene.converter.pose_correction.trans.weight.grad[data_idx] = fussion_grad[66:66+3]
            scene.converter.pose_correction.betas.weight.grad[data_idx] = fussion_grad[69:69+10]

            if data_idx > 0:
                joint_diff = gs_joint - pre_gs_joint

                smooth_loss = torch.nn.MSELoss()(torch.norm(joint_diff, p=2, dim=1), torch.zeros(22).cuda()) * 10

                smooth_loss.backward()

        # loss.backward()
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                # 'loss/mask_loss': loss_mask.item(),
                'loss/loss_skinning': loss_skinning.item(),
                'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                'loss/cov_aiap_loss': loss_aiap_cov.item(),
                'loss/total_loss': loss.item(),
                'iter_time': elapsed,
            }
            log_loss.update({
                'loss/loss_' + k: v for k, v in loss_reg.items()
            })
            wandb.log(log_loss)

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}"
                    , "loss/l1_loss": f"{loss_l1.item() * lambda_l1:.{7}f}"
                    , "loss/loss_l1_mask": f"{loss_l1_mask.item():.{7}f}"
                    # , "loss/mask_loss": f"{loss_mask.item() * lambda_mask:.{7}f}"
                    , "loss/loss_skinning": f"{loss_skinning.item() * lambda_skinning:.{7}f}"
                    , "loss/xyz_aiap_loss": f"{loss_aiap_xyz.item() * lambda_aiap_xyz:.{7}f}"
                    , "loss/cov_aiap_loss": f"{loss_aiap_cov.item() * lambda_aiap_cov:.{7}f}"
                    , "loss/skatting": f"{loss_skatting.item() * lambda_skatting:.{7}f}"
                    , "loss/diffusion_vel_loss": f"{diffusion_vel_loss.item() if diffusion_vel_loss is not None else 0:.{7}f}"
                    , "loss/diffusion_vel_loss_after": f"{diffusion_vel_loss_after.item() if diffusion_vel_loss_after is not None else 0:.{7}f}"

                }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

        # Optimizer step
        if iteration < opt.iterations:
            # if iteration >= model.pose_correction.delay:
            #     scene.converter.pose_correction.pose_bodys.weight.grad[:, down_half_body_index_expand] = 0
            #     scene.converter.pose_correction.root_orients.weight.grad.zero_()
            #     scene.converter.pose_correction.trans.weight.grad.zero_()
            scene.optimize(iteration)

        if iteration in checkpoint_iterations:
            scene.save_checkpoint(iteration)


@hydra.main(version_base=None, config_path="configs", config_name="config_total_prox")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config

    config.exp_dir = os.path.join(config.get('exp_dir') or os.path.join('./exp_total', config.name))
    os.makedirs(config.exp_dir, exist_ok=True)
    config.checkpoint_iterations.append(config.opt.iterations)

    # set wandb logger
    wandb_name = config.name
    wandb.init(
        mode="disabled" if config.wandb_disable else None,
        name=wandb_name,
        project='gaussian-splatting-avatar',
        # entity='fast-avatar',
        dir=config.exp_dir,
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method='spawn'),
    )

    print("Optimizing " + config.exp_dir)

    # Initialize system state (RNG)
    fix_random(config.seed)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    training(config)

    # All done
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
