import os

os.environ["WANDB_MODE"] = "offline"
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_background
from PIL import Image
# import faulthandler
#
# with open("fault_log.txt", "w") as f:
#     faulthandler.enable(file=f)
#
# faulthandler.dump_traceback_later(timeout=28)
import trimesh

from scene.scene_background import Scene_background
from scene.gaussian_model_background import GaussianModelBackground
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

    # define lpips
    lpips_type = config.opt.get('lpips_type', 'vgg')
    loss_fn_vgg = lpips.LPIPS(net=lpips_type).cuda()  # for training
    evaluator = PSEvaluator() if dataset.name == 'people_snapshot' else Evaluator()

    first_iter = 0
    gaussians = GaussianModelBackground(model.gaussian)
    scene = Scene_background(config, gaussians, config.exp_dir)
    scene.train()

    gaussians.training_setup(opt)
    if checkpoint:
        scene.load_checkpoint(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    data_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080, point_size=1.0)
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random data point
        if not data_stack:
            data_stack = list(range(len(scene.train_dataset)))
        data_idx = data_stack.pop(randint(0, len(data_stack) - 1))
        data = scene.train_dataset[data_idx]
        metadata = scene.train_dataset.metadata

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # lambda_mask = C(iteration, config.opt.lambda_mask)
        # use_mask = lambda_mask > 0.
        use_mask = 1
        render_pkg = render_background(data, iteration, scene, pipe, background, compute_loss=True, return_opacity=use_mask,
                            recording_name=recording_name)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]
        opacity = render_pkg["opacity_render"] if use_mask else None

        opacity_copy = opacity.clone().detach().cpu()
        # Loss
        gt_image_ori = data.original_image.cuda()
        gt_image = data.original_image.detach().cpu().numpy()
        gt_image[np.tile(~(data.original_mask.cpu().numpy() != 0), (3, 1, 1))] = 0.0
        gt_image = torch.from_numpy(gt_image).cuda()
        # image[~mask] = 255. if self.white_bg else 0.

        # if iteration >= 500:
        lambda_l1 = C(iteration, config.opt.lambda_l1)
        # else:
        #     lambda_l1 = 0
        lambda_dssim = C(iteration, config.opt.lambda_dssim)
        loss_l1 = torch.tensor(0.).cuda()
        loss_dssim = torch.tensor(0.).cuda()
        if lambda_l1 > 0.:
            loss_l1 = l1_loss(image, gt_image_ori,
                              mask=torch.from_numpy(np.tile((data.original_mask.cpu().numpy() == 0), (3, 1, 1))).cuda())
        if lambda_dssim > 0.:
            loss_dssim = 1.0 - ssim(image, gt_image)
        loss = lambda_l1 * loss_l1 + lambda_dssim * loss_dssim

        # perceptual loss
        lambda_perceptual = C(iteration, config.opt.get('lambda_perceptual', 0.))
        # lambda_perceptual = 0
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

        # mask loss
        # skip_mask_loss
        lambda_mask = 0
        # if iteration <= 500:
        #     lambda_mask = 0
        # else:
        #     lambda_mask = C(iteration, config.opt.lambda_mask)
        gt_mask = data.original_mask.cuda()
        if not use_mask:
            loss_mask = torch.tensor(0.).cuda()
        elif config.opt.mask_loss_type == 'bce':
            opacity = torch.clamp(opacity, 1.e-3, 1. - 1.e-3)
            loss_mask = F.binary_cross_entropy(opacity, gt_mask)
        elif config.opt.mask_loss_type == 'l1':
            loss_mask = F.l1_loss(opacity, gt_mask)
        else:
            raise ValueError
        loss += lambda_mask * loss_mask

        # # skinning loss
        # lambda_skinning = C(iteration, config.opt.lambda_skinning)
        # # skip skinning loss
        # # lambda_skinning = 0
        # if lambda_skinning > 0:
        #     loss_skinning = scene.get_skinning_loss()
        #     loss += lambda_skinning * loss_skinning
        # else:
        #     loss_skinning = torch.tensor(0.).cuda()

        # lambda_aiap_xyz = C(iteration, config.opt.get('lambda_aiap_xyz', 0.))
        # lambda_aiap_cov = C(iteration, config.opt.get('lambda_aiap_cov', 0.))
        # if lambda_aiap_xyz > 0. or lambda_aiap_cov > 0.:
        #     loss_aiap_xyz, loss_aiap_cov = full_aiap_loss(scene.gaussians, render_pkg["deformed_gaussian"])
        # else:
        #     loss_aiap_xyz = torch.tensor(0.).cuda()
        #     loss_aiap_cov = torch.tensor(0.).cuda()
        # loss += lambda_aiap_xyz * loss_aiap_xyz
        # loss += lambda_aiap_cov * loss_aiap_cov

        # regularization
        loss_reg = render_pkg["loss_reg"]
        for name, value in loss_reg.items():
            lbd = opt.get(f"lambda_{name}", 0.)
            lbd = C(iteration, lbd)
            loss += lbd * value
        loss.backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            elapsed = iter_start.elapsed_time(iter_end)
            log_loss = {
                'loss/l1_loss': loss_l1.item(),
                'loss/ssim_loss': loss_dssim.item(),
                'loss/perceptual_loss': loss_perceptual.item(),
                'loss/mask_loss': loss_mask.item(),
                # 'loss/loss_skinning': loss_skinning.item(),
                # 'loss/xyz_aiap_loss': loss_aiap_xyz.item(),
                # 'loss/cov_aiap_loss': loss_aiap_cov.item(),
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
                    , "loss/mask_loss": f"{loss_mask.item() * lambda_mask:.{7}f}"
                    # , "loss/loss_skinning": f"{loss_skinning.item() * lambda_skinning:.{7}f}"
                    # , "loss/xyz_aiap_loss": f"{loss_aiap_xyz.item() * lambda_aiap_xyz:.{7}f}"
                    # , "loss/cov_aiap_loss": f"{loss_aiap_cov.item() * lambda_aiap_cov:.{7}f}"
                })
                # progress_bar.set_postfix({"loss/l1_loss": f"{loss_l1.item():.{7}f}"})
                # progress_bar.set_postfix({"loss/mask_loss": f"{loss_mask.item():.{7}f}"})
                # progress_bar.set_postfix({"loss/loss_skinning": f"{loss_skinning.item():.{7}f}"})
                # progress_bar.set_postfix({"loss/xyz_aiap_loss": f"{loss_aiap_xyz.item():.{7}f}"})
                # progress_bar.set_postfix({"loss/cov_aiap_loss": f"{loss_aiap_cov.item():.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter and iteration > model.gaussian.delay:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt, scene, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                scene.optimize(iteration)

            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)

@hydra.main(version_base=None, config_path="configs", config_name="config_background_prox")
def main(config):
    print(OmegaConf.to_yaml(config))
    OmegaConf.set_struct(config, False)  # allow adding new values to config

    config.exp_dir = config.get('exp_dir') or os.path.join('./exp_background', config.name)
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
