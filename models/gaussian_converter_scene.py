import os.path

import torch
import torch.nn as nn
import numpy as np
from .deformer import get_deformer
from .pose_correction import get_pose_correction
from .texture import get_texture
from submodules import smplx
from utils.general_utils import build_rotation


class GaussianConverterScene(nn.Module):
    def __init__(self, cfg, metadata, is_simple=False):
        super().__init__()
        self.cfg = cfg
        self.metadata = metadata
        self.is_simple = is_simple
        # if not is_two_person:

        # self.scene_color = torch.from_numpy(np.load(os.path.join(self.cfg.dataset.root_dir, "recordings", self.cfg.dataset.recording_name, "scene_color.npy"))).cuda()

        self.pose_correction = get_pose_correction(cfg.model.pose_correction, metadata)
        self.deformer = get_deformer(cfg.model.deformer, metadata)
        self.texture = get_texture(cfg.model.texture, metadata)
        if not is_simple:
            self.texture_scene = get_texture(cfg.model.texture, metadata, is_scene=True)
            # self.texture_scene = get_texture(cfg.model.texture, metadata, is_scene=True)

            recording_name = cfg.dataset.recording_name
            if cfg.dataset.name == "prox_add_scene":
                exp_path = "prox_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
            elif cfg.dataset.name == "egobody_add_scene":
                exp_path = "egobody_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
            elif cfg.dataset.name == "i3db_add_scene":
                exp_path = "i3db_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
            elif cfg.dataset.name == "emdb_add_scene":
                exp_path = "emdb_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
            texture_scene_pretrained_info = torch.load("./exp_background/{}/ckpt50000.pth".format(exp_path))

            # 去掉前缀
            prefix = 'texture.'
            state_dict_without_prefix = {k.replace(prefix, ''): v for k, v in texture_scene_pretrained_info[1].items()}
            # state_dict_without_prefix["latent.weight"] = state_dict_without_prefix["latent.weight"][0, :].unsqueeze(0).repeat(len(self.texture_scene.latent.weight), 1)
            state_dict_without_prefix["latent.weight"] = state_dict_without_prefix["latent.weight"][
                                                         :len(self.texture_scene.latent.weight), :]
            self.texture_scene.load_state_dict(state_dict_without_prefix)

        self.optimizer, self.scheduler = None, None
        self.set_optimizer()

        self.smplx_neutral = smplx.create(model_path="./body_models/smplx_model", model_type="smplx", gender=metadata["gender"], use_pca=False).cuda()


    def set_optimizer(self):
        opt_params = [
            {'params': self.deformer.rigid.parameters(), 'lr': self.cfg.opt.get('rigid_lr', 0.)},
            # {'params': self.deformer.non_rigid.parameters(), 'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('non_rigid_lr', 0.)},
            {'params': [p for n, p in self.deformer.non_rigid.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('nr_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            {'params': self.pose_correction.parameters(), 'lr': self.cfg.opt.get('pose_correction_lr', 0.)},
            # {'params': self.pose_correction.parameters(), 'lr': 0.01},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' not in n],
             'lr': self.cfg.opt.get('texture_lr', 0.)},
            {'params': [p for n, p in self.texture.named_parameters() if 'latent' in n],
             'lr': self.cfg.opt.get('tex_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
            # {'params': [p for n, p in self.texture_scene.named_parameters() if 'latent' not in n],
            #  'lr': self.cfg.opt.get('texture_lr', 0.)},
            # {'params': [p for n, p in self.texture_scene.named_parameters() if 'latent' in n],
            #  'lr': self.cfg.opt.get('tex_latent_lr', 0.), 'weight_decay': self.cfg.opt.get('latent_weight_decay', 0.05)},
        ]
        self.optimizer = torch.optim.Adam(params=opt_params, lr=0.001, eps=1e-15)

        gamma = self.cfg.opt.lr_ratio ** (1. / self.cfg.opt.iterations)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

    def forward(self, gaussians, gaussians_scene, camera, iteration, compute_loss=True, is_block=False, is_unblock=False, is_diredcly=False):
        loss_reg = {}
        if is_diredcly:
            frame = camera.frame_id

            idx = torch.Tensor([self.pose_correction.frame_dict[frame]]).long().cuda()
            root_orient = self.pose_correction.root_orients(idx)
            pose_body = self.pose_correction.pose_bodys(idx)
            trans = self.pose_correction.trans(idx)
            betas = self.pose_correction.betas(idx)

            if iteration < self.cfg.model.pose_correction.get('delay', 0):
                with torch.no_grad():
                    deformed = self.smplx_neutral(
                        global_orient=root_orient,
                        transl=trans,
                        body_pose=pose_body[:, :63],
                        jaw_pose=pose_body[:, 63:63+3],
                        leye_pose=pose_body[:, 63 + 3:63 + 6],
                        reye_pose=pose_body[:, 63 + 6:63 + 9],
                        left_hand_pose=pose_body[:, 63 + 9:63 + 54],
                        right_hand_pose=pose_body[:, 63 + 54:63 + 99],
                        expression=torch.zeros(1, 10).cuda(),
                        betas=betas,
                    )
            else:
                deformed = self.smplx_neutral(
                    global_orient=root_orient,
                    transl=trans,
                    body_pose=pose_body[:, :63],
                    jaw_pose=pose_body[:, 63:63 + 3],
                    leye_pose=pose_body[:, 63 + 3:63 + 6],
                    reye_pose=pose_body[:, 63 + 6:63 + 9],
                    left_hand_pose=pose_body[:, 63 + 9:63 + 54],
                    right_hand_pose=pose_body[:, 63 + 54:63 + 99],
                    expression=torch.zeros(1, 10).cuda(),
                    betas=betas,
                )

            deformed_gaussians = gaussians.clone()
            deformed_gaussians._xyz = deformed.vertices.squeeze()
            tfs = deformed.A
            T_fwd = torch.matmul(torch.from_numpy(self.deformer.rigid.skinning_weights).cuda(), tfs.view(-1, 16)).view(-1, 4, 4).float()

            deformed_gaussians.set_fwd_transform(T_fwd.detach())
            rotation_hat = build_rotation(deformed_gaussians._rotation)
            # rotation_bar = torch.matmul(T_fwd[:, :3, :3], rotation_hat)
            setattr(deformed_gaussians, 'rotation_precomp', rotation_hat)

            modified_pose = {
                "root_orients": self.pose_correction.root_orients(torch.Tensor([self.metadata['frame_dict'][camera.frame_id]]).long().cuda()),
                "pose_bodys": self.pose_correction.pose_bodys(torch.Tensor([self.metadata['frame_dict'][camera.frame_id]]).long().cuda()),
                "trans": self.pose_correction.trans(torch.Tensor([self.metadata['frame_dict'][camera.frame_id]]).long().cuda()),
            }
        else:
            # loss_reg.update(gaussians.get_opacity_loss())
            camera, loss_reg_pose, new_gaussian_xyz = self.pose_correction(camera, iteration, is_block, is_unblock)

            modified_pose = {
                "root_orients": self.pose_correction.root_orients(torch.Tensor([self.metadata['frame_dict'][camera.frame_id]]).long().cuda()),
                "pose_bodys": self.pose_correction.pose_bodys(torch.Tensor([self.metadata['frame_dict'][camera.frame_id]]).long().cuda()),
                "trans": self.pose_correction.trans(torch.Tensor([self.metadata['frame_dict'][camera.frame_id]]).long().cuda()),
            }

            if new_gaussian_xyz is not None:
                gaussians._xyz = new_gaussian_xyz.squeeze()

            # pose augmentation
            pose_noise = self.cfg.pipeline.get('pose_noise', 0.)
            if self.training and pose_noise > 0 and np.random.uniform() <= 0.5:
                camera = camera.copy()
                camera.rots = camera.rots + torch.randn(camera.rots.shape, device=camera.rots.device) * pose_noise

            deformed_gaussians, loss_reg_deformer = self.deformer(gaussians, camera, iteration, compute_loss, is_block, is_unblock)

            loss_reg.update(loss_reg_pose)
            loss_reg.update(loss_reg_deformer)

        color_precompute = self.texture(deformed_gaussians, camera)
        if not self.is_simple:

            # color_precompute_scene =
            with torch.no_grad():
                color_precompute_scene = self.texture_scene(gaussians_scene, camera, is_scene=True)
                # color_precompute_scene = self.scene_color
        else:
            color_precompute_scene = None
        return deformed_gaussians, gaussians_scene, loss_reg, color_precompute, color_precompute_scene, modified_pose

    def optimize(self):
        grad_clip = self.cfg.opt.get('grad_clip', 0.)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()