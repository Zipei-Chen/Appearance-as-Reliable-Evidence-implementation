

import os
import torch
from models import GaussianConverter, GaussianConverterScene
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_background import GaussianModelBackground
from dataset import load_dataset
import trimesh
import numpy as np
from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB


class SceneNew:

    gaussians : GaussianModel

    def __init__(self, cfg, gaussians : GaussianModel, gaussians_scne : GaussianModelBackground, save_dir : str, gaussians_unblock=None, is_coarse=False):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.cfg = cfg

        self.save_dir = save_dir
        self.gaussians = gaussians
        self.gaussians_scene = gaussians_scne
        self.gaussians_unblock = None

        if gaussians_unblock is not None:
            self.gaussians_unblock = gaussians_unblock

        self.train_dataset = load_dataset(cfg.dataset, split='train', is_coarse=is_coarse)
        if gaussians_unblock is not None:
            self.metadata_block = self.train_dataset.metadata_block
            self.metadata_unblock = self.train_dataset.metadata_unblock
            self.cameras_extent = self.metadata_block['cameras_extent']
        else:
            self.metadata = self.train_dataset.metadata
            self.cameras_extent = self.metadata['cameras_extent']
        # if cfg.mode == 'train':
        #     self.test_dataset = load_dataset(cfg.dataset, split='val')
        # elif cfg.mode == 'test':
        #     self.test_dataset = load_dataset(cfg.dataset, split='test')
        # elif cfg.mode == 'predict':
        #     self.test_dataset = load_dataset(cfg.dataset, split='predict')
        # else:
        #     raise ValueError
        self.test_dataset = self.train_dataset

        # verts = self.metadata['minimal_shape']
        # faces = self.train_dataset.faces
        # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # n_points = 50_000
        #
        # xyz = mesh.sample(n_points)
        # rgb = np.ones_like(xyz) * 255
        # storePly("C:/3dgs-avatar-release/data/peoplesnapshot_arah-format/people_snapshot_public/female-4-casual/b_pose", xyz, rgb)
        #
        # verts = self.metadata['smpl_verts']
        # faces = self.train_dataset.faces
        # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # n_points = 50_000
        #
        # xyz = mesh.sample(n_points)
        # rgb = np.ones_like(xyz) * 255
        # storePly("C:/3dgs-avatar-release/data/peoplesnapshot_arah-format/people_snapshot_public/female-4-casual/star_pose", xyz, rgb)

        # self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(), spatial_lr_scale=self.cameras_extent, verts=self.metadata['smpl_verts'])

        self.gaussians_scene.create_from_pcd(self.test_dataset.readPointCloud(is_scene=True), spatial_lr_scale=self.cameras_extent, froze=True)

        if self.gaussians_unblock is not None:
            self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(is_block=True), spatial_lr_scale=self.cameras_extent)
            self.gaussians_unblock.create_from_pcd(self.test_dataset.readPointCloud(is_block=False), spatial_lr_scale=self.cameras_extent)
        else:
            self.gaussians.create_from_pcd(self.test_dataset.readPointCloud(),spatial_lr_scale=self.cameras_extent)

        # todo

        recording_name = cfg.dataset.recording_name
        if cfg.dataset.name == "prox_add_scene":
            exp_path = "prox_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
        elif cfg.dataset.name == "egobody_add_scene":
            exp_path = "egobody_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
        elif cfg.dataset.name == "i3db_add_scene":
            exp_path = "i3db_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
        elif cfg.dataset.name == "emdb_add_scene":
            exp_path = "emdb_{}-none-identity-identity-shallow_mlp-default".format(recording_name)
        gaussians_scene_pretrained_info = torch.load("./exp_background/{}/ckpt50000.pth".format(exp_path))
        self.gaussians_scene.restore(gaussians_scene_pretrained_info[0], cfg.opt)
        # self.gaussians_scene.load()

        if gaussians_unblock is None:
            self.converter = GaussianConverterScene(cfg, self.metadata).cuda()
        else:
            self.converter_block = GaussianConverterScene(cfg, self.metadata_block).cuda()
            self.converter_unblock = GaussianConverterScene(cfg, self.metadata_unblock, is_simple=True).cuda()

    def train(self):
        if self.gaussians_unblock is not None:
            self.converter_block.train()
            self.converter_unblock.train()
        else:
            self.converter.train()

    def eval(self):
        if self.gaussians_unblock is not None:
            self.converter_block.eval()
            self.converter_unblock.eval()
        else:
            self.converter.eval()

    def optimize(self, iteration):
        gaussians_delay = self.cfg.model.gaussian.get('delay', 0)
        if iteration >= gaussians_delay:
            self.gaussians.optimizer.step()
            if self.gaussians_unblock is not None:
                self.gaussians_unblock.optimizer.step()
            # self.gaussians_scene.optimizer.step()
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.gaussians_unblock is not None:
            self.gaussians_unblock.optimizer.zero_grad(set_to_none=True)
        if self.gaussians_unblock is not None:
            self.converter_block.optimize()
            self.converter_unblock.optimize()
        else:
            self.converter.optimize()

    def convert_gaussians(self, viewpoint_camera, iteration, compute_loss=True, is_diredcly=False):
        if self.gaussians_unblock is not None:
            pc, pc_scene, loss_reg, colors_precomp, colors_precomp_scene, modified_pose = self.converter_block(self.gaussians, self.gaussians_scene, viewpoint_camera, iteration, compute_loss, is_block=True, is_diredcly=is_diredcly)
            pc_unblock, _, loss_reg_unblock, colors_precomp_unblock, _, modified_pose_unblock = self.converter_unblock(self.gaussians_unblock, self.gaussians_scene, viewpoint_camera, iteration, compute_loss, is_unblock=True, is_diredcly=is_diredcly)
            return pc, pc_scene, loss_reg, colors_precomp, colors_precomp_scene, modified_pose, pc_unblock, loss_reg_unblock, colors_precomp_unblock, modified_pose_unblock
        else:
            return self.converter(self.gaussians, self.gaussians_scene, viewpoint_camera, iteration, compute_loss, is_diredcly=is_diredcly)

    def get_skinning_loss(self):
        if self.gaussians_unblock is not None:
            loss_reg_block = self.converter_block.deformer.rigid.regularization()
            loss_reg_unblock = self.converter_unblock.deformer.rigid.regularization()
            loss_skinning_block = loss_reg_block.get('loss_skinning', torch.tensor(0.).cuda())
            loss_skinning_unblock = loss_reg_unblock.get('loss_skinning', torch.tensor(0.).cuda())
            return loss_skinning_block + loss_skinning_unblock
        else:
            loss_reg = self.converter.deformer.rigid.regularization()
            loss_skinning = loss_reg.get('loss_skinning', torch.tensor(0.).cuda())
            return loss_skinning

    def save(self, iteration):
        point_cloud_path = os.path.join(self.save_dir, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_checkpoint(self, iteration):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        if self.gaussians_unblock is not None:
            torch.save((self.gaussians.capture(),
                        self.gaussians_unblock.capture(),
                        self.gaussians_scene.capture(),
                        self.converter_block.state_dict(),
                        self.converter_unblock.state_dict(),
                        self.converter_block.optimizer.state_dict(),
                        self.converter_block.scheduler.state_dict(),
                        self.converter_unblock.optimizer.state_dict(),
                        self.converter_unblock.scheduler.state_dict(),
                        iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")
        else:
            torch.save((self.gaussians.capture(),
                        self.gaussians_scene.capture(),
                        self.converter.state_dict(),
                        self.converter.optimizer.state_dict(),
                        self.converter.scheduler.state_dict(),
                        iteration), self.save_dir + "/ckpt" + str(iteration) + ".pth")

    def save_checkpoint_epoch(self, epoch):
        print("\n[epoch {}] Saving Checkpoint".format(epoch))
        if self.gaussians_unblock is not None:
            torch.save((self.gaussians.capture(),
                        self.converter_block.state_dict(),
                        self.converter_unblock.state_dict(),
                        self.converter_block.optimizer.state_dict(),
                        self.converter_block.scheduler.state_dict(),
                        self.converter_unblock.optimizer.state_dict(),
                        self.converter_unblock.scheduler.state_dict(),
                        epoch), self.save_dir + "/ckpt_epoch_" + str(epoch) + ".pth")
        else:
            torch.save((self.gaussians.capture(),
                        self.converter.state_dict(),
                        self.converter.optimizer.state_dict(),
                        self.converter.scheduler.state_dict(),
                        epoch), self.save_dir + "/ckpt_epoch_" + str(epoch) + ".pth")

    def load_checkpoint(self, path):
        if self.gaussians_unblock is not None:
            (gaussian_params, converter_block_sd, converter_unblock_sd, converter_opt_sd, converter_scd_sd, converter_unblock_opt_sd, converter_unblock_scd_sd, first_iter) = torch.load(path)
            self.gaussians.restore(gaussian_params, self.cfg.opt)
            self.converter_block.load_state_dict(converter_block_sd)
            self.converter_unblock.load_state_dict(converter_unblock_sd)
            # self.converter.optimizer.load_state_dict(converter_opt_sd)
            # self.converter.scheduler.load_state_dict(converter_scd_sd)
        else:
            (gaussian_params, scene_gaussian_params, converter_sd, converter_opt_sd, converter_scd_sd, first_iter) = torch.load(path)
            self.gaussians_scene.restore(scene_gaussian_params, self.cfg.opt)
            self.gaussians.restore(gaussian_params, self.cfg.opt)
            self.converter.load_state_dict(converter_sd)
            # self.converter.optimizer.load_state_dict(converter_opt_sd)
            # self.converter.scheduler.load_state_dict(converter_scd_sd)