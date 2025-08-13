

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os
import json

class Camera:
    def __init__(self, camera=None, is_two_person=False, **kwargs):
        if camera is not None:
            self.data = camera.data.copy()
            return

        self.data = kwargs
        self.data['trans'] = np.array([0.0, 0.0, 0.0])
        self.data['scale'] = 1.0

        self.data['original_image'] = self.image.clamp(0.0, 1.0).to(self.data_device)
        self.data['image_width'] = self.original_image.shape[2]
        self.data['image_height'] = self.original_image.shape[1]
        self.data['original_mask'] = self.mask.float().to(self.data_device)

        self.data['zfar'] = 100.0
        self.data['znear'] = 0.01

        # cam2world_dir = os.path.join("C:/RoHM/datasets/PROX", 'cam2world')
        # scene_name = "N0Sofa"
        # with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
        #     self.data['world_view_transform'] = torch.tensor(np.linalg.inv(np.array(json.load(f))).astype(np.float32)).cuda()

        self.data['world_view_transform'] = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.data['projection_matrix'] = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).cuda()
        self.data['full_proj_transform'] = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.data['camera_center'] = self.world_view_transform.inverse()[3, :3]

        if is_two_person:
            self.data['rots_block'] = self.rots_block.to(self.data_device)
            self.data['Jtrs_block'] = self.Jtrs_block.to(self.data_device)
            self.data['bone_transforms_block'] = self.bone_transforms_block.to(self.data_device)

            self.data['rots_unblock'] = self.rots_unblock.to(self.data_device)
            self.data['Jtrs_unblock'] = self.Jtrs_unblock.to(self.data_device)
            self.data['bone_transforms_unblock'] = self.bone_transforms_unblock.to(self.data_device)

        else:
            self.data['rots'] = self.rots.to(self.data_device)
            self.data['Jtrs'] = self.Jtrs.to(self.data_device)
            self.data['bone_transforms'] = self.bone_transforms.to(self.data_device)


    def __getattr__(self, item):
        return self.data[item]

    def update(self, **kwargs):
        self.data.update(kwargs)

    def copy(self):
        new_cam = Camera(camera=self)
        return new_cam

    def merge(self, cam):
        self.data['frame_id'] = cam.frame_id
        self.data['rots'] = cam.rots.detach()
        self.data['Jtrs'] = cam.Jtrs.detach()
        self.data['bone_transforms'] = cam.bone_transforms.detach()
