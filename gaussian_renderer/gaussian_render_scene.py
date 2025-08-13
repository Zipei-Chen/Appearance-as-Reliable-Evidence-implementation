

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np
# from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from plyfile import PlyData, PlyElement
import pickle
import os
from utils.general_utils import build_rotation


def calculate_view_range(fovx, fovy, znear, zfar):
    tan_half_fovx = math.tan(fovx * 0.5)
    tan_half_fovy = math.tan(fovy * 0.5)

    # 计算视野范围
    width = 2 * znear * tan_half_fovx
    height = 2 * znear * tan_half_fovy

    print("View range (width, height, depth):")
    print("Width:", width)
    print("Height:", height)
    print("Depth (near to far plane):", znear, "to", zfar)

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

def render_add_scene(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False, recording_name=None, dataset_name=None, is_directly=False, only_human=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if scene.gaussians_unblock is not None:
        pc, pc_scene, loss_reg, colors_precomp, colors_precomp_scene, modified_pose, pc_unblock, loss_reg_unblock, colors_precomp_unblock, modified_pose_unblock = scene.convert_gaussians(data, iteration, compute_loss, is_directly)
        total_gaussians = pc.clone()
        total_gaussians._xyz = torch.cat([pc._xyz, pc_scene._xyz, pc_unblock._xyz], dim=0)
        # total_gaussians._opacity = torch.cat([pc._opacity, pc_scene._opacity, pc_unblock.opacity], dim=0)
        # total_gaussians._rotation = torch.cat([pc.rotation_precomp, build_rotation(pc_scene._rotation), pc_unblock.rotation_precomp], dim=0)

        setattr(total_gaussians, 'rotation_precomp', torch.cat([pc.rotation_precomp, build_rotation(pc_scene._rotation), pc_unblock.rotation_precomp], dim=0))
        total_gaussians._scaling = torch.cat([pc._scaling, pc_scene._scaling, pc_unblock._scaling], dim=0)
        total_color_precomp = torch.cat([colors_precomp, colors_precomp_scene, colors_precomp_unblock], dim=0)
        opacity = torch.cat([pc.get_opacity, pc_scene.get_opacity, pc_unblock.get_opacity], dim=0)

        loss_reg = {key: loss_reg[key] + loss_reg_unblock[key] for key in loss_reg}
        # loss_reg = loss_reg + loss_reg_unblock
    else:
        pc, pc_scene, loss_reg, colors_precomp, colors_precomp_scene, modified_pose = scene.convert_gaussians(data, iteration, compute_loss, is_directly)
        total_gaussians = pc.clone()
        total_gaussians._xyz = torch.cat([pc._xyz, pc_scene._xyz], dim=0)
        # total_gaussians._opacity = torch.cat([pc._opacity, pc_scene._opacity], dim=0)
        # total_gaussians._rotation = torch.cat([pc.rotation_precomp, build_rotation(pc_scene._rotation)], dim=0)
        setattr(total_gaussians, 'rotation_precomp', torch.cat([pc.rotation_precomp, build_rotation(pc_scene._rotation)], dim=0))
        total_gaussians._scaling = torch.cat([pc._scaling, pc_scene._scaling], dim=0)
        total_color_precomp = torch.cat([colors_precomp, colors_precomp_scene], dim=0)
        opacity = torch.cat([pc.get_opacity, pc_scene.get_opacity], dim=0)

        # os.makedirs(os.path.join("C:/RoHM/datasets/PROX/recordings", "{}/tmp_human_mesh".format(recording_name)), exist_ok=True)
        # if iteration %50 == 0:
        #     rgb = (colors_precomp.detach().cpu() * 255).numpy().astype(np.uint8)
        #     storePly(os.path.join("C:/RoHM/datasets/PROX/recordings", "{}/tmp_human_mesh".format(recording_name),
        #                           "{}.ply".format(iteration)), pc.get_xyz.detach().cpu(), rgb)

    if iteration % 1000 == 0:
        if dataset_name == "prox_add_scene":
            root = "C:/RoHM/datasets/PROX/recordings"
        elif dataset_name == "egobody_add_scene":
            root = "C:/RoHM/datasets/EgoBody/kinect_color"
        elif dataset_name == "i3db_add_scene":
            root = "C:/humor/data/iMapper/i3DB"
        elif dataset_name == "emdb_add_scene":
            root = "c:/emdb/P0"
        os.makedirs(os.path.join(root, "{}/tmp_result_add_scene/{}".format(recording_name, iteration)),
            exist_ok=True)

        rgb = (total_color_precomp.detach().cpu() * 255).numpy().astype(np.uint8)
        storePly(os.path.join(root, "{}/tmp_result_add_scene/{}".format(recording_name, iteration), "deformed_pose_add_color.ply"), total_gaussians.get_xyz.detach().cpu(), rgb)

    # pkl_path = "C:/RoHM/datasets/PROX/recordings/N0Sofa_00034_02/deformed_pose_add_color.pkl"
    # with open(pkl_path, 'wb') as result_file:
    #     pickle.dump(pc.get_xyz.detach().cpu(), result_file, protocol=2)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(total_gaussians.get_xyz, dtype=total_gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # data.FoVx = 5.0
    # data.FoVy = 5.0

    # Set up rasterization configuration
    tanfovx = math.tan(data.FoVx * 0.5)
    tanfovy = math.tan(data.FoVy * 0.5)

    # data.world_view_transform = data.world_view_transform.transpose(0, 1)
    # projmatrix = torch.from_numpy(np.eye(4).astype(np.float32)).cuda()
    # data.camera_center = torch.tensor([ 0.0,  0.0, 0.0], device='cuda:0')

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        sh_degree=total_gaussians.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # calculate_view_range(data.FoVx, data.FoVy)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = total_gaussians.get_xyz
    means2D = screenspace_points
    # opacity = total_gaussians.get_opacity

    # opacity = opacity * 10

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    # pipe.compute_cov3D_python = False
    if pipe.compute_cov3D_python:
        cov3D_precomp = total_gaussians.get_covariance(scaling_modifier)
    else:
        scales = total_gaussians.get_scaling
        rotations = total_gaussians.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None

    # save the cov_matrix and 3d vertix

    # pkl_path = os.path.join("C:/3dgs-avatar-release/cov3d_matrix.pkl")
    # with open(pkl_path, 'wb') as result_file:
    #     pickle.dump(cov3D_precomp.cpu(), result_file, protocol=2)
    #
    # pkl_path = os.path.join("C:/3dgs-avatar-release/3d_xyz.pkl")
    # with open(pkl_path, 'wb') as result_file:
    #     pickle.dump(means3D.cpu(), result_file, protocol=2)


    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    # def transformPoint44(means3D, full_proj_transform):
    #
    #     return np.array([(torch.matmul(means3D, full_proj_transform[:3, 0]) + full_proj_transform[3, 0]).detach().cpu(),
    #             (torch.matmul(means3D, full_proj_transform[:3, 1]) + full_proj_transform[3, 1]).detach().cpu(),
    #             (torch.matmul(means3D, full_proj_transform[:3, 2]) + full_proj_transform[3, 2]).detach().cpu(),
    #             (torch.matmul(means3D, full_proj_transform[:3, 3]) + full_proj_transform[3, 3]).detach().cpu()
    #             ])
    #
    # p_hom = transformPoint44(means3D, data.full_proj_transform)
    # p_hom = torch.matmul(torch.cat([means3D, torch.ones([1, 1]).cuda()], dim=1), data.full_proj_transform).squeeze()
    # x_world = torch.matmul(torch.cat([means3D, torch.ones([1, 1]).cuda()], dim=1), data.world_view_transform)
    # p_hom_tmp = torch.matmul(x_world, data.projection_matrix)
    # p_w = 1.0 / (p_hom[3] + 0.0000001)
    # p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]
    # p_proj_xy = [((p_proj[0] + 1.0) * 1920 ) * 0.5, ((p_proj[1] + 1.0) * 1080 ) * 0.5]

    if only_human:
        means3D = pc._xyz
        means2D = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            means2D.retain_grad()
        except:
            pass

        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            # colors_precomp = torch.tensor([[0, 0.0, 0], [0, 0.0, 0]], device=opacity.device),
            opacities=pc.get_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=pc.get_covariance(scaling_modifier)
            # cov3D_precomp = torch.tensor([[1e-10, 0, 0, 1e-10, 0, 1e-10], [1e-10, 0, 0, 1e-10, 0, 1e-10]], device=opacity.device),
        )
    else:
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = total_color_precomp,
            # colors_precomp = torch.tensor([[0, 0.0, 0], [0, 0.0, 0]], device=opacity.device),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
            # cov3D_precomp = torch.tensor([[1e-10, 0, 0, 1e-10, 0, 1e-10], [1e-10, 0, 0, 1e-10, 0, 1e-10]], device=opacity.device),
        )
    opacity_image = None
    if return_opacity:
        opacity_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.cat([torch.ones(10475, 3, device=opacity.device), torch.zeros(opacity.shape[0] - 10475, 3, device=opacity.device)], dim=0),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        opacity_image = opacity_image[:1]

        # with open("C:/Paper/vertic_tmp.pkl", 'rb') as f:
        #     mesh_tmp = pickle.load(f)
        #
        # opacity_image, _ = rasterizer(
        #     means3D=torch.from_numpy(mesh_tmp.astype(np.float32)),
        #     means2D=means2D,
        #     shs=None,
        #     colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
        #     opacities=opacity,
        #     scales=scales,
        #     rotations=rotations,
        #     cov3D_precomp=cov3D_precomp)


    # add the scene render
    scene_rendered_image = None
    scene_screenspace_points = torch.zeros_like(pc_scene.get_xyz, dtype=pc_scene.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        scene_screenspace_points.retain_grad()
    except:
        pass

    scene_means3D = pc_scene.get_xyz
    scene_means2D = scene_screenspace_points

    scene_rendered_image, _ = rasterizer(
        means3D=scene_means3D,
        means2D=scene_means2D,
        shs=shs,
        colors_precomp=colors_precomp_scene,
        # colors_precomp = torch.tensor([[0, 0.0, 0], [0, 0.0, 0]], device=opacity.device),
        opacities=pc_scene.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=pc_scene.get_covariance(scaling_modifier)
        # cov3D_precomp = torch.tensor([[1e-10, 0, 0, 1e-10, 0, 1e-10], [1e-10, 0, 0, 1e-10, 0, 1e-10]], device=opacity.device),
    )

    # add the hunman opacity render
    if scene.gaussians_unblock is not None:
        human_opacity_rendered_image = None
        human_opacity_screenspace_points = torch.zeros_like(torch.cat([pc.get_xyz, pc_unblock.get_xyz], dim=0), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            human_opacity_screenspace_points.retain_grad()
        except:
            pass
        human_opacity_means3D = torch.cat([pc.get_xyz, pc_unblock.get_xyz], dim=0)
        human_opacity_means2D = human_opacity_screenspace_points

        human_opacity_rendered_image, _ = rasterizer(
            means3D=human_opacity_means3D,
            means2D=human_opacity_means2D,
            shs=shs,
            colors_precomp=torch.ones(pc.get_opacity.shape[0] + pc_unblock.get_opacity.shape[0], 3, device=opacity.device),
            # colors_precomp = torch.tensor([[0, 0.0, 0], [0, 0.0, 0]], device=opacity.device),
            opacities=torch.cat([pc.get_opacity, pc_unblock.get_opacity], dim=0),
            scales=scales,
            rotations=rotations,
            cov3D_precomp=torch.cat([pc.get_covariance(scaling_modifier), pc_unblock.get_covariance(scaling_modifier)], dim=0)
            # cov3D_precomp = torch.tensor([[1e-10, 0, 0, 1e-10, 0, 1e-10], [1e-10, 0, 0, 1e-10, 0, 1e-10]], device=opacity.device),
        )

    else:
        human_opacity_rendered_image = None
        human_opacity_screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            human_opacity_screenspace_points.retain_grad()
        except:
            pass

        human_opacity_means3D = pc.get_xyz
        human_opacity_means2D = human_opacity_screenspace_points

        human_opacity_rendered_image, _ = rasterizer(
            means3D=human_opacity_means3D,
            means2D=human_opacity_means2D,
            shs=shs,
            colors_precomp=torch.ones(pc.get_opacity.shape[0], 3, device=opacity.device),
            # colors_precomp = torch.tensor([[0, 0.0, 0], [0, 0.0, 0]], device=opacity.device),
            opacities=pc.get_opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=pc.get_covariance(scaling_modifier)
            # cov3D_precomp = torch.tensor([[1e-10, 0, 0, 1e-10, 0, 1e-10], [1e-10, 0, 0, 1e-10, 0, 1e-10]], device=opacity.device),
        )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    if scene.gaussians_unblock is not None:
        return {"deformed_gaussian": pc,
                "scene_gaussian": pc_scene,
                "deformed_unblock_gaussian": pc_unblock,
                "total_gaussian": total_gaussians,
                "render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "loss_reg": loss_reg,
                "opacity_render": opacity_image,
                "modified_pose": modified_pose,
                "modified_pose_unblock": modified_pose_unblock,
                "scene_rendered_image": scene_rendered_image,
                "human_opacity_rendered_image": human_opacity_rendered_image,
                "visible_human_3d_point": torch.cat([total_gaussians.get_xyz[:10475][(radii>0)[:10475]], total_gaussians.get_xyz[-10475:][(radii>0)[-10475:]]], dim=0),
                }
    else:
        return {"deformed_gaussian": pc,
                    "scene_gaussian": pc_scene,
                    "total_gaussian": total_gaussians,
                    "render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "visibility_filter": radii > 0,
                    "radii": radii,
                    "loss_reg": loss_reg,
                    "opacity_render": opacity_image,
                    "modified_pose": modified_pose,
                    "scene_rendered_image": scene_rendered_image,
                    "human_opacity_rendered_image": human_opacity_rendered_image,
                    "visible_human_3d_point": pc.get_xyz[:10475][(radii>0)[:10475]],
                }
