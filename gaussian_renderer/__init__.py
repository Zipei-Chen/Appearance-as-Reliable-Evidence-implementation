

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import numpy as np
# from utils.dataset_utils import get_02v_bone_transforms, fetchPly, storePly, AABB
from plyfile import PlyData, PlyElement
import pickle
import os


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

def render(data,
           iteration,
           scene,
           pipe,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           compute_loss=True,
           return_opacity=False, recording_name=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp, modified_pose = scene.convert_gaussians(data, iteration, compute_loss)

    if iteration % 500 == 0:
        os.makedirs("C:/RoHM/datasets/PROX/recordings/{}/tmp_result/{}".format(recording_name, iteration),
            exist_ok=True)

        rgb = (colors_precomp.detach().cpu() * 255).numpy().astype(np.uint8)
        storePly("C:/RoHM/datasets/PROX/recordings/{}/tmp_result/{}/deformed_pose_add_color.ply".format(recording_name, iteration), pc.get_xyz.detach().cpu(), rgb)

    # pkl_path = "C:/RoHM/datasets/PROX/recordings/N0Sofa_00034_02/deformed_pose_add_color.pkl"
    # with open(pkl_path, 'wb') as result_file:
    #     pickle.dump(pc.get_xyz.detach().cpu(), result_file, protocol=2)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        # projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # calculate_view_range(data.FoVx, data.FoVy)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # opacity = opacity * 10

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

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
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    opacity_image = None
    if return_opacity:
        opacity_image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
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




    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            "modified_pose": modified_pose,
            }


def render_background(data,
           iteration,
           scene,
           pipe,
           bg_color: torch.Tensor,
           scaling_modifier=1.0,
           override_color=None,
           compute_loss=True,
           return_opacity=False, recording_name=None, dataset_name=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    pc, loss_reg, colors_precomp = scene.convert_gaussians(data, iteration, compute_loss)
    # np.save("C:/RoHM/datasets/PROX/recordings/{}/scene_color.npy".format(recording_name), colors_precomp.detach().cpu().numpy())
    # sys.exit()

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
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

    raster_settings = GaussianRasterizationSettings(
        image_height=int(data.image_height),
        image_width=int(data.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data.world_view_transform,
        projmatrix=data.full_proj_transform,
        # projmatrix=projmatrix,
        sh_degree=pc.active_sh_degree,
        campos=data.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    # calculate_view_range(data.FoVx, data.FoVy)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    # opacity = opacity * 10

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

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
    rendered_image, radii, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    opacity_image = None
    if return_opacity:
        opacity_image, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=torch.ones(opacity.shape[0], 3, device=opacity.device),
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

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"deformed_gaussian": pc,
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "loss_reg": loss_reg,
            "opacity_render": opacity_image,
            # "modified_pose": modified_pose,
            }
