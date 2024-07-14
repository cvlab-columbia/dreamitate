import os
from copy import deepcopy
import json
import sys
import trimesh
import glob
import numpy as np
from pathlib import Path

import argparse
from bop_toolkit_lib import config
from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import renderer
from bop_toolkit_lib import visibility


# PARAMETERS.
################################################################################
p = {
  # See dataset_params.py for options.
  'dataset': 'itodd',

  # Dataset split. Options: 'train', 'val', 'test'.
  'dataset_split': 'train_pbr_v2',

  # Dataset split type. None = default. See dataset_params.py for options.
  'dataset_split_type': 'pbr',

  # Whether to save visualizations of visibility masks.
  'vis_visibility_masks': False,

  # Tolerance used in the visibility test [mm].
  'delta': 15,

  # Type of the renderer.
  'renderer_type': 'cpp',  # Options: 'cpp', 'python'.

  # Folder containing the BOP datasets.
  'datasets_path': config.datasets_path,

  # Path template for output images with object masks.
  'vis_mask_visib_tpath': os.path.join(
    config.output_path, 'vis_gt_visib_delta={delta}',
    'vis_gt_visib_delta={delta}', '{dataset}', '{split}', '{scene_id:06d}',
    '{im_id:06d}_{gt_id:06d}.jpg'),
}

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

################################################################################
################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--chunk-dir', type=str)
parser.add_argument('--shapenet-dir', type=str)
parser.add_argument('--gso-dir', type=str)
parser.add_argument('--renderer-type', type=str, default='cpp')
parser.add_argument('--overwrite-models', action='store_true')
args = parser.parse_args()

chunk_dir = Path(args.chunk_dir)
chunk_infos = json.loads((chunk_dir / 'chunk_infos.json').read_text())
cam_infos_path = (chunk_dir / 'bop_data/camera.json')
cam_infos = json.loads(cam_infos_path.read_text())

scene_gt_tpath = (chunk_dir / 'bop_data/train_pbr/{scene_id:06d}/scene_gt.json')
scene_gt_info_tpath = (chunk_dir / 'bop_data/train_pbr/{scene_id:06d}/scene_gt_info.json')
depth_gt_info_tpath = (chunk_dir / 'bop_data/train_pbr/{scene_id:06d}/depth/{im_id:06d}.png')
vis_mask_visib_tpath = (chunk_dir / 'bop_data/train_pbr/{scene_id:06d}/mask_visib/{im_id:06d}_{inst_id:06d}.png')

if args.shapenet_dir:
    shapenet_dir = Path(args.shapenet_dir)
    is_shapenet = True
else:
    is_shapenet = False
    gso_dir = Path(args.gso_dir)
scale = chunk_infos['scale']

p = dict(
    dataset=chunk_dir,
    dataset_split='train_pbr',
    dataset_split_type='train_pbr',
    # renderer_type='python',
    delta=15,
)
p['renderer_type'] = args.renderer_type


# Initialize a renderer.
im_width, im_height = cam_infos['width'], cam_infos['height']
ren_width, ren_height = 3 * im_width, 3 * im_height
ren_cx_offset, ren_cy_offset = im_width, im_height
large_ren = renderer.create_renderer(
  ren_width, ren_height, p['renderer_type'],
    mode='depth')

misc.log('Initializing renderer...')
obj_name_to_id = dict()
for obj_id, obj in enumerate(chunk_infos['scene_infos']['objects']):
    if is_shapenet:
        synset_id, source_id = obj['synset_id'], obj['source_id']
        obj_name = obj['category_id']
        ply_path = Path(shapenet_dir) / f'{synset_id}/{source_id}' / 'models/model_normalized_scaled.ply'
    else:
        obj_name = obj['category_id']
        gso_id = obj_name.split('gso_')[1]
        ply_path = Path(gso_dir) / f'{gso_id}' / 'meshes/model.ply'
    obj_name_to_id[obj_name] = obj_id
    large_ren.add_object(obj_id, str(ply_path))

scene_ids = [0]

misc.log(f'Processing scene ids: {scene_ids}')

for scene_id in scene_ids:
    # Load scene info and ground-truth poses.
    scene_dir =  chunk_dir / f'bop_data/train_pbr/{scene_id:06d}'
    scene_camera = inout.load_scene_camera(scene_dir / 'scene_camera.json')
    scene_gt = inout.load_scene_gt(str(scene_gt_tpath).format(scene_id=scene_id))

    scene_gt_info = {}
    im_ids = sorted(scene_gt.keys())
    for im_counter, im_id in enumerate(im_ids):
        depth_path = str(scene_dir / f'depth/{im_id:06d}.png')
        K = scene_camera[im_id]['cam_K']
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # Load depth image.
        depth_im = inout.load_depth(depth_path)
        depth_im *= scene_camera[im_id]['depth_scale']  # to [mm]
        dist_im = misc.depth_im_to_dist_im_fast(depth_im, K)
        im_size = (depth_im.shape[1], depth_im.shape[0])


        # Calc gt info
        if im_counter % 5 == 0:
            misc.log(
                'Calculating GT info - dataset: {} ({}, {}), scene: {}, im: {}'.format(
                p['dataset'], p['dataset_split'], p['dataset_split_type'], scene_id, im_id))

        scene_gt_info[im_id] = []
        for gt_id, gt in enumerate(scene_gt[im_id]):
            if gt['obj_id'] not in obj_name_to_id:
                continue
            # Render depth image of the object model in the ground-truth pose.
            depth_gt_large = large_ren.render_object(
                obj_name_to_id[gt['obj_id']], gt['cam_R_m2c'], gt['cam_t_m2c'],
                fx, fy, cx + ren_cx_offset, cy + ren_cy_offset)['depth']
            depth_gt = depth_gt_large[
                        ren_cy_offset:(ren_cy_offset + im_height),
                        ren_cx_offset:(ren_cx_offset + im_width)]

            # Convert depth images to distance images.
            # dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
            # dist_im = misc.depth_im_to_dist_im(depth, K)
            #
            dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
            dist_im = misc.depth_im_to_dist_im_fast(depth_im, K)

            # Estimation of the visibility mask.
            visib_gt = visibility.estimate_visib_mask_gt(
            dist_im, dist_gt, p['delta'], visib_mode='bop19')

            # Mask of the object in the GT pose.
            obj_mask_gt_large = depth_gt_large > 0
            obj_mask_gt = dist_gt > 0

            # Number of pixels in the whole object silhouette
            # (even in the truncated part).
            px_count_all = np.sum(obj_mask_gt_large)

            # Number of pixels in the object silhouette with a valid depth measurement
            # (i.e. with a non-zero value in the depth image).
            px_count_valid = np.sum(dist_im[obj_mask_gt] > 0)

            # Number of pixels in the visible part of the object silhouette.
            px_count_visib = visib_gt.sum()

            # Visible surface fraction.
            if px_count_all > 0:
                visib_fract = px_count_visib / float(px_count_all)
            else:
                visib_fract = 0.0

            # Bounding box of the whole object silhouette
            # (including the truncated part).
            bbox = [-1, -1, -1, -1]
            if px_count_visib > 0:
                ys, xs = obj_mask_gt_large.nonzero()
                ys -= ren_cy_offset
                xs -= ren_cx_offset
                bbox = misc.calc_2d_bbox(xs, ys, im_size)

            # Bounding box of the visible surface part.
            bbox_visib = [-1, -1, -1, -1]
            if px_count_visib > 0:
                ys, xs = visib_gt.nonzero()
                bbox_visib = misc.calc_2d_bbox(xs, ys, im_size)

            # Store the calculated info.
            scene_gt_info[im_id].append({
                'px_count_all': int(px_count_all),
                'px_count_valid': int(px_count_valid),
                'px_count_visib': int(px_count_visib),
                'visib_fract': float(visib_fract),
                'bbox_obj': [int(e) for e in bbox],
                'bbox_visib': [int(e) for e in bbox_visib]
            })
    # Save the info for the current scene.
    scene_gt_info_path = str(scene_gt_info_tpath).format(scene_id=scene_id)
    misc.ensure_dir(os.path.dirname(scene_gt_info_path))
    inout.save_json(scene_gt_info_path, scene_gt_info)
