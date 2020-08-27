from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

import open3d as o3d
import pulp

from fit_ransac import fit_plane_RANSAC, fit_line_RANSAC
from rl_config import *
from layout_structs import *
from search_polygons import *
from utils import *

class RoomLayoutEstimator:
    def __init__(self, example):

        # Initialize randomness
        np.random.seed(seed=seed)

        # Load color image
        img_path = example_folder + "color/" + str(example) + ".jpg"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))

        # Load planes masks
        planes_masks_path = example_folder + "planercnn_seg/" + str(example) + ".npy"
        planes_masks = np.load(planes_masks_path)

        # Load depth
        depth_path = example_folder + "depth/" + str(example) + ".png"
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
        depth = cv2.resize(depth, (640, 480)).astype(np.float32)

        depth[np.greater(depth,max_depth_value)] = 0

        # Load filled depth
        depth_filled_path = example_folder + "depth_filled/" + str(example) + ".png"
        depth_filled = cv2.imread(depth_filled_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
        depth_filled = cv2.resize(depth_filled, (640, 480)).astype(np.float32)

        depth_filled[np.greater(depth_filled,max_depth_value)] = 0


        # Load semantic segmentation
        segmentation_path = example_folder + "mseg_labels/" + str(example) + ".png"
        segmentation_orig = cv2.imread(segmentation_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        segmentation_conf_path = example_folder + "mseg_confidence/" + str(example) + ".png"
        seg_confidence = cv2.imread(segmentation_conf_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 2 ** 8

        segmentation = np.zeros_like(segmentation_orig)

        wall_indices_bool = np.isin(segmentation_orig, wall_indices)
        floor_indices_bool = np.isin(segmentation_orig, floor_indices)
        ceil_indices_bool = np.isin(segmentation_orig, ceiling_indices)
        segmentation[wall_indices_bool] = wall_type * (seg_confidence[wall_indices_bool] > seg_confidence_wall_thresh)
        segmentation[floor_indices_bool] = floor_type * (seg_confidence[floor_indices_bool] > seg_confidence_floor_thresh)
        segmentation[ceil_indices_bool] = ceil_type * (seg_confidence[ceil_indices_bool] > seg_confidence_ceil_thresh)

        # Create image boundary mask (predictions around image border are often bad)
        boundary_mask = np.zeros_like(img[:, :, 0])
        cv2.rectangle(boundary_mask, (0, 0), (depth.shape[1] - 1, depth.shape[0] - 1), 1.,
                      thickness=boundary_mask_thickness)
        boundary_mask = 1. - boundary_mask

        height, width = img.shape[:2]

        if plot_intermediate_results:
            plt.figure()
            plt.subplot(141)
            plt.imshow(img)
            plt.subplot(142)
            plt.imshow(depth, vmin=0, vmax=10)
            plt.subplot(143)
            plt.imshow(segmentation)
            plt.subplot(144)
            vis_plane_seg = np.zeros_like(segmentation)
            for i, plane_mask in enumerate(planes_masks):
                vis_plane_seg += (plane_mask * (i + 1)).astype(np.uint8)
            plt.imshow(vis_plane_seg)
            plt.show()


        # ScanNet intrinsics
        sx = width / 1296
        sy = height / 968
        K = np.reshape(np.array([1170.187988 * sx, 0., 647.750000 * sx, 0., 1170.187988 * sy, 483.75 * sy, 0., 0., 1.]), newshape=(3, 3))

        self.example = example
        self.img = img / 255
        self.depth = depth
        self.depth_filled = depth_filled
        self.planes_masks = planes_masks
        self.segmentation = segmentation
        layout_seg_mask = np.isin(segmentation, floor_wall_ceil)
        self.layout_seg_mask = layout_seg_mask
        # self.layout_seg_no_refl_mask = layout_seg_no_refl_mask
        self.depth_layout_masked = layout_seg_mask * depth
        # self.depth_layout_no_refl_masked = layout_seg_no_refl_mask * depth

        self.val_seg_indices = floor_wall_ceil
        self.boundary_mask = boundary_mask

        self.h, self.w = h, w = self.depth.shape[:2]
        self.set_helpers(K)


        # Calculate flattened one hot segmentation
        seg = cv2.resize(segmentation.astype(np.uint8), (w, h))
        self.seg_flat = seg.reshape(-1)
        self.seg_one_hot_flat = np.eye(4)[self.seg_flat].T
        self.val_ind = floor_wall_ceil

        self.use_virtual_floor_plane = False

    def set_helpers(self, K):
        '''
        Set helpers

        :param K:
        :return:
        '''
        h, w = self.h, self.w

        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.coordinate_grid, self.coordinate_grid_2d = get_coord_grid(h, w)

        self.K_dot_coord = np.dot(self.K_inv, self.coordinate_grid)

        depth_flat = np.reshape(self.depth, newshape=(1, -1))
        self.depth_mask_flat = np.greater(depth_flat, min_depth)
        self.depth_mask = self.depth_mask_flat.reshape((h, w))

        self.depth_K_dot_coord = depth_flat * self.K_dot_coord

        self.K_dot_coord = np.dot(self.K_inv, self.coordinate_grid)

        depth_filled_flat = np.reshape(self.depth_filled, newshape=(1, -1))
        self.depth_filled_mask_flat = np.greater(depth_filled_flat, min_depth)
        self.depth_filled_mask = self.depth_filled_mask_flat.reshape((h, w))

        self.depth_filled_K_dot_coord = depth_filled_flat * self.K_dot_coord

        self.K_dot_coord_2d = np.dot(self.coordinate_grid_2d, np.transpose(self.K_inv))
        self.depth_K_dot_coord_2d = self.depth[:,:,None] * self.K_dot_coord_2d

    def calc_frustum_planes(self):
        '''
        Calculate frustum planes

        :return: A list of frustum planes (LayoutPlane)
        '''

        h, w = self.h, self.w
        K_inv = self.K_inv

        c1 = np.array([0, 0, 1])
        c2 = np.array([w - 1, 0, 1])
        c3 = np.array([0, h - 1, 1])
        c4 = np.array([w - 1, h - 1, 1])
        v1 = K_inv.dot(c1)
        v2 = K_inv.dot(c2)
        v3 = K_inv.dot(c3)
        v4 = K_inv.dot(c4)

        n12 = np.cross(v1, v2)
        n12 = n12 / np.sqrt(n12[0] ** 2 + n12[1] ** 2 + n12[2] ** 2)

        n13 = -np.cross(v1, v3)
        n13 = n13 / np.sqrt(n13[0] ** 2 + n13[1] ** 2 + n13[2] ** 2)

        n24 = -np.cross(v2, v4)
        n24 = n24 / np.sqrt(n24[0] ** 2 + n24[1] ** 2 + n24[2] ** 2)

        n34 = -np.cross(v3, v4)
        n34 = n34 / np.sqrt(n34[0] ** 2 + n34[1] ** 2 + n34[2] ** 2)

        plane1 = LayoutPlane(plane=np.concatenate((n12, [0])), mask=np.ones((h, w)), type=-1)
        plane2 = LayoutPlane(plane=np.concatenate((n13, [0])), mask=np.ones((h, w)), type=-1)
        plane3 = LayoutPlane(plane=np.concatenate((n24, [0])), mask=np.ones((h, w)), type=-1)
        plane4 = LayoutPlane(plane=np.concatenate((n34, [0])), mask=np.ones((h, w)), type=-1)

        frustum_planes = [plane1, plane2, plane3, plane4]

        return frustum_planes

    def merge_planes_iteratively(self, layout_planes):
        '''
        Merge planes iteratively

        :param layout_planes:
        :return: list of merged planes (LayoutPlane)
        '''
        print("Merging planes...")

        while True:
            merged_layout_planes = self.merge_planes(layout_planes)

            if len(merged_layout_planes) == len(layout_planes):
                break
            else:
                layout_planes = merged_layout_planes

        return merged_layout_planes

    def merge_planes(self, layout_planes):
        '''
        Merge planes

        :param layout_planes:
        :return: list of merged planes (LayoutPlane)
        '''
        depth_K_dot_coord_2d = self.depth_K_dot_coord_2d
        depth_mask = self.depth_mask

        are_planes_merged = [False] * len(layout_planes)

        # Prepare planes masks
        planes_masks = []
        for plane_index, layout_plane in enumerate(layout_planes):
            planes_masks.append(layout_plane.mask)

        # Merge planes
        merged_layout_planes = []
        for plane_index, layout_plane in enumerate(layout_planes):
            plane_params, plane_mask, plane_type = layout_plane.plane, layout_plane.mask, layout_plane.type
            is_plane_merged = are_planes_merged[plane_index]
            if is_plane_merged:
                continue

            merged_plane_mask = plane_mask
            merged_plane_params = plane_params
            merged_plane_type = plane_type

            # Compare the plane to all of the other planes
            for plane_index2, layout_plane2 in enumerate(layout_planes):
                if plane_index2 <= plane_index:
                    continue
                plane_params2, plane_mask2, plane_type2 = layout_plane2.plane, layout_plane2.mask, layout_plane2.type

                n1 = plane_params[:3]
                n2 = plane_params2[:3]

                # Compare the planes parameters
                if np.all(np.less(np.abs(np.cross(n1, n2)), merge_parallel_threshold)) and \
                                np.abs(plane_params[3] - plane_params2[3]) < merge_offset_threshold:

                    # Check whether the two planes are neighbours
                    if self.check_if_planes_neighbs(planes_masks, plane_params, plane_mask, plane_params2, plane_mask2):
                        is_plane_merged = True

                        merged_plane_mask = np.clip(merged_plane_mask + plane_mask2, a_min=0., a_max=1.)
                        are_planes_merged[plane_index2] = True

                        depth_K_dot_coord_masked1 = depth_K_dot_coord_2d[(depth_mask * merged_plane_mask).astype(np.bool)]

                        depth_K_dot_coord_sampled = depth_K_dot_coord_masked1
                        ones = np.ones((depth_K_dot_coord_sampled.shape[0], 1))

                        depth_K_dot_coord_sampled_ones = np.concatenate((depth_K_dot_coord_sampled, ones), axis=1)

                        plane_params = self.calc_plane_params_from_pcd(depth_K_dot_coord_sampled_ones)
                        merged_plane_params = plane_params

            merged_layout_plane = LayoutPlane(
                plane=merged_plane_params, mask=merged_plane_mask, type=merged_plane_type)
            merged_layout_planes.append(merged_layout_plane)


        return merged_layout_planes

    def calc_valid_planes(self):
        '''
        Calculate valid planes

        :return: list of LayoutPlanes
        '''
        h = self.h
        w = self.w

        layout_planes = self.calc_valid_planes_ransac()

        layout_planes = self.merge_planes_iteratively(layout_planes)

        if plot_intermediate_results:
            planes_filtered_segs_image, normals_image_overlay = self.vis_layout_planes(layout_planes)
            plt.figure()
            plt.subplot(121)
            plt.imshow(planes_filtered_segs_image)
            plt.subplot(122)
            plt.imshow(normals_image_overlay)
            plt.show()


        return layout_planes

    def calc_valid_planes_ransac(self):
        '''
        Calculate valid planes

        :return: list of planes_params
        '''
        h = self.h
        w = self.w

        # Flatten planes masks
        planes_masks_flat = np.reshape(self.planes_masks, newshape=(-1, h * w)) * np.reshape(self.boundary_mask, (-1))

        # Find layout planes
        planes = []
        for plane_ind in range(len(planes_masks_flat)):
            layout_plane = self.calc_plane_params(plane_ind)
            if layout_plane is not None:
                planes.append(layout_plane)

        return planes

    def calc_plane_params_from_pcd(self, sampled_pc):
        '''
        Calculate plane from given PCD

        :param sampled_pc: point cloud (N, 4)
        :return: plane parameters
        '''

        plane_params, inlier_list1, outlier_list1 = fit_plane_RANSAC(sampled_pc, inlier_thresh=ransac_thresh)
        plane_params = norm_plane_params(plane_params)

        return plane_params

    def calc_plane_params(self, i, validate=True):
        '''
        Calculate planes parameters

        :param i: plane index
        :param validate: whether to validate the feasibility of plane
        :return: LayoutPlane
        '''

        # Get helper attributes
        h, w = self.h, self.w
        depth_mask_flat = self.depth_mask_flat
        depth_K_dot_coord = self.depth_K_dot_coord

        # Calculate flattened one hot segmentation
        seg_one_hot_flat = self.seg_one_hot_flat
        val_ind = self.val_seg_indices

        # Flatten planes masks
        planes_masks_flat = np.reshape(self.planes_masks, newshape=(-1, h * w)) * np.reshape(self.boundary_mask, (-1))
        valid_seg = (1. - (self.segmentation == 0))
        valid_seg_flat = valid_seg.reshape((-1))

        plane_mask_flat = planes_masks_flat[i]

        # Calculate overlap of valid semantic segmentations and given plane mask
        plane_seg_flat = valid_seg_flat * plane_mask_flat * seg_one_hot_flat
        plane_seg_flat_sum_classes = np.sum(plane_seg_flat, axis=1)

        plane_seg_overlap_joint = np.vstack([np.sum(plane_seg_flat_sum_classes[wall_type]),
                                       np.sum(plane_seg_flat_sum_classes[floor_type]),
                                       np.sum(plane_seg_flat_sum_classes[ceil_type])
                                       ]).sum(axis=0) / np.sum(plane_mask_flat)


        # Validate the plane based on val_thresholf
        if (validate and plane_seg_overlap_joint < val_threshold):
            return None

        # Determine the plane type
        plane_type = calc_plane_type(plane_mask_flat, plane_seg_flat_sum_classes, val_ind)
        if plane_type == -1:
            return None

        # Calculate plane parameters from 3D points of the given plane using RANSAC
        val_plane_indices = valid_layout_ind_dict[plane_type]
        plane_seg_mask_flat = np.sum(plane_seg_flat[val_plane_indices], axis=0)
        depth_K_dot_coord_masked1 = depth_K_dot_coord[:, (depth_mask_flat[0] * plane_seg_mask_flat).astype(np.bool)]

        depth_K_dot_coord_sampled = depth_K_dot_coord_masked1
        ones = np.ones((1, depth_K_dot_coord_sampled.shape[1]))
        depth_K_dot_coord_sampled_ones = np.concatenate((depth_K_dot_coord_sampled, ones), axis=0).T

        if depth_K_dot_coord_sampled_ones.shape[0] < min_plane_size:
            return None

        plane_params = self.calc_plane_params_from_pcd(depth_K_dot_coord_sampled_ones)

        layout_plane = LayoutPlane(
            plane=plane_params, mask=np.reshape(plane_seg_mask_flat, (h, w)), type=plane_type)

        return layout_plane

    def add_virtual_floor_plane(self, layout_planes):
        '''
        Add virtual floor plane if floor is not visible

        :param layout_planes:
        :return: LayoutPlane
        '''
        # Get Helpers
        h, w = self.h, self.w

        self.use_virtual_floor_plane = True
        self.use_virtual_ceil_plane = True

        for layout_plane in layout_planes:
            plane_params, plane_type = layout_plane.plane, layout_plane.type
            if plane_type == floor_type:
                floor_plane_params = plane_params
                self.use_virtual_floor_plane = False
            if plane_type == ceil_type:
                self.use_virtual_ceil_plane = False

        one_valid_floor_virtual = False
        if self.use_virtual_floor_plane:
            floor_perp_cand_list = []

            # Add virtual floor that is orthogonal to the walls in the scene
            for plane_ind, layout_plane in enumerate(layout_planes):
                plane_params, plane_mask, plane_type = layout_plane.plane, \
                                                       layout_plane.mask, layout_plane.type

                if plane_type == wall_type:
                    n1 = plane_params[:3]

                    orth_n1 = np.array([-n1[1], n1[0], 0.])
                    orth_n2 = np.array([0, -n1[2], n1[1]])

                    if np.abs(orth_n1[1]) > np.abs(orth_n2[1]):
                        orth_n = orth_n1
                    else:
                        orth_n = orth_n2

                    orth_n = norm_plane_params(orth_n)

                    if orth_n[1] > 0.:
                        orth_n *= -1

                    if np.abs(orth_n[1]) < 0.5:
                        continue

                    floor_perp_cand_list.append(orth_n)
                    one_valid_floor_virtual = True

            floor_plane_params = np.mean(np.array(floor_perp_cand_list), axis=0)
            if np.any(np.isnan(floor_plane_params)):
                self.use_virtual_floor_plane = False
                return None, self.use_virtual_floor_plane

            floor_plane_params = np.append(floor_plane_params, [[camera_height]])
            print("Virt. floor plane params", floor_plane_params)
            floor_plane_params = norm_plane_params(floor_plane_params)

        if one_valid_floor_virtual:
            floor_plane = LayoutPlane(plane=floor_plane_params, mask=np.zeros((h, w)), type=floor_type)
            return floor_plane
        else:
            self.use_virtual_floor_plane = False
            return None

    def get_floor_planes(self, layout_planes):
        found_floor_plane = False
        floor_plane_params = None
        for layout_plane in layout_planes:
            plane_params, plane_type = layout_plane.plane, layout_plane.type
            d = plane_params[-1]
            if plane_type == floor_type:
                if not found_floor_plane or d < floor_plane_params[-1]:
                    floor_plane_params = plane_params
                    found_floor_plane = True
        return floor_plane_params

    def check_if_planes_neighbs(self, planes_masks, plane_params, plane_mask, plane_params2, plane_mask2, plot=True):
        '''
        Check if planes are neighbours

        :param planes_masks: planes mask
        :param plane_mask:
        :param plane_mask2:
        :param plot:
        :return:
        '''
        def calc_grad_mask(mask):
            grad_y = np.abs(mask[1:] - mask[:-1])
            grad_x = np.abs(mask[:, 1:] - mask[:, :-1])

            grad_mask = np.zeros_like(mask)
            grad_mask[1:] = grad_y
            grad_mask[:,1:] = np.maximum(grad_mask[:,1:], grad_x)


            return grad_mask

        depth = self.depth_filled
        depth_mask = self.depth_filled_mask
        K_dot_coord_2d = self.K_dot_coord_2d

        # Two planes are neighbours if there are no other planes between them
        other_masks_joined = np.sum(np.array(planes_masks), axis=0) - plane_mask - plane_mask2

        plane_mask2_grad_mask = calc_grad_mask(plane_mask2).astype(np.uint8)
        plane_mask2_dist_transf = cv2.distanceTransform(1 - plane_mask2_grad_mask, cv2.DIST_L2, 3)
        plane_mask2_dist_transf[np.logical_not(plane_mask.astype(np.bool))] = 1e4
        plane_mask_point = np.argmin(plane_mask2_dist_transf)
        plane_mask_point = np.unravel_index(plane_mask_point, plane_mask.shape)

        plane_mask1_grad_mask = calc_grad_mask(plane_mask).astype(np.uint8)
        plane_mask1_dist_transf = cv2.distanceTransform(1 - plane_mask1_grad_mask, cv2.DIST_L2, 3)
        plane_mask1_dist_transf[np.logical_not(plane_mask2.astype(np.bool))] = 1e4
        plane_mask2_point = np.argmin(plane_mask1_dist_transf)
        plane_mask2_point = np.unravel_index(plane_mask2_point, plane_mask2.shape)

        line12_mask = np.zeros_like(other_masks_joined)
        cv2.line(line12_mask, (plane_mask_point[1], plane_mask_point[0]),
                 (plane_mask2_point[1], plane_mask2_point[0]), 1., thickness=5)

        n = plane_params[:3]
        d = plane_params[3]
        z1 = line12_mask * (-d * K_dot_coord_2d[:, :, 2] / (K_dot_coord_2d.dot(n) + 1e-6))
        planes_dist = (line12_mask * depth_mask * np.maximum(depth - z1, 0)).sum() / np.maximum((depth_mask * line12_mask).sum(), 1.)

        return np.all(np.less(other_masks_joined + line12_mask, 2)) and planes_dist < merge_offset_threshold

    def intersect_3_planes(self, layout_planes, i, j, k, floor_plane_params):
        '''
        Intersect 3 planes

        :param layout_planes:
        :param i: index of plane 1
        :param j: index of plane 2
        :param k: index of plane 3
        :return:
        '''
        # Get helper attributes
        h, w = self.h, self.w
        K = self.K

        j_is_frustum = layout_planes[j].type == 0
        k_is_frustum = layout_planes[k].type == 0

        plane1 = layout_planes[i].plane
        plane2 = layout_planes[j].plane
        plane3 = layout_planes[k].plane

        plane1_type = layout_planes[i].type
        plane2_type = layout_planes[j].type
        plane3_type = layout_planes[k].type

        plane1_normal, d1 = plane1[:3], plane1[3]
        plane2_normal, d2 = plane2[:3], plane2[3]
        plane3_normal, d3 = plane3[:3], plane3[3]

        denom = np.dot(plane1_normal, np.cross(plane2_normal, plane3_normal))

        # None of the planes should be parallel
        if not((j_is_frustum or np.abs(np.cross(plane1_normal, plane2_normal)).mean() > par_thresh) and \
                (k_is_frustum or np.abs(np.cross(plane1_normal, plane3_normal)).mean() > par_thresh) and \
                ((j_is_frustum and k_is_frustum) or np.abs(np.cross(plane2_normal, plane3_normal)).mean() > par_thresh)):
            return None

        nom = -d1 * np.cross(plane2_normal, plane3_normal) - \
              d2 * np.cross(plane3_normal, plane1_normal) - \
              d3 * np.cross(plane1_normal, plane2_normal)
        intersection_3d = nom / denom

        # Check if the intersection is under the floor plane (Check explicitly if the intersection includes floor
        # due to frustum planes inconsistencies]
        if not floor_type in [plane1_type, plane2_type, plane3_type] and floor_plane_params is not None:
            floor_plane_rel = np.dot(floor_plane_params[:3], intersection_3d) + floor_plane_params[3]
            if floor_plane_rel < 0:
                return None

        intersection_image_plane = np.dot(K, np.reshape(intersection_3d, (3, 1)))

        intersection_norm = (intersection_image_plane / (intersection_image_plane[2] + 1e-6))
        x = intersection_norm[0][0]
        y = intersection_norm[1][0]
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        x = np.clip(x, a_min=-1e6, a_max=1e6)
        y = np.clip(y, a_min=-1e6, a_max=1e6)

        x = int(np.round(x))
        y = int(np.round(y))

        if (intersection_3d[2] > max_depth_thresh or intersection_3d[2] < min_depth_thresh):
            # print("Intersection point is far away")
            return None

        if not((0 <= x < w) and (0 <= y < h)):
            return None

        line12_u = np.cross(plane1_normal, plane2_normal)
        line12_u = line12_u / np.linalg.norm(line12_u, ord=2)

        line13_u = np.cross(plane1_normal, plane3_normal)
        line13_u = line13_u / np.linalg.norm(line13_u, ord=2)

        line23_u = np.cross(plane2_normal, plane3_normal)
        line23_u = line23_u / np.linalg.norm(line23_u, ord=2)

        intersection_dict = {}
        intersection_dict['inter'] = intersection_3d
        intersection_dict['inter_image'] = [x, y]
        intersection_dict['inter_edges'] = [line12_u, line13_u, line23_u]
        intersection_dict['inter_planes'] = [i, j, k]
        intersection_dict['inter_planes_frustum'] = [False,
                                                     j > len(layout_planes) - 5,
                                                     k > len(layout_planes) - 5]
        intersection_dict['inter_frustum'] = k > len(layout_planes) - 5

        return intersection_dict

    def intersect_planes(self, layout_planes):
        '''
        Intersect layout planes

        :param layout_planes:
        :return: list of xs, ys, intersection dictionary
        '''

        xs = []
        ys = []
        intersection_dict_list = []
        if plot_intermediate_results:
            plt.figure()

        # Planes below floor will be ignored
        floor_plane_params = self.get_floor_planes(layout_planes)

        # Intersect plane triplets
        plot_counter = 1
        for i, layout_plane1 in enumerate(layout_planes):
            if layout_plane1.type == 0:
                continue
            if layout_plane1.type == -1:
                continue
            for j, layout_plane2 in enumerate(layout_planes):
                if j > len(layout_planes) - 1:
                    continue
                if j <= i:
                    continue
                for k, layout_plane3 in enumerate(layout_planes):
                    if k <= i or k <= j:
                        continue

                    intersection_dict = self.intersect_3_planes(layout_planes, i, j, k, floor_plane_params)
                    if intersection_dict is None:
                        continue

                    x, y = intersection_dict['inter_image']
                    xs.append(x)
                    ys.append(y)
                    intersection_dict_list.append(intersection_dict)

                    if plot_intermediate_results:
                        plane_mask1 = layout_plane1.mask
                        plane_mask2 = layout_plane2.mask
                        plane_mask3 = layout_plane3.mask

                        im = np.copy(self.img) * 255
                        ch_r = (plane_mask1 * 255)[:, :, None]
                        ch_g = (plane_mask2 * 255)[:, :, None]
                        ch_b = (plane_mask3 * 255)[:, :, None]
                        image_plus_3surf = np.minimum(0.3 * np.concatenate((ch_r, ch_g, ch_b), axis=2) + 0.7 * im,
                                                      255).astype(np.uint8)

                        if plot_counter < 50:
                            plt.subplot(5,10,plot_counter)
                            plt.title(str(i) + "," + str(j) + "," + str(k))
                            plot_counter += 1
                            implot = plt.imshow(
                                (plane_mask1[:, :, None] +
                                 plane_mask2[:, :, None] +
                                 plane_mask3[:,:, None] * im).astype(np.uint8))
                            plt.scatter(x=[x], y=[y], c='r', s=40)
                            implot = plt.imshow(image_plus_3surf)
        if plot_intermediate_results:
            plt.show()
        print("Number of intersections found: " + str(len(intersection_dict_list)))
        self.rl_intersection_dict_list = intersection_dict_list
        return (xs, ys), intersection_dict_list

    def find_candidate_edges(self, intersection_dict_list, layout_planes):
        '''
        Find candidate edges

        :param intersection_dict_list: List of intersection dict
        :param layout_planes: List of layout planes
        :return: list of candidate edges, dict of candidate edges index by plane
        '''
        candidate_edges = []
        planes_n = len(layout_planes)

        planes_candidate_edges = {}
        for plane_index in range(planes_n):
            planes_candidate_edges[plane_index] = []

        for i, intersection_dict_i in enumerate(intersection_dict_list):
            intersection_dict_i_planes = intersection_dict_i['inter_planes']
            for j, intersection_dict_j in enumerate(intersection_dict_list):
                if j <= i:
                    continue
                # Consider only intersections that were created by same plane
                intersection_dict_j_planes = intersection_dict_j['inter_planes']

                # Check if two intersections share two planes
                check_i_j_plane0 = intersection_dict_i_planes[0] in intersection_dict_j_planes[:3]
                check_i_j_plane1 = intersection_dict_i_planes[1] in intersection_dict_j_planes[:3]
                check_i_j_plane2 = intersection_dict_i_planes[2] in intersection_dict_j_planes[:3]
                if (check_i_j_plane0 and check_i_j_plane1) or (check_i_j_plane0 and check_i_j_plane2) or \
                        (check_i_j_plane1 and check_i_j_plane2):

                    # If two planes are common for two intersections, there are some lines that coincide, hence edge
                    candidate_edges.append([i, j])

                    if check_i_j_plane0:
                        planes_candidate_edges[intersection_dict_i_planes[0]].append([i, j])
                    if check_i_j_plane1:
                        planes_candidate_edges[intersection_dict_i_planes[1]].append([i, j])
                    if check_i_j_plane2:
                        planes_candidate_edges[intersection_dict_i_planes[2]].append([i, j])

        return candidate_edges, planes_candidate_edges

    def get_poly_mask(self, poly):
        '''
        Get polygon mask

        :param poly: list of polygon vertices
        :return: polygon mask (H, W)
        '''
        h, w = self.h, self.w

        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))

        polygon_mask = np.zeros((h, w))
        cv2.fillPoly(polygon_mask, [pts], 1.)

        return polygon_mask

    def calc_new_plane_masks(self, planes_polygons):
        '''
        Calculate polygon masks for each of the given polygons

        :param planes_polygons: List of polygons
        :return: List of polygons masks
        '''
        h, w = self.h, self.w
        new_planes_masks = []
        for plane_ind, plane_polygon in enumerate(planes_polygons):
            polygon_mask = self.get_poly_mask(plane_polygon)

            new_planes_masks.append(polygon_mask)
        return new_planes_masks

    def calc_layout_edges(self, layout_components, thickness=1):
        '''
        Calculate layout edges from the given components

        :param layout_components: list of layout components
        :param thickness: edge thickness
        :return: Layout edge mask (H,W)
        '''
        h, w = self.h, self.w
        layout_edge_mask = np.zeros((h, w))

        for plane_ind, layout_comp in enumerate(layout_components):
            comp_type = layout_comp.type
            if comp_type in floor_wall_ceil:
                pts = np.array(layout_comp.poly, np.int32)
                pts = pts.reshape((-1, 1, 2))

                cv2.polylines(layout_edge_mask, [pts], False, 1., thickness=thickness)

        return layout_edge_mask

    def solve_with_pulp(self, layout_candidates):
        '''
        Layout solver, PULP implementation

        :param layout_candidates: list of layout candidate components
        :return: list of optimal layout components, Solver status: 'Optimal', 'Infeasible'
        '''
        print("-------------Solver------------------")
        print("Solving binary discrete optimization....")
        cand_incompatible_mat, cand_neighbour_mat = self.getCandidateCompatibilityMatrices(layout_candidates)

        # Instantiate our problem class
        model = pulp.LpProblem("Layout_solver", pulp.LpMinimize)

        polys_vars = pulp.LpVariable.dicts("plane_poly_bool",
                                           ((i) for i in range(len(layout_candidates))),
                                           cat=pulp.LpBinary)

        # Objective Function
        cost_function = [layout_candidates[i].cost * polys_vars[i]
                         for i in range(len(layout_candidates))]

        model += pulp.lpSum(cost_function)

        # Constraints

        # One polygon per plane
        print("Adding Constraints: One polygon per plane constraint...")
        layout_planes = [layout_candidate.plane for layout_candidate in layout_candidates]
        layout_planes = np.unique(np.array(layout_planes), axis=0)

        for plane in layout_planes:
            one_poly_per_plane_constr = [polys_vars[i] for i in range(len(layout_candidates)) if np.all(layout_candidates[i].plane == plane)]
            model += pulp.lpSum(one_poly_per_plane_constr) == 1

        # No polygons intersect
        print("Adding Constraints: Compatible components only...")
        for i in range(len(layout_candidates)):
            for j in range(i + 1, len(layout_candidates)):
                if cand_incompatible_mat[i, j]:
                    incomp_constr = polys_vars[i] + polys_vars[j] <= 1
                    model.__iadd__(incomp_constr)

        print("Adding Constraint: Sum of polygon areas should cover 100% of the image...")
        # sum of polygon areas should cover 100% of the image
        # Area seems to be not exactly equal to the number of pixels in the image (probably a rounding error in shapely)
        shapely_poly = Polygon([(0., 0.), (self.w - 1., 0.), (self.w - 1., self.h - 1.), (0., self.h - 1.), (0., 0.)])
        image_poly_area = shapely_poly.area
        areas_sums = [layout_candidates[i].area * polys_vars[i] for i in range(len(layout_candidates))]
        area_thresh = 1. #0.999
        model += pulp.lpSum(areas_sums) >= image_poly_area * area_thresh

        print("Number of polygon variables: ", len(polys_vars.keys()))

        # Solve the problem
        print("Solving...")
        model.solve()
        solver_status = pulp.LpStatus[model.status]
        print("Solver status: ", pulp.LpStatus[model.status])
        print("Cost: ", pulp.lpSum(cost_function).value())

        solution_components = []
        for var in polys_vars:
            var_value = polys_vars[var].varValue
            if var_value != 0:
                solution_comp = layout_candidates[var]
                solution_components.append(solution_comp)

        if plot_intermediate_results:
            plt.figure()
            num_plots = len(solution_components)
            for i, solution_comp in enumerate(solution_components):
                img_overlay = np.clip(0.5 * self.img + 0.5 * solution_comp.mask[:, :, None], a_min=0, a_max=255)
                # img_cpy = copy.deepcopy(self.img)
                pts = np.array(solution_comp.poly, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img_overlay, [pts], True, (0, 255, 0), thickness=10)

                plt.subplot(1, num_plots, i + 1)
                plt.imshow(img_overlay)
            plt.show()

        return solution_components, solver_status

    def getCandidateCompatibilityMatrices(self, candidates):
        '''
        Get Compatibility matrices

        :param candidates: List of candidate layout components
        :return: Incompability matrix, Neighbourhood matrix
        '''
        cand_incompatible_mat = np.eye(len(candidates))
        cand_neighbour_mat = np.zeros((len(candidates), len(candidates)))

        for i in range(len(candidates)):
            print("Compatibility Candidate %d of %d \r" % (i + 1, len(candidates)), end="", flush=True)
            for j in range(i, len(candidates)):
                    # Get first candidate
                    poly1 = candidates[i].poly

                    poly1_edges = [np.concatenate([vtx, poly1[vtx_ind + 1]], axis=0) for vtx_ind, vtx in enumerate(poly1[:-1])]
                    poly1_edges = np.array(poly1_edges)

                    plane1 = candidates[i].plane

                    # Get second candidate
                    poly2 = candidates[j].poly
                    poly2_edges = [np.concatenate([vtx, poly2[vtx_ind + 1]], axis=0) for vtx_ind, vtx in enumerate(poly2[:-1])]
                    poly2_edges = np.array(poly2_edges)
                    poly2_edges_rev = [np.concatenate([poly2[vtx_ind + 1], vtx], axis=0) for vtx_ind, vtx in enumerate(poly2[:-1])]
                    poly2_edges_rev = np.array(poly2_edges_rev)

                    plane2 = candidates[j].plane

                    same_plane = np.all(plane1 == plane2)
                    if same_plane:
                        continue

                    # Check if poly1 and poly2 share an edge
                    poly1_diff_poly2 = np.any(np.all(np.abs(poly1_edges[:, None, :] - poly2_edges).reshape(-1,4) < 1e-2, axis=1))
                    poly1_diff_poly2_rev = np.any(np.all(np.abs(poly1_edges[:, None, :] - poly2_edges_rev).reshape(-1,4) < 1e-2, axis=1))

                    poly1_poly2_share_an_edge = poly1_diff_poly2 or poly1_diff_poly2_rev

                    incompatible_polys = False
                    if len(poly1) >= 3 and len(poly2) >= 3:
                        poly1_sh = Polygon(
                            [(float(vtx[0]), float(vtx[1])) for vtx in poly1])
                        poly2_sh = Polygon(
                            [(float(vtx[0]), float(vtx[1])) for vtx in poly2])

                        poly1_sh_buff = poly1_sh.buffer(-5)
                        poly1_sh = Polygon(poly1_sh_buff.exterior)

                        poly2_sh_buff = poly2_sh.buffer(-5)
                        poly2_sh = Polygon(poly2_sh_buff.exterior)

                        # TODO Why can there be an exception?
                        try:
                            incompatible_polys = poly1_sh.intersects(poly2_sh) or \
                                                 poly1_sh.contains(poly2_sh) or poly2_sh.contains(poly1_sh)
                        except:
                            print("WARNING: Exception reached in getCandidateCompatibilityMatrices")
                            incompatible_polys = True

                    if incompatible_polys:
                            cand_incompatible_mat[i, j] = 1
                            cand_incompatible_mat[j, i] = 1
                    elif (poly1_poly2_share_an_edge or len(poly1) < 3 or len(poly2) < 3):
                        cand_neighbour_mat[i, j] = 1
                        cand_neighbour_mat[j, i] = 1

        print('Calculated candidate interesection matrix of size %dx%d' % (
        len(candidates), len(candidates)))
        print("-------------------------------")
        return cand_incompatible_mat, cand_neighbour_mat

    def find_best_polygons(self, xys, intersection_dict_list, planes_candidate_edges, layout_planes):
        '''
        Find the optimal layout given a set of corners and edges and layout planes

        :param xys: XY locations of candidate corners (planes intersections)
        :param intersection_dict_list:  Intersection dict with details on each of the planes interesctions
        :param planes_candidate_edges: List of candidate edges
        :param layout_planes: List of layout planes
        :return: list of optimal layout components, Solver status: 'Optimal', 'Infeasible'
        '''
        def find_candidate_plane_polygons(xys_list, plane_edges_list, plane_ind):
            '''
            Parse polygons for given plane

            :param xys_list:
            :param plane_edges_list:
            :param plane_ind:
            :return:
            '''
            start_list = [(edge_ind, start_edge, plane_edges_list, xys_list, intersection_dict_list, plane_ind) for
                          edge_ind, start_edge in enumerate(plane_edges_list)]

            init_plane_polygons_list = []
            for start_edge in start_list:
                start_edge_polys = start_search_edge(start_edge)
                if start_edge_polys is not None:
                    init_plane_polygons_list += [start_edge_polys]

            valid_plane_polygons_list = []
            for pol in init_plane_polygons_list:
                if pol is not None:
                    valid_plane_polygons_list += pol
            init_plane_polygons_list = valid_plane_polygons_list

            valid_plane_polygons_list = []
            for cycle in init_plane_polygons_list:
                # cycle += [cycle[0]]
                cp_coord = [[xs[pol_vtx], ys[pol_vtx]] for pol_vtx in cycle]
                cycle_poly = Polygon([(float(vtx[0]), float(vtx[1])) for vtx in cp_coord])

                cycle_poly_sh_buff = cycle_poly.buffer(-5)
                if isinstance(cycle_poly_sh_buff, MultiPolygon):
                    continue
                cycle_poly = Polygon(cycle_poly_sh_buff.exterior)

                if cycle_poly.is_empty:
                    continue
                if not cycle_poly.is_simple:
                    continue
                if not cycle_poly.is_valid:
                    continue
                if cycle_poly.area < min_plane_size:
                    continue
                valid_plane_polygons_list.append(cycle)

            return valid_plane_polygons_list

        def get_candidates():
            '''
            Get layout candidates including their precomputed costs

            :return:
            '''
            joint_plane_mask = np.zeros((h, w))
            for plane_ind, layout_plane in enumerate(layout_planes):
                if layout_plane.type == -1:
                    continue
                joint_plane_mask += layout_plane.mask
            joint_plane_mask = np.minimum(joint_plane_mask, 1.)

            candidates = []
            for plane_ind, layout_plane in enumerate(layout_planes):
                plane_mask = layout_plane.mask
                plane_type = layout_plane.type

                other_planes_masks = []
                for other_plane_ind, other_layout_plane in enumerate(layout_planes):
                    other_plane_mask = other_layout_plane.mask
                    other_plane_type = other_layout_plane.type

                    if other_plane_type != -1 and other_plane_ind != plane_ind:
                        other_planes_masks.append(other_plane_mask)
                if plane_type == -1:
                    continue

                plane_candidate_edges = planes_candidate_edges[plane_ind]
                tmp_candidate_plane_polygons = find_candidate_plane_polygons(xys_list, plane_candidate_edges, plane_ind)

                tmp_candidate_plane_polygons_coord = []
                for plane_poly in tmp_candidate_plane_polygons:
                    cp_coord = [[xs[pol_vtx], ys[pol_vtx]] for pol_vtx in plane_poly]
                    tmp_candidate_plane_polygons_coord.append(cp_coord)

                tmp_plane_poly_masks = self.calc_new_plane_masks(tmp_candidate_plane_polygons_coord)

                # Initialize plane candidate list with an empty polygon.
                empty_canditate = CandidateLayoutComp(plane=layout_plane.plane, mask=layout_plane.mask,
                                                      type=layout_plane.type,
                                                      poly=[[0, 0], [0, 0]], poly_mask=np.zeros((h, w)),
                                                      cost=empty_poly_cost, area=0)
                candidates.append(empty_canditate)
                for pol_ind, plane_poly_mask in enumerate(tmp_plane_poly_masks):

                    polygon_mask = joint_plane_mask * plane_poly_mask
                    poly_plane_sum = polygon_mask + plane_mask

                    # 2D Cost
                    # Good IOU
                    inter = np.equal(poly_plane_sum, 2.).astype(np.float32)
                    union = np.greater(poly_plane_sum, 0.).astype(np.float32)
                    poly_iou_score = np.sum(inter) / max(np.sum(union), 1.)

                    # Remove candidates that do not overlap with their component plane mask (safe speedup)
                    # Only when poly_mask.sum() > 0
                    # if poly_iou_score == 0:
                    #     continue

                    # 2D Cost
                    # Bad IOU
                    other_planes_iou_score = 0.
                    if len(other_planes_masks) > 0:
                        all_other_poly_plane_sum = np.zeros_like(poly_plane_sum)
                        for other_planes_ind, other_plane_mask in enumerate(other_planes_masks):
                            if layout_planes[other_planes_ind].type == -1:
                                continue
                            other_poly_plane_sum = polygon_mask.astype(np.float32) + other_plane_mask.astype(np.float32)
                            inter = np.equal(other_poly_plane_sum, 2.).astype(np.float32)
                            all_other_poly_plane_sum = np.maximum(all_other_poly_plane_sum, inter)
                            union = np.greater(other_poly_plane_sum, 0.).astype(np.float32)
                            other_planes_iou_score += np.sum(inter) / max(np.sum(union), 1.)
                        other_planes_iou_score /= len(other_planes_masks)

                    # 3D Cost
                    plane_params = layout_planes[plane_ind].plane
                    d = plane_params[3]
                    n = plane_params[:3]
                    z1 = plane_poly_mask * (-d * K_dot_coord_2d[:, :, 2] / (K_dot_coord_2d.dot(n) + 1e-6))

                    depth_discr = np.maximum(depth - z1, 0.)
                    depth_discr = depth_mask * plane_poly_mask * depth_discr
                    depth_discr_cost = np.sum(depth_discr) / max(np.sum(depth_mask * plane_poly_mask), 1)


                    poly = tmp_candidate_plane_polygons_coord[pol_ind]
                    poly_cost = depth_discr_w * (depth_discr_cost) + poly_iou_w * (
                                1. - poly_iou_score) + other_poly_iou_w * other_planes_iou_score

                    shapely_poly = Polygon([(float(vtx[0]), float(vtx[1])) for vtx in poly])
                    poly_area = shapely_poly.area

                    canditate = CandidateLayoutComp(plane=layout_plane.plane, mask=layout_plane.mask,
                                                    type=layout_plane.type,
                                                    poly=poly, poly_mask=plane_poly_mask, cost=poly_cost,
                                                    area=poly_area)

                    candidates.append(canditate)
            return candidates

        # Get helper attributes
        h, w = self.h, self.w
        K_dot_coord_2d = self.K_dot_coord_2d
        depth = self.depth
        depth_mask = self.depth_mask

        xs = xys[0]
        ys = xys[1]
        xys_list = list(zip(xs, ys))

        candidates = get_candidates()
        solution_components, solver_status = self.solve_with_pulp(candidates)
        return solution_components, solver_status

    def generate_depth_map_from_solution(self, layout_components):
        '''
        Generate depth map from a given solution

        :param layout_components: List of layout components
        :return: Layout depth map (H, W)
        '''

        # Get helper attributes
        K_dot_coord_2d = self.K_dot_coord_2d

        z_final = 0.
        for i, layout_plane in enumerate(layout_components):
            poly, plane_params = layout_plane.poly, layout_plane.plane

            poly_mask = self.get_poly_mask(poly)

            plane1_normal = plane_params[:3]
            d1 = plane_params[3]

            # # Plot for debugging
            z1 = poly_mask * (-d1 * K_dot_coord_2d[:, :, 2] / (K_dot_coord_2d.dot(plane1_normal) + 1e-6))
            # z_final += z1
            z_final = np.maximum(z_final, z1)

        return z_final

    def get_refinement_line(self, plane_ind,
                            init_solution_components,
                            depth_diff,
                            depth_mask):
        '''
        Get refinement line

        :param plane_ind: plane index
        :param init_solution_components: List of layout components
        :param depth_diff: Depth diff between layout and depth map (H, W)
        :param depth_mask: Mask of valid depth values (H, W)
        :return: Line Vertex1, Line Vertex2
        '''
        def fit_line_ransac(X, y):
            # # Robustly fit linear model with RANSAC algorithm
            ones = np.ones_like(X)
            xy_ones = np.concatenate((X,y,ones), axis=1)
            (m_fit, b_fit), inliers, _ = fit_line_RANSAC(xy_ones, inlier_thresh=ransac_line_thresh)
            if m_fit is None:
                return None, None, None, None, None

            xy1_fit = np.array([xy_ones[inliers[0]][0], xy_ones[inliers[0]][1]])
            xy2_fit = np.array([xy_ones[inliers[-1]][0], xy_ones[inliers[-1]][1]])
            v_fit = xy2_fit - xy1_fit
            v_perp_fit = np.array([-v_fit[1], v_fit[0]])

            ransac_line = (m_fit, b_fit)
            return ransac_line, m_fit, b_fit, xy1_fit, xy2_fit


        # Mask out parts around polygon edges
        plane_poly_mask = init_solution_components[plane_ind].poly_mask
        kernel = np.ones((20, 20), np.uint8)
        boundary_mask = np.zeros_like(plane_poly_mask)
        cv2.rectangle(boundary_mask, (0, 0), (boundary_mask.shape[1] - 1, boundary_mask.shape[0] - 1), 1.,
                      thickness=10)
        boundary_mask = 1. - boundary_mask
        depth_eroded_mask = cv2.erode((boundary_mask * plane_poly_mask * depth_mask).astype(np.uint8), kernel,
                                      iterations=1)

        clutter_mask = np.less(depth_diff, plane_depth_alignment_thresh) # small value to deal with measurement noise
        clutter_mask = cv2.erode((clutter_mask).astype(np.uint8), kernel,
                                      iterations=1)

        discrepancy_change_map = depth_eroded_mask * clutter_mask * np.abs(
            calc_depth_grad(np.clip(depth_diff, a_min=-10, a_max=0.)))
        max_disc = np.max(discrepancy_change_map)

        aligned_depth_mask = np.logical_and(depth_diff > -plane_depth_alignment_thresh, depth_diff < plane_depth_alignment_thresh)

        arg_disc = np.argwhere(aligned_depth_mask * discrepancy_change_map > -plane_depth_inc_thresh)
        xys = arg_disc

        xs = xys[:, 1][:, None]
        ys = xys[:, 0][:, None]

        _, m_fit, b_fit, xy1_fit, xy2_fit = fit_line_ransac(xs, ys)
        if m_fit is None:
            return None, None, None
        line_xy1 = np.array([(xy1_fit[1] - b_fit) / m_fit, xy1_fit[1]])
        line_xy2 = np.array([(xy2_fit[1] - b_fit) / m_fit, xy2_fit[1]])

        if plot_intermediate_results:
            plt.figure()
            plt.subplot(121)
            plt.imshow(plane_poly_mask)
            plt.subplot(122)
            discr_image = (-1 * depth_eroded_mask) * depth_diff
            discr_image[np.equal(discr_image, 0)] = -1
            plt.imshow(discr_image)

            plt.scatter(xs[:, 0], ys[:, 0], c="tab:orange", s=120)

            plt.plot([line_xy1[0], line_xy2[0]], [line_xy1[1], line_xy2[1]], color='r', linewidth=4)
            plt.show()

        return line_xy1, line_xy2


    def refine_layout_polygons(self, init_layout_planes, init_solution_components):
        '''
        Refine layout estimate

        :param init_layout_planes: List of layout planes
        :param init_solution_components: List of layout components
        :return: List of Refined layout components
        '''

        # Get helper attributes
        h, w = self.h, self.w
        K_dot_coord_2d = self.K_dot_coord_2d
        K_inv = self.K_inv
        depth = self.depth
        depth_mask = self.depth_mask
        dilation_kernel = np.ones((3, 3), np.uint8)
        depth_mask = cv2.erode(depth_mask.astype(np.uint8), dilation_kernel, iterations=1)
        segmentation = self.segmentation

        depth_filled = self.depth_filled
        depth_filled_mask = self.depth_filled_mask

        # Iterate through layout planes and find discrepancies
        found_new_plane = False
        support_layout_planes = []
        for plane_ind in range(len(init_layout_planes)):

            plane_params = init_layout_planes[plane_ind].plane
            plane_mask = init_layout_planes[plane_ind].mask
            plane_type = init_layout_planes[plane_ind].type

            if plane_type != wall_type:
                continue

            plane_poly_mask = init_solution_components[plane_ind].poly_mask

            # Compare Depth from plane and sensor depth

            # Get depth from plane
            n1 = plane_params
            plane1_normal = n1[:3]
            d1 = n1[3]
            z1 = plane_poly_mask * (-d1 * K_dot_coord_2d[:, :, 2] / (K_dot_coord_2d.dot(plane1_normal) + 1e-6))

            # Check for inconsistency of the plane with the depth map
            depth_diff = z1 - depth_filled
            depth_diff_mask = np.less(depth_diff, plane_depth_inc_thresh).astype(np.float32)

            valid_depth_diff_mask = depth_mask * plane_poly_mask * depth_diff_mask
            if np.sum(valid_depth_diff_mask) > min_plane_size:
                found_new_plane = True
                print("Detected Bad discrepancy for plane " + str(plane_ind))
                layout_comp_seg_mask = np.isin(segmentation, valid_layout_ind_dict[plane_type])
                depth_diff_seg_mask = layout_comp_seg_mask * valid_depth_diff_mask

                depth_diff_seg_mask_mean = np.sum(depth_diff_seg_mask) / np.sum(valid_depth_diff_mask)

                # Get refinement line based on inconsitency
                line_xy1, line_xy2 = self.get_refinement_line(plane_ind,
                                                              init_solution_components,
                                                              depth_diff, depth_filled_mask)

                if line_xy1 is None:
                    continue

                p1 = np.array([line_xy1[0], line_xy1[1], 1])
                # p1_d = z1[int(p1[1]), int(p1[0])]
                p1_ones = np.array([[p1[0], p1[1], 1.]])
                K_inv_p1 = K_inv.dot(p1_ones.T)

                p1_d = (-d1 * K_inv_p1[2, :] / (K_inv_p1[:,0].dot(plane1_normal) + 1e-6))
                K_p1 = p1_d * K_inv.dot(p1)

                p2 = np.array([line_xy2[0], line_xy2[1], 1])
                # p2_d = z1[int(p2[1]), int(p2[0])]
                p2_ones = np.array([[p2[0], p2[1], 1.]])
                K_inv_p2 = K_inv.dot(p2_ones.T)
                p2_d = (-d1 * K_inv_p2[2, :] / (K_inv_p2[:,0].dot(plane1_normal) + 1e-6))
                K_p2 = p2_d * K_inv.dot(p2)

                K_p3 = np.array([0, 0, 0])

                v1 = K_p2 - K_p1
                v2 = K_p3 - K_p1
                n = np.cross(v1, v2)
                d = np.array([0.])

                plane_params = np.concatenate((n, d), axis=0)

                support_layout_plane = LayoutPlane(
                    plane=plane_params, mask=np.zeros((h, w)), type=-1)
                support_layout_planes.append(support_layout_plane)

        if found_new_plane:
            new_layout_planes = init_layout_planes + support_layout_planes

            # print("---------Intersect Valid Planes----------")
            ((xs, ys), intersection_dict_list) = self.intersect_planes(new_layout_planes)


            # print("---------Find Candidate Edges----------")
            candidate_edges, planes_candidate_edges = self.find_candidate_edges(intersection_dict_list,
                                                                                new_layout_planes)

            if plot_intermediate_results:
                plt.figure()
                # plt.title("Predicted planes corners")
                implot = plt.imshow(self.img)

                for (l1, l2) in candidate_edges:
                    plt.plot([xs[l1], xs[l2]], [ys[l1], ys[l2]], linewidth=2)
                plt.scatter(x=xs, y=ys, c='r', s=240)
                plt.axis('off')
                # plt.savefig("/home/sinisa/Sinisa_Projects/Robotino/corridor_localization/planercnn/reports/Vincent_ICCV/ex_20/6_1_intersections.png",
                #             bbox_inches="tight", pad_inches=0)

                plt.show()

            # Find the refined solution
            solution_components, solver_status = \
                self.find_best_polygons((xs, ys), intersection_dict_list, planes_candidate_edges, new_layout_planes)

            if solver_status == "Infeasible":
                assert False, "Layout estimate seems to be infeasible according to PULP solution"
        else:
            new_layout_planes = init_layout_planes
            solution_components = init_solution_components

        return new_layout_planes, solution_components, found_new_plane

    def refine_layout(self, init_layout_planes, init_solution_components):
        '''
        Refine layout from a given solution

        :param init_layout_planes: list of layout planes
        :param init_solution_components: List of layout components
        :return:
        '''
        print("-----------------Refine Layout---------------")
        init_solution_depth = self.generate_depth_map_from_solution(init_solution_components)

        discr_map = self.depth_mask * np.abs(np.maximum(self.depth - init_solution_depth, 0.))
        # ------------- Adjust ------------------------------

        prev_discr_err = 1e6
        curr_discr_err = np.sum(self.depth_mask * discr_map > discr_threshold) / np.sum(self.depth_mask)

        found_new_plane = True
        refinement_iter = 0

        final_solution_components = init_solution_components
        while curr_discr_err > min_discr_error and curr_discr_err < prev_discr_err and \
                found_new_plane and refinement_iter < refinement_max_iter:
            refinement_iter += 1

            new_layout_planes, solution_components, found_new_plane = self.refine_layout_polygons(init_layout_planes,
                                                                                                  init_solution_components)

            refined_solution_depth = self.generate_depth_map_from_solution(init_solution_components)

            discr_map = self.depth_mask * np.abs(np.maximum(self.depth - refined_solution_depth, 0.))

            tmp_prev_discr_err = curr_discr_err
            curr_discr_err = np.sum(self.depth_mask * discr_map) / np.sum(self.depth_mask)

            if prev_discr_err > curr_discr_err:
                final_solution_components = solution_components
                prev_discr_err = tmp_prev_discr_err

        return final_solution_components

    def infer_layout(self):
        '''
        Infer layout

        :return:
        '''

        img = self.img

        print("----------Validate Planes---------")
        print("Extracting valid planes...")

        # Find valid planes
        layout_planes = self.calc_valid_planes()

        # If floor is not visible, add virtual one
        virt_floor = self.add_virtual_floor_plane(layout_planes)

        if self.use_virtual_floor_plane:
            layout_planes.append(virt_floor)

        frustum_planes = self.calc_frustum_planes()
        layout_planes += frustum_planes

        print("---------Intersect Valid Planes----------")
        # Intersect planes
        ((xs, ys), intersection_dict_list) = self.intersect_planes(layout_planes)
        candidate_edges, planes_candidate_edges = self.find_candidate_edges(intersection_dict_list, layout_planes)
        if plot_intermediate_results:
            plt.figure()

            for (l1, l2) in candidate_edges:
                plt.plot([xs[l1], xs[l2]], [ys[l1], ys[l2]], linewidth=2)
            plt.scatter(x=xs, y=ys, c='r', s=240)
            plt.axis('off')
            plt.show()

        # Find the initial solution
        solution_components, solver_status = \
            self.find_best_polygons((xs, ys), intersection_dict_list, planes_candidate_edges, layout_planes)

        # For now assert infeasible solution. (It can be useful in multi-view ;) )
        if solver_status == "Infeasible":
            assert False, "Layout estimate seems to be infeasible according to PULP solution"

        # Refine layout
        final_solution_components = self.refine_layout(layout_planes, solution_components)

        print("Done!")
        self.layout_components = final_solution_components

    #-------------------------------------------------------------------------------------------------------------------
    # Visualization and prints code

    def visualize_2d_layout_helper(self, overlay_color=True):
        layout_components = self.layout_components

        poly_layout_img = np.zeros_like(self.img)
        poly_layout_img[:, :, 1] = poly_edges_mask = self.calc_layout_edges(layout_components,
                                                                            thickness=10)
        poly_edges_mask = poly_edges_mask.astype(np.bool)

        if overlay_color:
            layout_image = poly_edges_mask[:, :, None] * poly_layout_img + self.img
        else:
            layout_image = np.zeros_like(self.img) + 0.5
            layout_image[poly_edges_mask,0] = 0.
            layout_image[poly_edges_mask,1] = 1.
            layout_image[poly_edges_mask,2] = 0.

        layout_image = np.clip(layout_image, a_min=0, a_max=1)

        return layout_image
    def visualize_layout_2d(self):

        layout_image = self.visualize_2d_layout_helper()

        plt.figure()
        plt.imshow(layout_image)
        plt.show()

    def vis_layout_planes(self, layout_planes):
        normals_image = np.zeros((self.h, self.w, 3))
        planes_filt_segs_image = np.zeros((self.h, self.w, 3))

        for i, layout_plane in enumerate(layout_planes):
            plane_mask = layout_plane.mask
            plane_params = layout_plane.plane

            normals_image[:, :, 0] += plane_mask * (0.5 * (-plane_params[0] + 1)) * 255
            normals_image[:, :, 1] += plane_mask * (0.5 * (-plane_params[1]) + 0.5) * 255
            normals_image[:, :, 2] += plane_mask * (0.5 * (-plane_params[2] + 1)) * 255

            color = np.random.rand(3)
            planes_filt_segs_image[plane_mask.astype(bool), :] = color * 255
            # print(plane_params)

        normals_image = normals_image.astype(np.uint8)


        normals_image_overlay = (0.5 * normals_image + 0.5 * 255 * self.img).astype(np.uint8)
        planes_filt_segs_image_overlay = (0.5 * planes_filt_segs_image + 0.5 * 255 * self.img).astype(np.uint8)

        return planes_filt_segs_image_overlay, normals_image_overlay

    def get_clutter_pc(self, visualize=False):
        # Get helpers
        img = self.img
        K = self.K
        # clutter_mask = self.clutter_mask

        layout_depth = self.generate_depth_map_from_solution(self.layout_components)

        clutter_thresh = 0.04
        layout_mask = \
            np.logical_or(self.layout_seg_mask, np.less((layout_depth - self.depth),
                                                        layout_depth * clutter_thresh)).astype(np.bool)

        clutter_mask = np.logical_and((1. - self.layout_seg_mask).astype(np.bool),
                                      np.greater_equal((layout_depth - self.depth_filled),
                                                       layout_depth * clutter_thresh)).astype(np.bool)

        kernel = np.ones((11, 11), np.uint8)
        clutter_mask = cv2.erode(clutter_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)
        layout_mask = cv2.erode(layout_mask.astype(np.uint8), kernel, iterations=1).astype(np.bool)

        clutter_image = img

        boundary_mask = np.zeros_like((clutter_mask))
        boundary_mask[10:-10, 10:-10] = 1
        o3d_im = o3d.geometry.Image((clutter_image * 255).astype(np.uint8))
        depth_clutter = (boundary_mask[:, :] * clutter_mask * self.depth).astype("float32")

        o3d_depth = o3d.geometry.Image(depth_clutter)
        camera_intr = o3d.camera.PinholeCameraIntrinsic(clutter_image.shape[1], clutter_image.shape[0], K[0, 0],
                                                        K[1, 1], K[0, 2], K[1, 2])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_im, o3d_depth, depth_scale=1., convert_rgb_to_intensity=False,
            depth_trunc=self.depth.max() - 0.01)
        pcd_clutter = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intr)

        return pcd_clutter

    def get_layout_pc(self):
        # Get helpers
        img = self.img
        K = self.K

        layout_components = self.layout_components

        depth_from_planes = self.generate_depth_map_from_solution(layout_components)
        layout_image = self.visualize_2d_layout_helper(overlay_color=False)

        # Layout Point Cloud
        o3d_im_poly = o3d.geometry.Image((layout_image * 255).astype(np.uint8))
        o3d_depth_from_planes_poly = o3d.geometry.Image(depth_from_planes.astype("float32"))
        camera_intr = o3d.camera.PinholeCameraIntrinsic(img.shape[1], img.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        o3d_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_im_poly, o3d_depth_from_planes_poly, depth_scale=1., convert_rgb_to_intensity=False,
            depth_trunc=depth_from_planes.max() - 0.01)
        o3d_pcd_layout = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d_rgbd_image, camera_intr)

        return o3d_pcd_layout

    def visualize_layout_3d(self):
        # Clutter Point Cloud
        pcd_clutter = self.get_clutter_pc()

        # Layout Point Cloud
        pcd_layout = self.get_layout_pc()

        input('Press Enter to visualize the point cloud...')
        o3d.visualization.draw_geometries([pcd_clutter, pcd_layout])
