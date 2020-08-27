import numpy as np

def calc_depth_grad(depth):
    '''
    Calculate depth gradient

    :param depth: depth map
    :return: depth gradient
    '''

    # [1 0 -1] kernel
    depth_grad_y = np.abs(depth[2:] - depth[:-2])
    depth_grad_x = np.abs(depth[:, 2:] - depth[:, :-2])

    depth_grad = np.zeros_like(depth)
    depth_grad[1:-1] = depth_grad_y
    depth_grad[:, 1:-1] = np.maximum(depth_grad[:, 1:-1], depth_grad_x)

    return depth_grad

def get_coord_grid(h, w):
    '''
    Get coordinate grid

    :param h: image height
    :param w: image width
    :return: coordinate grid (3, h * w), coordinate grid (h, w, 3)
    '''

    y_coords, x_coords = np.meshgrid(range(h), range(w), indexing='ij')
    z_coords = np.ones_like(x_coords)
    coordinate_grid_flat = np.array([x_coords, y_coords, z_coords])
    coordinate_grid_flat = np.reshape(coordinate_grid_flat, newshape=(3, -1))
    coordinate_grid_2d = np.concatenate((x_coords[:, :, None], y_coords[:, :, None], z_coords[:, :, None]), axis=2)

    return coordinate_grid_flat, coordinate_grid_2d

def calc_plane_type(plane_mask_flat, plane_seg_flat_sum_classes, val_ind):
    '''
    Calculate plane type

    :param plane_mask_flat: Plane mask
    :param plane_seg_flat_sum_classes: Number of points on plane for each of the segmentation classes
    :param val_ind: valid segmentation indices
    :return: Plane type
    '''

    plane_type = 0
    plane_type_area = 0
    for plane_type_cand in val_ind:
        plane_type_cand_area = np.sum(plane_seg_flat_sum_classes[plane_type_cand]) / np.sum(plane_mask_flat)
        if plane_type_cand_area > plane_type_area:
            plane_type_area = plane_type_cand_area
            plane_type = plane_type_cand

    if plane_type > 0:
        return plane_type
    else:
        return -1


def norm_plane_params(plane_params):
    '''
    Normalize plane parameters

    :param plane_params:
    :return:
    '''
    plane_params = plane_params / np.sqrt(
        plane_params[0] ** 2 + plane_params[1] ** 2 + plane_params[2] ** 2)
    if plane_params.shape[0] == 4 and plane_params[-1] < 0.:
        plane_params *= -1

    return plane_params

