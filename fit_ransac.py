import numpy as np

from rl_config import *
import itertools

def fit_plane(points):
    '''
    Fit plane to points

    :param points: 3d points (N, 4)
    :return: plane (4,)
    '''
    assert points.shape[0] >= 3  # at least 3 points needed
    _, _, vh = np.linalg.svd(points)
    plane = vh[-1, :]
    return plane

def get_point2plane_dist(points, plane):
    '''
    Get distance of points to the plane

    :param points: 3d points (N, 4)
    :param plane: plane (4,)
    :return: distances (N,)
    '''
    dists = np.abs(points @ plane) / np.sqrt(plane[0] ** 2 + plane[1] ** 2 + plane[2] ** 2)
    return dists


def fit_plane_RANSAC(points, inlier_thresh=0.05, iters=3000):
    '''
    Fit plane using RANSAC

    :param points:  3d points (N, 4)
    :param inlier_thresh: Inlier threshold value
    :param return_outlier_list: If True, the function return the outlier list
    :return:
    '''

    max_inlier_num = -1
    max_inlier_list = None
    best_plane = None

    points_ransac = np.round(points, decimals=2)
    points_ransac = np.unique(points_ransac, axis=0)
    cand_points_ind = np.arange(points_ransac.shape[0])

    points_fit = np.round(points, decimals=2)
    points_fit = np.unique(points_fit, axis=0)

    N = points_ransac.shape[0]
    assert N >= 3

    for i in range(iters):
        chose_id = np.random.choice(cand_points_ind, 3, replace=False)

        chose_points = points_ransac[chose_id, :]

        tmp_plane = fit_plane(chose_points)

        dists = get_point2plane_dist(points_fit, tmp_plane)
        tmp_inlier_list = np.where(dists < inlier_thresh)[0]
        tmp_inliers = points_fit[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list
            best_plane = tmp_plane
            # print('iter %d, %d inliers' % (i, max_inlier_num))

    plane = best_plane

    dists = get_point2plane_dist(points_fit, plane)

    inlier_list = np.where(dists < inlier_thresh)[0]

    outlier_list = np.where(dists >= inlier_thresh)[0]
    return plane, inlier_list, outlier_list

#LINE-----RANSAC--------------------------------------------------------------------------------------------------------

def fit_line(points):
    '''
    Fit line to points

    :param points: 2d points (N, 3)
    :return: line (3,)
    '''
    assert points.shape[0] >= 2  # at least 3 points needed
    _, _, vh = np.linalg.svd(points)
    line = vh[-1, :]
    return line

def get_point2line_dist(points, line):
    '''
    Get distance of points to the line

    :param points: 2d points (N, 3)
    :param line: line (3,)
    :return: distances (N,)
    '''
    dists = np.abs(points @ line) / np.sqrt(line[0] ** 2 + line[1] ** 2 + line[2] ** 2)
    return dists

def fit_line_RANSAC(points, inlier_thresh=0.05):
    '''
    Fit line using RANSAC

    :param points:  2d points (N, 3)
    :param inlier_thresh: Inlier threshold value
    :param return_outlier_list: If True, the function return the outlier list
    :return:
    '''
    max_inlier_num = -1
    max_inlier_list = None
    best_line = None

    stride = max(int(points.shape[0] / (ransac_line_points_n)), 1)
    # stride = 1
    points_ransac = points[::stride, :]

    stride = 1
    points_fit = points[::stride, :]

    N = points_ransac.shape[0]
    assert N >= 2

    chose_ids = itertools.permutations(list(range(N)), 2)

    for chose_id in chose_ids:
        chose_points = points_ransac[chose_id, :]
        tmp_line = fit_line(chose_points)

        tmp_m = - tmp_line[0] / tmp_line[1]
        if np.abs(tmp_m) < 5:
            continue
        dists = get_point2line_dist(points_fit, tmp_line)
        tmp_inlier_list = np.where(dists < inlier_thresh)[0]
        tmp_inliers = points_fit[tmp_inlier_list, :]
        num_inliers = tmp_inliers.shape[0]
        if num_inliers > max_inlier_num:
            max_inlier_num = num_inliers
            max_inlier_list = tmp_inlier_list
            best_line = tmp_line

    if max_inlier_num < 2:
        return (0, 0), [[0,0],[1,1]]
    # final_points = points_fit[max_inlier_list, :]
    # line = fit_line(final_points)
    # fit_variance = np.var(get_point2line_dist(final_points, line))
    # print('RANSAC fit variance: %f' % fit_variance)

    line = best_line
    dists = get_point2line_dist(points, line)

    m = - line[0] / line[1]
    b = - line[2] / line[1]

    inlier_list = np.where(dists < inlier_thresh)[0]

    outlier_list = np.where(dists >= inlier_thresh)[0]
    return (m, b), inlier_list, outlier_list

