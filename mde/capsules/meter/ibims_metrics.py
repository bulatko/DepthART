
"""
Based on a script made by Tobias Koch, tobias.koch@tum.de
Remote Sensing Technology, Technical University of Munich
www.lmf.bgu.tum.de
"""

import numpy as np
import skimage
import scipy
from sklearn.decomposition import PCA
import math
import os


def dbe_metric(edges, pred, max_dist_thr=10.0):
    # edges: 1 x H x W
    # pred: 1 x H x W
    if edges.sum() == 0:
        return (0, 0), (0, 0)

    edges = edges[0].cpu().numpy()
    pred = pred[0].cpu().numpy()

    pred_normalized = (pred - pred.min()) / (pred.max() - pred.min())
    edges_est = skimage.feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=0.1, high_threshold=0.2)

    D_gt = scipy.ndimage.distance_transform_edt(1 - edges)
    D_est = scipy.ndimage.distance_transform_edt(1 - edges_est)

    mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map
    E_fin_est_filt = edges_est * mask_D_gt  # compute the shortest distance for all predicted edges

    if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be found in prediction
        dbe_acc = max_dist_thr
        dbe_com = max_dist_thr
    else:
        dbe_acc = (D_gt * E_fin_est_filt).sum() / E_fin_est_filt.sum()  # accuracy: directed chamfer distance
        dbe_com = np.clip(D_est * edges, a_min=None, a_max=max_dist_thr).sum() / edges.sum()
        # Original metric is in the line below, but it looks like bullshit
        # dbe_com = (D_est * edges).sum() / edges.sum()  # completeness: directed chamfer distance (reversed)

    return (dbe_acc, 1), (dbe_com, 1)


def dbe_acc_metric(edges, pred):
    return dbe_metric(edges, pred)[0]


def dbe_com_metric(edges, pred):
    return dbe_metric(edges, pred)[1]


def compute_planarity_error(gt, pred, calib, plane_mask, plane_params):
    # mask invalid and missing depth values
    pred[pred == 0] = np.nan
    gt[gt == 0] = np.nan

    # number of planes of the current plane type
    nr_planes = plane_params.shape[0]

    # initialize PE errors
    pe_fla = np.empty(0)
    pe_ori = np.empty(0)

    fx_d, fy_d, cx_d, cy_d = calib

    for j in range(nr_planes):  # loop over number of planes
        # only consider depth values for this specific planar mask
        curr_plane_mask = plane_mask.copy()
        curr_plane_mask[curr_plane_mask < (j+1)] = 0
        curr_plane_mask[curr_plane_mask > (j+1)] = 0
        remain_mask = curr_plane_mask.astype(float)
        remain_mask[remain_mask == 0] = np.nan
        remain_mask[np.isnan(remain_mask) == 0] = 1

        # only consider plane masks which are larger than 5% of the image dimension
        if np.nansum(remain_mask) / (640. * 480.) < 0.05:
            flat = np.nan
            orie = np.nan
        else:
            # scale remaining depth map of current plane towards gt depth map
            mean_depth_est = np.nanmedian(pred * remain_mask)
            mean_depth_gt = np.nanmedian(gt * remain_mask)
            est_depth_scaled = pred / (mean_depth_est / mean_depth_gt) * remain_mask

            # project masked and scaled depth values to 3D points
            c, r = np.meshgrid(range(1, gt.shape[1] + 1), range(1, gt.shape[0] + 1))
            tmp_x = ((c - cx_d) * est_depth_scaled / fx_d)
            tmp_y = est_depth_scaled
            tmp_z = (-(r - cy_d) * est_depth_scaled / fy_d)
            X = tmp_x.flatten()
            Y = tmp_y.flatten()
            Z = tmp_z.flatten()
            X = X[~np.isnan(X)]
            Y = Y[~np.isnan(Y)]
            Z = Z[~np.isnan(Z)]
            pointCloud = np.stack((X, Y, Z))

            # fit 3D plane to 3D points (normal, d)
            pca = PCA(n_components=3)
            pca.fit(pointCloud.T)
            normal = -pca.components_[2, :]
            point = np.mean(pointCloud, axis=1)
            d = -np.dot(normal, point)

            # PE_flat: deviation of fitted 3D plane
            flat = np.std(np.dot(pointCloud.T, normal.T) + d) * 100.0

            n_gt = plane_params[j, 4:7]
            if np.dot(normal, n_gt) < 0:
                normal = -normal

            # PE_ori: 3D angle error between ground truth plane and normal vector of fitted plane
            orie = math.atan2(
                np.linalg.norm(np.cross(n_gt, normal)),
                np.dot(n_gt, normal)
            ) * 180. / np.pi

        pe_fla = np.append(pe_fla, flat)  # append errors
        pe_ori = np.append(pe_ori, orie)

    return pe_fla, pe_ori


def compute_all_planarity_errors(
        **kwargs
        # target, predict, mask, camera,
        # mask_table, table_planes,
        # mask_walls, wall_planes,
        # mask_floor, floor_planes
):
    gt, pred, mask, mask_table, mask_walls, mask_floor = [
        kwargs[x][0].cpu().numpy() for x in ["target", "predict", "mask", "mask_table", "mask_walls", "mask_floor"]
    ]  # [480, 640] np.array

    table_planes, floor_planes, wall_planes, camera = [
        kwargs[x].cpu().numpy() for x in ["table_planes", "floor_planes", "wall_planes", "camera"]
    ]
    gt = gt * mask
    pred = pred * mask
    mask_table = mask_table * mask
    mask_walls = mask_walls * mask
    mask_floor = mask_floor * mask

    table_fla, table_ori = compute_planarity_error(gt, pred, camera, mask_table, table_planes)
    walls_fla, walls_ori = compute_planarity_error(gt, pred, camera, mask_walls, wall_planes)
    floor_fla, floor_ori = compute_planarity_error(gt, pred, camera, mask_floor, floor_planes)

    all_fla = np.append(np.append(table_fla, walls_fla), floor_fla)
    all_ori = np.append(np.append(table_ori, walls_ori), floor_ori)

    fla_enum = np.nansum(all_fla)
    fla_denum = np.isfinite(all_fla).sum()

    ori_enum = np.nansum(all_ori)
    ori_denum = np.isfinite(all_ori).sum()

    return (fla_enum, fla_denum), (ori_enum, ori_denum)


def pe_fla_metric(**kwargs):
    return compute_all_planarity_errors(**kwargs)[0]


def pe_ori_metric(**kwargs):
    return compute_all_planarity_errors(**kwargs)[1]
