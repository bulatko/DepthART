# -*- coding: utf-8 -*-
"""
Created on Thu Nov 01 19:18:59 2018

@author: Tobias Koch, tobias.koch@tum.de
Remote Sensing Technology, Technical University of Munich
www.lmf.bgu.tum.de


Ibims Evaluation Script v0.9

---------------------- Updates -----------------------
- "Distance-related Assessment" not yet implemented
- "DDE" for different thresholds not yet implemented
- Canny Edge detector for DBE differs from Matlab implementation 
    --> DBE errors do not match to the results in the paper so far...
------------------------------------------------------


This script computes both established global errors and geometric errors
(planarity error, directed depth error, depth boundary error) on the ibims-1
dataset according to the following publication: 
    
Tobias Koch, Lukas Liebel, Friedrich Fraundorfer, Marco Körner:
Evaluation of CNN-based Single-Image Depth Estimation Methods. 
European Conference on Computer Vision (ECCV) Workshops, 2018). 

    
In order to allow an automatic evaluation, please prepare the predictions as follows:
    - individual .mat files for each image in the dataset, named with
      '*image_name*_results.mat' 
    - prediction matrix (float,single, ...) should be named: 'pred_depths'
      (otherwise you need to update the loading command in line 88).
    - Ensure, that the dimension of the predictions matches the dimension of 
      the dataset images (480x640). Please fill missing depth values with NaNs.
       

Please ensure that the dataset ('imagename*'.mat, 'imagelist.txt'), 
your predictions and the python scripts are stored in the same directory.

Run this script...

"""


import numpy as np
from scipy import io
from evaluate_ibims_error_metrics import compute_global_errors,\
                                         compute_directed_depth_error,\
                                         compute_depth_boundary_error,\
                                         compute_planarity_error


## Read imagelist.txt for image names of the ibims-1 dataset
with open('imagelist.txt') as f:
    image_names = f.readlines()
image_names = [x.strip() for x in image_names] 

num_samples = len(image_names) # number of images


# Initialize global and geometric errors ...
rms     = np.zeros(num_samples, np.float32)
log10   = np.zeros(num_samples, np.float32)
abs_rel = np.zeros(num_samples, np.float32)
sq_rel  = np.zeros(num_samples, np.float32)
thr1    = np.zeros(num_samples, np.float32)
thr2    = np.zeros(num_samples, np.float32)
thr3    = np.zeros(num_samples, np.float32)
dde_0   = np.zeros(num_samples, np.float32)
dde_m   = np.zeros(num_samples, np.float32)
dde_p   = np.zeros(num_samples, np.float32)
dbe_acc = np.zeros(num_samples, np.float32)
dbe_com = np.zeros(num_samples, np.float32)
pe_fla = np.empty(0)
pe_ori = np.empty(0)



# Loop over all images in dataset ...
for i in range(num_samples):
    
    # Current image name
    image_name=image_names[i]
    print(image_name)
    
     # Load Predictions (as 'image_name'_results.mat)
    pred = io.loadmat(image_name+'_results')['pred_depths'] 
    pred[np.isnan(pred)] = 0
    pred_org = pred.copy()
    pred_invalid = pred.copy()
    pred_invalid[pred_invalid!=0]=1
    
    # load ground truth data as a .mat file 
    image_data = io.loadmat(image_name)  
    data = image_data['data']
    
    # extract neccessary data
    rgb   = data['rgb'][0][0]   # RGB image
    depth = data['depth'][0][0] # Raw depth map
    edges = data['edges'][0][0] # Ground truth edges
    calib = data['calib'][0][0] # Calibration parameters
    mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
    mask_transp = data['mask_transp'][0][0]    # Mask for transparent pixels
    
    mask_missing = depth.copy() # Mask for further missing depth values in depth map
    mask_missing[mask_missing!=0]=1
    
    mask_valid = mask_transp*mask_invalid*mask_missing*pred_invalid # Combine masks
    
    # Apply 'valid_mask' to raw depth map
    depth_valid = depth*mask_valid 
    
    gt = depth_valid
    gt_vec = gt.flatten()
    
   
    # Apply 'valid_mask' to raw depth map
    pred = pred*mask_valid 
    pred_vec = pred.flatten()

    # Compute errors ... 
    abs_rel[i], sq_rel[i], rms[i], log10[i], thr1[i], thr2[i], thr3[i] = compute_global_errors(gt_vec,pred_vec)
    dde_0[i], dde_m[i], dde_p[i] = compute_directed_depth_error(gt_vec,pred_vec,3.0)
    dbe_acc[i],dbe_com[i] = compute_depth_boundary_error(edges,pred_org)
    
    mask_wall = data['mask_wall'][0][0]*mask_valid
    paras_wall = data['mask_wall_paras'][0][0]
    if paras_wall.size>0:
        pe_fla_wall,pe_ori_wall = compute_planarity_error(gt,pred,paras_wall,mask_wall,calib)
        pe_fla = np.append(pe_fla,pe_fla_wall)
        pe_ori = np.append(pe_ori,pe_ori_wall)

        
    mask_table = data['mask_table'][0][0]*mask_valid
    paras_table = data['mask_table_paras'][0][0]
    if paras_table.size>0:
        pe_fla_table,pe_ori_table = compute_planarity_error(gt,pred,paras_table,mask_table,calib)
        pe_fla = np.append(pe_fla,pe_fla_table)
        pe_ori = np.append(pe_ori,pe_ori_table)

        
    mask_floor = data['mask_floor'][0][0]*mask_valid
    paras_floor = data['mask_floor_paras'][0][0]
    if paras_floor.size>0:
        pe_fla_floor,pe_ori_floor = compute_planarity_error(gt,pred,paras_floor,mask_floor,calib)
        pe_fla = np.append(pe_fla,pe_fla_floor)
        pe_ori = np.append(pe_ori,pe_ori_floor)


# results
print('Results:')
print('rel    = ',  np.nanmean(abs_rel))
print('sq_rel = ',  np.nanmean(sq_rel))
print('rms    = ',  np.nanmean(rms))
print('log10  = ',  np.nanmean(log10))
print('thr1   = ',  np.nanmean(thr1))
print('thr2   = ',  np.nanmean(thr2))
print('thr3   = ',  np.nanmean(thr3))
print('dde_0  = ',  np.nanmean(dde_0))
print('dde_p  = ',  np.nanmean(dde_p))
print('dde_m  = ',  np.nanmean(dde_m))
print('pe_fla = ',  np.nanmean(pe_fla))
print('pe_ori = ',  np.nanmean(pe_ori))
print('dbe_acc = ',  np.nanmean(dbe_acc))
print('dbe_com = ',  np.nanmean(dbe_com))