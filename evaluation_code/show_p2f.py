#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:17:03 2022
read point to surface distance 
@author: pingping
"""
import os
import numpy as np
from glob import glob
import torch
from uniformity import get_uniform_loss


UNIFORM_PRECENTAGE_NAMES = ['0.4%', '0.6%', '0.8%', '1.0%', '1.2%']
precentages = np.array([0.004, 0.006, 0.008, 0.010, 0.012])


PRED_DIR = os.path.abspath('./result_ear')
MESH_DIR = '../data/PU1K/test/original_meshes/'


global_p2f=[]
metric_global={}
for uniform in UNIFORM_PRECENTAGE_NAMES:
        metric_global[uniform] = 0.0

for D in [PRED_DIR]:
    pred_paths = glob(os.path.join(D, "*.xyz"))
    for pred_path in pred_paths:
        if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
            #point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.xyz")
            xyzd = np.loadtxt(pred_path[:-4] + "_point2mesh_distance.xyz").astype(np.float32)
            if xyzd.size == 0:
                continue
            point2mesh_distance = xyzd[:, 3]
            current_p2f = np.nanmean(point2mesh_distance)
            global_p2f.append(current_p2f)


            ##uniformity
            #from uniformity.uniform import point_uniformity, UNIFORM_PRECENTAGE_NAMES
            pcd = xyzd[:, 0:3]
            pcd =torch.from_numpy(pcd).to('cuda').view(1,-1,3)
            #print(pred_path)
            loss_uniforms = get_uniform_loss(pcd)

            for (i, uniform) in enumerate(UNIFORM_PRECENTAGE_NAMES):
                #metric_local[uniform] = loss_uniforms[i]
                metric_global[uniform] += np.nan_to_num(loss_uniforms[i])

            
mean_p2f = np.nanmean(global_p2f)
print(mean_p2f)

uniforms_str = '\t'
for uniform in UNIFORM_PRECENTAGE_NAMES:
    uniforms_str += f'  [{uniform}]{metric_global[uniform]/len(pred_paths)}'
print(uniforms_str)
