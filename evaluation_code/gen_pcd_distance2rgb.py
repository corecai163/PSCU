#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 17:17:03 2022
read point to surface distance 
@author: pingping
"""
import os
import sys
import numpy as np
from glob import glob
import open3d as o3d


def distance2rgb(values, scale = 1.0, color1=np.array([1, 0.2, 0.2]), color2=np.array([0.2, 0.2, 1])):
    valuesColors = []
    for currVal in values:
        if np.isnan(currVal):
            currVal=0
        clipVal = min(currVal/scale, 1.0)
        color = color1*clipVal + color2*(1.0-clipVal)
        valuesColors.append(color)
    return np.array(valuesColors)

def save_pcd(name,xyz,rgb):
    ## save pcd
    save_path = name
    print('saving:'+save_path)
#    print(save_path)
#    print(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    pcd.colors = o3d.utility.Vector3dVector(np.array(rgb))
    o3d.io.write_point_cloud(save_path,pcd)

if __name__ == "__main__":
    #check system argv
    if len(sys.argv) != 2:
        print('Usage: '+ sys.argv[0] + 'dir/to/result')
        exit(0)

    PRED_DIR = sys.argv[1]

    global_p2f=[]
    pred_paths = glob(os.path.join(PRED_DIR, "*.xyz"))

    for pred_path in pred_paths:
        if os.path.isfile(pred_path[:-4] + "_point2mesh_distance.xyz"):
            #point2mesh_distance = load(pred_path[:-4] + "_point2mesh_distance.xyz")
            xyzd = np.loadtxt(pred_path[:-4] + "_point2mesh_distance.xyz").astype(np.float32)
            if xyzd.size == 0:
                continue
            xyz = xyzd[:,0:3]
            point2mesh_distance = xyzd[:, 3]

            dis_rgb = distance2rgb(point2mesh_distance,0.01)
            name = pred_path[:-4] + '.ply'
            save_pcd(name,xyz,dis_rgb)
            current_p2f = np.nanmean(point2mesh_distance)
            global_p2f.append(current_p2f)
                
    mean_p2f = np.nanmean(global_p2f)
    print(mean_p2f)