# implement
import torch
import math
import pointnet2_ops.pointnet2_utils as pn2_utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample, \
    gather_operation, ball_query, three_nn, three_interpolate, grouping_operation
from knn_cuda import KNN
def get_uniform_loss(pcd, percentage=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0):
    B, N, C = pcd.shape[0], pcd.shape[1], pcd.shape[2]

    knn_uniform = KNN(k=16, transpose_mode=True)
    npoint = int(N * 0.05)
    loss = [0]*len(percentage)
    #print(loss)
    further_point_idx = pn2_utils.furthest_point_sample(pcd.permute(0, 2, 1).contiguous(), npoint)
    new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
    for (i, p) in enumerate(percentage):
        #print(i)
        nsample = int(N * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) / N
        idx = pn2_utils.ball_query(r, nsample, pcd.contiguous(), new_xyz.permute(0, 2, 1).contiguous())  # b N nsample

        expect_len = math.sqrt(disk_area)

        grouped_pcd = pn2_utils.grouping_operation(pcd.permute(0, 2, 1).contiguous(), idx)  # B C N nsample
        grouped_pcd = grouped_pcd.permute(0, 2, 3, 1)  # B N nsample C

        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, dim=1), dim=0)  # B*N nsample C

        dist, _ = knn_uniform(grouped_pcd, grouped_pcd)
        # print(dist.shape)
        uniform_dist = dist[:, :, 1:]  # B*N nsample 1
        uniform_dist = torch.abs(uniform_dist + 1e-8)
        uniform_dist = torch.mean(uniform_dist, dim=1)
        uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + 1e-8)
        mean_loss = torch.mean(uniform_dist)
        mean_loss = mean_loss * math.pow(p * 100, 2)
        loss[i]= mean_loss.item()
    return loss