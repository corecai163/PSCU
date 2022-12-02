import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist
#from Emd.emd_module import emdFunction
from models.pointnet import fps_subsample

chamfer_dist = chamfer_3DDist()


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)


def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1

#def EMD(p1,p2):
#    dist, _ = emdFunction.apply(p1, p2, 0.005, 3000)
#    return torch.sqrt(dist).mean()

def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side
    #RL = torch.nn.MSELoss()
    
    Pc, P1, P2, z,rot3 = pcds_pred
    #P1, P2, r = pcds_pred
#    gt_rot = torch.eye(3).to('cuda')
#    rot_loss = RL(gt_rot.repeat(r.size(0),1,1),r)
    loss_z = torch.mean(torch.square(z))
#    loss_rm = torch.mean(torch.abs(rm))
    #gt_2 = fps_subsample(gt, P2.shape[1])
    #gt_1 = fps_subsample(gt_2, P1.shape[1])
    #gt_c = fps_subsample(gt_1, Pc.shape[1])
    #P3 = fps_subsample(P3,gt.shape[1])
    cdc = CD(Pc, gt)
    cd1 = CD(P1, gt)
    cd2 = CD(P2, gt)
    #cd3 = CD(P3, gt)


    loss_all = (1*cdc + 1*cd1 + 1*cd2) * 1e3 + 1e3*(loss_z)
    losses = [cdc, cd1, cd2, loss_z]
    return loss_all, losses, rot3


