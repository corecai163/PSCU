#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 7 10:27:52 2022

@author: pingping
"""
import torch
import torch.nn as nn
from models.pointnet import MLP_CONV, Transformer
from utils.tools import compute_rotation_matrix_from_ortho6d

def pos_emb(dxy):
    '''
    pos_embedding for dx and dy
    input dxy # B by N*4 by 2 
    output pos_fea (-1,16,1)
    '''
    nk = dxy.view(-1,2)
    x = nk[:,0]
    y = nk[:,1]
    one = torch.zeros_like(x, device=x.device)
    x2 = torch.pow(x, 2)
    y2 = torch.pow(y, 2)
    x3 = torch.pow(x, 3)
    y3 = torch.pow(y, 3)
    xx = torch.stack([one,x,x2,x3],dim=0).view(-1,4,1)
    yy = torch.stack([one,y,y2,y3],dim=0).view(-1,1,4)
    emb = torch.bmm(xx,yy) + 1e-4
    return emb.view(-1,16,1)

class SurfaceConstrainedUp(nn.Module):

    '''
    Manifold Deconvolution
    '''
    def __init__(self,upscale,manifold_mlp):
        super(SurfaceConstrainedUp,self).__init__()
        self.manifold_mlp = manifold_mlp
        self.dxy_mlp = MLP_CONV(in_channel=128, layer_dims=[32, 2])
        
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + 512, layer_dims=[512, 128])
        self.transformer_1 = Transformer(128, dim=64)
        
        self.coef_mlp = MLP_CONV(in_channel=64, layer_dims=[16, 16])
        self.rot_mlp = MLP_CONV(in_channel=64, layer_dims=[16, 6])

        self.upscale=upscale
        self.mlp_ps = MLP_CONV(in_channel=128+128+64, layer_dims=[256, 64])
        self.ps = nn.ConvTranspose1d(64, 128, upscale, upscale, bias=False)   # point-wise splitting
        self.mlp_ch = MLP_CONV(in_channel=128+64, layer_dims=[128, 128])
        
    def forward(self,global_shape_fea, parent_pos, parent_fea, parent_manifold):
        '''
        inputs : ptcloud, ptnormal, patch_indicator
        parent pos: batch by 3 by N
        '''
        batch = parent_pos.size(0)
        N = parent_pos.size(2)
        upscale = self.upscale

        ## duplicate global shape fea
        shape_fea = global_shape_fea.repeat(1,1,N)
        
        ## Step1: Spatial Feature Aggregration 
        feat_1 = parent_fea
        feat_2 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            shape_fea], 1)
        xyz_parent_fea = self.mlp_2(feat_2)
        if parent_fea != None:
            xyz_parent_fea = xyz_parent_fea + parent_fea
        #new relative parent fea
        relative_parent_fea = self.transformer_1(xyz_parent_fea, parent_pos)
        # if (torch.isnan(relative_parent_fea).any()):
        #     print('Problem in relative_parent_fea')
        
        ## manifold fea
        if parent_manifold != None:
            manifold_fea = (self.manifold_mlp(torch.cat([parent_pos,parent_manifold,relative_parent_fea,shape_fea],-2))) #B by 16 by N
            manifold_fea = manifold_fea + parent_manifold
        else:
            manifold_fea = (self.manifold_mlp(torch.cat([parent_pos,relative_parent_fea,shape_fea],-2))) #B by 16 by N
        
        ## output for next MD-Conv block
        childmanifold_fea = torch.repeat_interleave(manifold_fea, upscale, dim=2)  # B by 32 by N*4
        
        ## rot fea
        rot = self.rot_mlp(manifold_fea)
        rot = torch.repeat_interleave(rot, upscale, dim=2)  # B by 32 by N*4
        rot_fea = rot.permute(0, 2, 1).contiguous()
        rot_fea = rot_fea.view(-1,6)
        
        #pytorch3d.transforms.axis_angle_to_matrix
#        rot_fea = manifold_fea[:,:,0:6]
#        rot_fea = rot_fea.view(-1,6)
        rot_matrix = compute_rotation_matrix_from_ortho6d(rot_fea)
        # if (not torch.isnan(rot_fea).any() and torch.isnan(rot_matrix).any()):
        #     print('Problem in compute_rotation_matrix_from_ortho6d')

        ## calculate dxy of child points on manifold
        dmfeat_child = self.mlp_ps(torch.cat([xyz_parent_fea,relative_parent_fea-xyz_parent_fea,manifold_fea],dim=-2))
        dmchild_fea = self.ps(dmfeat_child)  # (B, 128, N_prev * up_factor)
        deform_uv = torch.tanh(self.dxy_mlp(dmchild_fea)) #(batch,2*upscale,N)

        ## generate child fea
        current_parent_fea = torch.repeat_interleave(xyz_parent_fea, upscale, dim=2)  # B by M by N*upscale
        child_fea = self.mlp_ch(torch.cat([dmchild_fea,childmanifold_fea],dim=-2))
        child_fea = child_fea + current_parent_fea
        
        ## pos_embedding
        child_duv = deform_uv.view(batch,2,-1) # B by 2 by N*upscale
        pos = child_duv.permute(0, 2, 1).contiguous() # B by N*4 by 2 
        pos_fea = pos_emb(pos)
        # if (torch.isnan(pos_fea).any()):
        #     print('Problem in compute_pos_emb')
        
        ## calculate dw on surface
        coef = self.coef_mlp(manifold_fea)
        coef = torch.repeat_interleave(coef, upscale, dim=2)  # B by 32+32 by N*4
        coef = coef.permute(0, 2, 1).contiguous().view(-1,1,16)
        child_dw = torch.bmm(coef,pos_fea)
        #child_dz = child_dz.view(batch,1,-1)
        child_duv = child_duv.permute(0, 2, 1).contiguous()
        child_duv = child_duv.view(-1,2)
        child_duvw = torch.cat([child_duv,child_dw.view(-1,1)],-1)
        
        ## rotation
        #rot_m = torch.repeat_interleave(rot_matrix, upscale, dim=0)
        delta_rot_xyz = torch.bmm(rot_matrix,child_duvw.view(-1,3,1))

        delta_rot_xyz = delta_rot_xyz.view(batch,N*upscale,3)
        delta_rot_xyz = delta_rot_xyz.permute(0,2,1).contiguous()
        
        # child position
        child_pos = delta_rot_xyz + torch.repeat_interleave(parent_pos, upscale, dim=-1)
            
        # minimize dz
        #child_duv = child_duv.permute(0, 2, 1).contiguous().view(-1,2)
        #dk = 1e-8 + torch.sqrt(torch.sum(child_duv ** 2, -1))
        a =  coef[:,:,0]
        #dw_cons = torch.div(child_dz.view(batch,-1),r.view(batch,-1))
        #dw_cons = child_dz.view(-1)/dk - a.view(-1)
        dw_cons = child_dw.view(-1)
        return child_pos,child_fea,childmanifold_fea, rot_matrix.view(batch,N,-1), dw_cons
    
    
def min_dist(x):
    '''
    x: Tensor, (b,3,n)
    r: Tensor, (b,1)
    '''
    #B = x.size(0)
    
    xT = x.permute(0, 2, 1).contiguous()
    xx = torch.bmm(xT,x)
    diag_rx = torch.diagonal(xx,dim1=1,dim2=2)#(xx.diag().unsqueeze(0).expand_as(xx))
    rx = diag_rx.unsqueeze(1).expand_as(xx)
    rxT = rx.permute(0, 2, 1).contiguous()
    bias = torch.ones([xT.size(0),xT.size(1)], dtype=torch.float32, device='cuda')
    P = (rxT + rx - 2*xx) + torch.diag_embed(bias, offset=0, dim1=-2, dim2=-1)
    min_v,min_i = torch.min(P,1) # (B, N)
    r = torch.max(min_v,1)
    #r = torch.mean(min_v,1) #(B, )
    return r