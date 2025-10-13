import sys; sys.path.append('/home/xp/stereo_project/')
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

from stereo_toolbox.disparity_regression import *

def save_results(results, file_path):
    results = results.flatten()

    with open(file_path, 'a') as file:
        file.write(' '.join(map(str, results)))
        file.write('\n')  # 换行


def mode_split(pred, gt, maxdisp=192, epsilon=1e-3):
    """
    input
        pred: the disparity distribution of model output
        gt: dispairty ground-truth
        maxdisp: disparity search range
        epsilon: threshold of first-order derivative for judging monotonicity
    output
        target_mode: target uni-modal distribution which contains the ground-truth
        non_target_mode: non-target multi-modal distribution which contains the dark knowledge
    """
    assert pred.dim() == 4
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    else:
        assert gt.dim() == 4 and gt.shape[1] == 1

    N, maxdisp, H, W = pred.shape

    # extract mode edge
    diff = torch.diff(pred, n=1, dim=1, prepend=torch.zeros(N,1,H,W,device=pred.device))
    mode_edge = ((diff < epsilon) * torch.concat(((diff > epsilon)[:,1:,...],torch.zeros(N,1,H,W, device=pred.device)), dim=1)
                + (diff < -epsilon) * torch.concat(((diff > -epsilon)[:,1:,...],torch.zeros(N,1,H,W, device=pred.device)), dim=1)) > 0

    # prevent index overflow  
    mode_edge[:,[0,-1],...] = 1

    # find the mode range
    mask = torch.arange(maxdisp,dtype=pred.dtype,device=pred.device).reshape([1,maxdisp,1,1]).repeat(N,1,H,W)
    mask_l = maxdisp - 1 - torch.argmax((mode_edge * (mask <= gt) * 1.0).flip(dims=[1]) ,dim=1,keepdim=True)
    mask_r = torch.argmax((mode_edge * (mask > gt) * 1.0),dim=1,keepdim=True)

    mask = (mask>=mask_l) * (mask<=mask_r)
    target_mode = pred * mask
    non_target_mode = pred * ~mask

    return target_mode, non_target_mode

# def fit_laplacian_scale(target_mode, scale_range=[0.5,1.5]):
#     def cal_entropy(x, dim=1, epsilon=1e-6):
#         epsilon = torch.finfo(x.dtype).eps if epsilon is None else epsilon
#         norm_x = x / (x.sum(dim=dim,keepdim=True) + epsilon)
#         return (- norm_x * torch.log2(norm_x + epsilon)).sum(dim=dim, keepdim=True)
    
#     minn, maxx = scale_range
#     entropy = cal_entropy(target_mode)
#     scale = ((entropy - entropy.mean()) / entropy.std() * (maxx-minn)/2 + (maxx+minn) / 2).clamp(min=minn, max=maxx)
#     return scale

def fit_laplacian_scale(target_mode, scale_range=[0.5,1.5]):
    target_mode = target_mode / target_mode.sum(dim=1, keepdim=True)
    disp = torch.arange(target_mode.shape[1], device=target_mode.device).reshape(1,-1,1,1)
    mu = (target_mode * disp).sum(dim=1,keepdim=True)
    scale = (torch.abs(mu-disp) * target_mode).sum(dim=1, keepdim=True)
    
    # re-scale
    minn, maxx = scale_range
    meann = (minn + maxx) / 2
    scale = ((scale - scale.mean()) / scale.std() + meann).clamp(min=minn, max=maxx)
    return scale


def fusion_weight(error_map_list:list, fusion_method:str):
    error_map_list = torch.stack(error_map_list, dim = 0)

    if fusion_method == 'WTA':
        _, max_indices = torch.max(error_map_list, dim=0)
        weight = torch.zeros_like(error_map_list, dtype=torch.bool)
        weight = weight.scatter_(0, max_indices.unsqueeze(0), True)
    elif fusion_method == "random":
        error_map_list = torch.rand_like(error_map_list)
        _, max_indices = torch.max(max_indices, dim=0)
        weight = torch.zeros_like(max_indices, dtype=torch.bool)
        weight = weight.scatter_(0, max_indices.unsqueeze(0), True)
    elif fusion_method == "average":
        weight = torch.ones_like(error_map_list) / error_map_list.shape[0]
    elif fusion_method == "soft":
        weight =  F.softmax(-error_map_list, dim = 0)

    return weight

    
def groundtruth_modeling(pseudo_label_list, disp, fusion_method='soft', non_target_thread=0.5, scale_range=[0.5,1.5]):
    gt = disp.to(pseudo_label_list[0].device)

    maxdisp = pseudo_label_list[0].shape[1]

    # calculate the error map
    scale_list = []
    non_target_mode_list = []
    error_map_list = []

    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    regression = unimodal_disparityregression_SA()
    for i, x in  enumerate(pseudo_label_list):
        error_map_list.append(torch.abs(regression(x) - gt))

        target_mode, non_target_mode = mode_split(x, gt)
        scale_list.append(fit_laplacian_scale(target_mode, scale_range)) # fit target mode with Laplacian distribution
        non_target_mode_list.append(non_target_mode)

    weight = fusion_weight(error_map_list, fusion_method)

    assert torch.all(torch.abs(weight.sum(0) - 1) < 1e6)
    
    non_target_mode = (weight * torch.stack(non_target_mode_list, dim=0)).sum(dim=0)
    
    scale = (weight * torch.stack(scale_list, dim=0)).sum(dim=0)
    target_mode = F.softmax(-torch.abs(torch.arange(maxdisp, device=weight.device).reshape(1,maxdisp,1,1) - gt) / scale, dim=1)

    embedding_weight = non_target_mode.sum(dim=1,keepdim=True).clamp(max = non_target_thread)

    gt = embedding_weight * non_target_mode + (1-embedding_weight) * target_mode

    return gt.detach_()