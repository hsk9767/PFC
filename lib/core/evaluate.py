# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]])
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

def get_dist(output, gt_coords, target_weight):
    '''
    output : [B, J, H, W], tensor, detached
    gt_coords : [B, J, 3], tensor
    '''
    B, J, H, W = output.shape
    
    gt_coords = gt_coords[:, :, :2]
    gt_coords[:, :, 0] = gt_coords[:, :, 0] / W
    gt_coords[:, :, 1] = gt_coords[:, :, 1] / H
    
    pred_coords = torch.zeros(size=(B, J, 2))
    
    pred_heatmap = output.reshape(shape=(B, J, -1))
    max_idx = pred_heatmap.max(dim=2)[1]
    pred_coords[:, :, 0] = (max_idx % W) / float(W) # normalized coordinates
    pred_coords[:, :, 1] = (max_idx // W) / float(H) # normalized coordinates
    
    dist = (gt_coords.cpu() - pred_coords).pow(2).sum(dim=2).sqrt().unsqueeze(2)
    
    dist[target_weight == 0.] = -1.
    
    return dist.cuda() # [B, J, 1]


def get_spreaded(output, target_weight, thre = 0.7):
    ''' 
    output : [B, J, H, W], detached
    target_weight : [B, J, 2]
    '''
    B, J, H, W = output.shape
    output_reshaped = output.reshape((B, J, -1))
    
    b_j_max = output_reshaped.max(dim=2, keepdim=True)[0]
    b_j_max[b_j_max == 0.] = 1.
    
    output = output_reshaped.reshape((B, J, H, W))
    output = output / b_j_max.unsqueeze(3)
    
    output_x = output.sum(dim=2) # [B, J, 48]
    output_y = output.sum(dim=3) # [B, J, 64]
    output_x_max = output_x.max(dim=2, keepdim=True)[0]
    output_y_max = output_y.max(dim=2, keepdim=True)[0]
    
    output_x_max[output_x_max == 0.] = 1.0
    output_y_max[output_y_max == 0.] = 1.0
    
    output_x = output_x / output_x_max
    output_y = output_y / output_y_max
    
    output_x_over = (output_x >= 0.7).float().cpu() # [B, J, W]
    output_y_over = (output_y >= 0.7).float().cpu() # [B, J, H]
    
    x_idx_canvas = torch.arange(W).repeat(B, J, 1)
    y_idx_canvas = torch.arange(H).unsqueeze(1).repeat(B, J, 1, 1).squeeze(3)
    
    x_idxs = output_x_over * x_idx_canvas # [B, J, H, W]
    y_idxs = output_y_over * y_idx_canvas # [B, J, H, W]
    # print(x_idxs[0, 0, :])
    
    x_idxs_max = torch.max(x_idxs, dim=2, keepdim=True)[0] # [B, J, 1]
    y_idxs_max = torch.max(y_idxs, dim=2, keepdim=True)[0] # [B, J, 1]
    # print(x_idxs_max[0, 0, :])
    
    
    x_idxs_zero = (x_idxs == 0.).float()
    y_idxs_zero = (y_idxs == 0.).float()
    
    x_idxs = x_idxs_zero * (x_idxs_max * 0.999) + (1 - x_idxs_zero) * x_idxs
    y_idxs = y_idxs_zero * (y_idxs_max * 0.999) + (1 - y_idxs_zero) * y_idxs
    
    x_idxs_min = torch.min(x_idxs, dim=2, keepdim=True)[0]
    y_idxs_min = torch.min(y_idxs, dim=2, keepdim=True)[0]
    
    x_spreaded = (x_idxs_max - x_idxs_min) / W
    y_spreaded = (y_idxs_max - y_idxs_min) / H
    
    return x_spreaded.cuda(), y_spreaded.cuda()