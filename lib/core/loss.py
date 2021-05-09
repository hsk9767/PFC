# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):#, logical):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight :#and logical is not None:
                loss+=0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),#.mul(logical[:, idx].unsqueeze(-1)),
                    heatmap_gt.mul(target_weight[:, idx])#.mul(logical[:, idx].unsqueeze(-1))
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
    
    def soft_argmax(self, heatmap, alpha=100.):
        batch_size, num_joint, h, w = heatmap.shape
        heatmap_ = heatmap.reshape((batch_size, num_joint, -1))
        heatmap_ = F.softmax(heatmap_ * alpha, dim=2)
        heatmap_ = heatmap_.reshape((batch_size, num_joint, h, w))
        
        accu_x = heatmap_.sum(dim=2)
        accu_y = heatmap_.sum(dim=3)
        
        accu_x = accu_x * torch.arange(w).float().cuda()[None, None, :]
        accu_y = accu_y * torch.arange(h).float().cuda()[None, None, :]
        
        accu_x = accu_x.sum(dim=2, keepdim=True)
        accu_y = accu_y.sum(dim=2, keepdim=True)
        
        coord_out = torch.cat((accu_x, accu_y), dim=2)
        
        return coord_out
    
    def get_coord_loss(self, output, gt_coords, target_weight):
        pred_coord = self.soft_argmax(output)
        loss = torch.abs(gt_coords[:, :, :-1] - pred_coord) * target_weight
        return loss.mean()

