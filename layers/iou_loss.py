import torch
from torch import nn
import torch.nn.functional as F

class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None, addition_avg = 0):
        """
        function: calculate IOULoss
        args:
            pred (Tensor) - the prediction
            target (Tensor) - the ground-truth
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / (weight.sum() + addition_avg)
        else:
            assert losses.numel() != 0
            return losses.mean()

class IOULoss_XYXY(nn.Module):
    def forward(self, pred, target):
        #print('pref.shape', pred.shape)
        #print('target.shape', target.shape)
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_right - target_left) * \
                      (target_bottom - target_top)
        pred_area = (pred_right - pred_left) * \
                    (pred_bottom - pred_top)
        intersect_left = torch.max(target_left, pred_left)
        intersect_right = torch.min(target_right, pred_right)
        intersect_top = torch.max(target_top, pred_top)
        intersect_bottom = torch.min(target_bottom, pred_bottom)
        area_intersect = (intersect_right - intersect_left) * (intersect_bottom - intersect_top)
        area_union = target_area + pred_area - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        return losses.mean()

class IOULoss_ori(nn.Module):
    def forward(self, pred, target, weight=None):
        """
        function:计算IOULoss
        args:
            pred (Tensor) - 预测值
            target (Tensor) - gt
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        #losses = -torch.log((torch.abs(area_intersect) + 1.0) / (torch.abs(area_union) + 1.0))
        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()

class GIOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        """
        function: calculate GIOULoss
        args:
            pred (Tensor) - l, t, r, b prediction
            target (Tensor) - l, t, r, b gt
        """
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)
        w_enclose = torch.max(pred_left, target_left) + \
                  torch.max(pred_right, target_right)
        h_enclose = torch.max(pred_bottom, target_bottom) + \
                  torch.max(pred_top, target_top)

        area_enclosing = w_enclose * h_enclose
        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        giou = (area_intersect)/(area_union) - ((area_enclosing-area_union)/(area_enclosing))

        losses = -torch.log((1+giou)/2)
        #losses = 1-giou
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


def bounded_iou_loss(pred, target, beta=1.0, reduction = "mean"):
    """
    args:
        pred (tensor) - shape: [batch*num_anchors, 4]
        target (tensor) - shape: [batch*num_anchors, 4]
        beta (float) - the threshold of L1 loss
    return:
        if none: shape of loss is
    """
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    target = target.type_as(pred)

    diff, _ = torch.min(torch.stack((pred/(target + 1e-6), target/(pred + 1e-6))), dim=0)

    diff = 1-diff
    #torch.where(condition, x, y)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # nono:0, mean:1, sum:2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()

def weighted_bounded_iou_loss(pred, target, weight, beta=1.0, avg_factor=None):
    """
    args:(for our case)
        pred (tensor) - shape: [batch*num_anchors, 4]
        target (tensor) - shape: [batch*num_anchors, 4]
        weight (tensor) - shape: [batch*num_anchors, 4]
        beta (float) - the threshold of L1 loss
        avg_factor - average factor
    """
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 2 + 1e-6
    loss = bounded_iou_loss(pred, target, beta, reduction='none')

    return torch.sum(loss * weight)[None] / avg_factor

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()

def weighted_smooth_l1_loss(pred, target, weight, beta=1.0, avg_factor=None):
    """
    args:(for our case)
        pred (tensor) - shape: [batch*num_anchors, 4]
        target (tensor) - shape: [batch*num_anchors, 4]
        weight (tensor) - shape: [batch*num_anchors, 4]
        beta (float) - the threshold of L1 loss
        avg_factor - average factor
    """
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor
