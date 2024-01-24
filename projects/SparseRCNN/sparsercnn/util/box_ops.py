# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Utilities for bounding box manipulation and GIoU.
"""
import torch, math
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


# def gen_sineembed_for_position(pos_tensor):
#     # n_query, bs, _ = pos_tensor.size()
#     # sineembed_tensor = torch.zeros(n_query, bs, 256)
#     scale = 2 * math.pi
#     dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
#     dim_t = 10000 ** (2 * (dim_t // 2) / 128)
#     x_embed = pos_tensor[:, :, 0] * scale
#     y_embed = pos_tensor[:, :, 1] * scale
#     pos_x = x_embed[:, :, None] / dim_t
#     pos_y = y_embed[:, :, None] / dim_t
#     pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
#     pos = torch.cat((pos_y, pos_x), dim=2)
#     return pos

def gen_sineembed_for_4d_position(proposals):
    num_pos_feats = 128
    temperature = 10000
    scale = 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    # N, L, 4
    proposals = proposals * scale
    # N, L, 4, 64
    pos = proposals[:, :, :, None] / dim_t
    # N, L, 4, 64, 2
    pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
    return pos

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def gen_time_emb(x):
    scale = 2 * math.pi
    emb = torch.arange(128, dtype=torch.float32, device=x.device)
    emb = 10000 ** (2 * (emb // 2) / 128)
    x_embed = x.unsqueeze(0) * scale
    emb = torch.outer(x_embed, emb)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

def gen_centerness_for_position(pooler_resolution):
    x = torch.ones(pooler_resolution, pooler_resolution)
    y_embed = x.cumsum(0, dtype=torch.float32) - 0.5
    x_embed = x.cumsum(1, dtype=torch.float32) - 0.5
    t = y_embed - 0
    b = pooler_resolution - y_embed
    l = x_embed - 0
    r = pooler_resolution - x_embed
    centerness = torch.sqrt((torch.min(t, b) / torch.max(t, b)) * (torch.min(l, r) / torch.max(l, r)))
    return centerness

def gen_adacenterness_for_position(pooler_resolution, points):
    x = torch.ones(pooler_resolution, pooler_resolution).cuda()
    y_embed = x.cumsum(0, dtype=torch.float32) - 0.5
    x_embed = x.cumsum(1, dtype=torch.float32) - 0.5
    t = (y_embed - 0)[None, :, :] * torch.exp(-(1 + points[..., 1]))[:, None, None]
    b = (pooler_resolution - y_embed)[None, :, :] * torch.exp(-(1 - points[..., 1]))[:, None, None]
    l = (x_embed - 0)[None, :, :] * torch.exp(-(1 + points[..., 0]))[:, None, None]
    r = (pooler_resolution - x_embed)[None, :, :] * torch.exp(-(1 - points[..., 0]))[:, None, None]
    centerness = torch.sqrt((torch.min(t, b) / torch.max(t, b)) * (torch.min(l, r) / torch.max(l, r)))
    return centerness

# def gen_adacenterness_for_position_scale(pooler_resolution, points):
#     x = torch.ones(pooler_resolution, pooler_resolution).cuda()
#     y_embed = x.cumsum(0, dtype=torch.float32) - 0.5
#     x_embed = x.cumsum(1, dtype=torch.float32) - 0.5
#     t = (y_embed - 0)[None, :, :] * (1 - points[..., 1])[:, None, None]
#     b = (pooler_resolution - y_embed)[None, :, :] * (1 + points[..., 1])[:, None, None]
#     l = (x_embed - 0)[None, :, :] * (1 - points[..., 0])[:, None, None]
#     r = (pooler_resolution - x_embed)[None, :, :] * (1 + points[..., 0])[:, None, None]
#     centerness = torch.sqrt((torch.min(t, b) / torch.max(t, b)) * (torch.min(l, r) / torch.max(l, r)))
#     return centerness
#
# def gen_adacenterness_for_position_scale(pooler_resolution, points):
#     x = torch.ones(pooler_resolution, pooler_resolution).cuda()
#     y_embed = x.cumsum(0, dtype=torch.float32) - 0.5
#     x_embed = x.cumsum(1, dtype=torch.float32) - 0.5
#     t = (y_embed - 0)[None, :, :] * torch.exp(-points[..., 1])[:, None, None]
#     b = (pooler_resolution - y_embed)[None, :, :] * torch.exp(points[..., 1])[:, None, None]
#     l = (x_embed - 0)[None, :, :] * torch.exp(-points[..., 0])[:, None, None]
#     r = (pooler_resolution - x_embed)[None, :, :] * torch.exp(points[..., 0])[:, None, None]
#     centerness = torch.sqrt((torch.min(t, b) / torch.max(t, b)) * (torch.min(l, r) / torch.max(l, r)))
#     return centerness

def gen_adacenterness_for_position_scale(pooler_resolution, points):
    x = torch.ones(pooler_resolution, pooler_resolution).cuda()
    y_embed = x.cumsum(0, dtype=torch.float32) - 0.5
    x_embed = x.cumsum(1, dtype=torch.float32) - 0.5
    t = (y_embed - 0)[None, :, :] * torch.exp(-points[..., 1])[:, None, None]
    b = (pooler_resolution - y_embed)[None, :, :] * torch.exp(points[..., 1])[:, None, None]
    l = (x_embed - 0)[None, :, :] * torch.exp(-points[..., 0])[:, None, None]
    r = (pooler_resolution - x_embed)[None, :, :] * torch.exp(points[..., 0])[:, None, None]
    centerness = torch.sqrt((torch.min(t, b) / torch.max(t, b)) * (torch.min(l, r) / torch.max(l, r)))
    return centerness