#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
SparseRCNN Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .util.misc import inverse_sigmoid
from .util.box_ops import gen_sineembed_for_position, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .attention import MultiheadAttention

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.SparseRCNN.NUM_CLASSES
        d_model = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.SparseRCNN.DIM_FEEDFORWARD
        nhead = cfg.MODEL.SparseRCNN.NHEADS
        dropout = cfg.MODEL.SparseRCNN.DROPOUT
        activation = cfg.MODEL.SparseRCNN.ACTIVATION
        self.num_heads = cfg.MODEL.SparseRCNN.NUM_HEADS
        self.rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation, is_first=False)
        self.rcnn_head_first = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation, is_first=True)
        self.return_intermediate = cfg.MODEL.SparseRCNN.DEEP_SUPERVISION

        # Init parameters.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        self.num_classes = num_classes
        if self.use_focal:
            prior_prob = cfg.MODEL.SparseRCNN.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            if self.use_focal:
                if p.shape[-1] == self.num_classes:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):

        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def forward(self, features, init_bboxes, init_features, images_whwh, pos_imgs, pos_local):

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        nr_boxes = init_features.size(0)
        bboxes = init_bboxes

        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()

        for i in range(self.num_heads):
            if i == 0:
                class_logits, pred_bboxes, proposal_features = self.rcnn_head_first(features, bboxes, proposal_features,
                                                                     self.box_pooler, images_whwh, pos_imgs, pos_local)
            else:
                class_logits, pred_bboxes, proposal_features = self.rcnn_head(features, bboxes, proposal_features,
                                                                                    self.box_pooler, images_whwh,
                                                                                    pos_imgs, pos_local)
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0), is_first=False):
        super().__init__()

        self.is_first = is_first
        self.d_model = d_model

        self.pos_xywh = MLP(2 * d_model, d_model, d_model, 2)

        # dynamic.
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = cfg.MODEL.SparseRCNN.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.SparseRCNN.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.SparseRCNN.USE_FOCAL
        if self.use_focal:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, images_whwh, pos_imgs, pos_local):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        """

        N, nr_boxes = bboxes.shape[:2]

        # roi_feature.
        proposal_boxes = list()
        proposal_box = box_cxcywh_to_xyxy(bboxes) * images_whwh[:, None, :]
        for b in range(N):
            proposal_boxes.append(Boxes(proposal_box[b]))
        roi_features = pooler(features, proposal_boxes)
        pos_img = pooler(pos_imgs, proposal_boxes)
        pos_img = pos_img * pos_local
        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)
        pos_img = pos_img.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        pos_xywh = gen_sineembed_for_position(bboxes)
        pos_box_selfattn = self.pos_xywh(pos_xywh).permute(1, 0, 2)

        # self_attn.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        q_content = self.sa_qcontent_proj(pro_features)
        k_content = self.sa_kcontent_proj(pro_features)
        v = self.sa_v_proj(pro_features)
        q_pos = self.sa_qpos_proj(pos_box_selfattn)
        k_pos = self.sa_kpos_proj(pos_box_selfattn)
        q = q_content + q_pos
        k = k_content + k_pos
        pro_features2 = self.self_attn(q, k, value=v)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pos_xywh = pos_xywh.reshape(1, N * nr_boxes, -1)
        pro_features2 = self.inst_interact(pro_features, roi_features, pos_xywh, pos_img, pos_local, bboxes)
        pro_features = pro_features + self.dropout2(pro_features2)
        pro_features = self.norm2(pro_features)
        # 2nd inst_interact
        if self.is_first:
            obj_features = pro_features
        else:
            pro_features2 = self.inst_interact(pro_features, roi_features, pos_xywh, pos_img, pos_local, bboxes)
            pro_features = pro_features + self.dropout2(pro_features2)
            obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        bboxes_before_sigmoid = inverse_sigmoid(bboxes.view(-1, 4))
        pred_bboxes = bboxes_before_sigmoid + bboxes_deltas
        pred_bboxes = pred_bboxes.sigmoid()

        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.SparseRCNN.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.SparseRCNN.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.SparseRCNN.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.query_scale = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 2)
        self.ref_point_head = MLP(self.hidden_dim, self.hidden_dim, 2, 2)
        self.ca_qpos_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ca_kpos_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features, pos_xywh, pos_img, pos_local, bboxes_xywh):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features_pos_img = self.ca_qpos_proj(pos_img).permute(1, 0, 2)
        features = roi_features.permute(1, 0, 2)
        features_pos = features_pos_img
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        # transformation
        pos_transformation = self.query_scale(pro_features)
        pos_xywh = pos_xywh[..., :self.hidden_dim] * pos_transformation

        # modulated hw
        refHW_cond = self.ref_point_head(pro_features).sigmoid()
        bboxes_xywh = bboxes_xywh.reshape(1, -1, 4)
        pos_xywh[..., self.hidden_dim // 2:] *= (refHW_cond[..., 0] / bboxes_xywh[..., 2]).unsqueeze(-1)
        pos_xywh[..., :self.hidden_dim // 2] *= (refHW_cond[..., 1] / bboxes_xywh[..., 3]).unsqueeze(-1)
        pos_xywh = pos_xywh.permute(1, 0, 2) * pos_local.flatten()[None, :, None]

        param1_pos_xywh = self.ca_kpos_proj(pos_xywh)
        param1_pos = param1_pos_xywh

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features_pos = (features_pos * param1_pos).sum(-1)
        features = features + features_pos[:, :, None]  #equal to torch.bmm(torch.cat([features, features_pos], dim=-1), torch.cat([param1, param1_pos], dim=1))
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
