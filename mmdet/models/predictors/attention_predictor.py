# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmdet.models import PREDICTORS


@PREDICTORS.register_module()
class AttentionPredictor(BaseModule):
    def __init__(self, in_channels, num_attention_blocks=16, init_cfg=None,
                 num_prompts=5, embed_dims=96
                 ):
        super(AttentionPredictor, self).__init__(init_cfg)

        self.reduction = nn.Linear(in_channels, embed_dims)

        self.query = nn.Parameter(torch.zeros(num_prompts, embed_dims))
        nn.init.normal_(self.query)

        self.fc_embed = nn.Linear(embed_dims, embed_dims)
        self.ref_fc_embed = nn.Linear(embed_dims, embed_dims)
        self.fc = nn.Linear(embed_dims, embed_dims)
        self.ref_fc = nn.Linear(embed_dims, embed_dims)
        self.num_attention_blocks = num_attention_blocks

    def forward(self, ref_x):
        # [n, c] -> [n, emb]
        ref_x = self.reduction(ref_x)

        x = self.query
        roi_n = x.shape[0]
        ref_roi_n = ref_x.shape[0]

        x_embed = self.fc_embed(x)
        # [num_attention_blocks, roi_n, C / num_attention_blocks]
        x_embed = x_embed.view(roi_n, self.num_attention_blocks,
                               -1).permute(1, 0, 2)

        ref_x_embed = self.ref_fc_embed(ref_x)
        # [num_attention_blocks, C / num_attention_blocks, ref_roi_n]
        ref_x_embed = ref_x_embed.view(ref_roi_n, self.num_attention_blocks,
                                       -1).permute(1, 2, 0)

        # [num_attention_blocks, roi_n, ref_roi_n]
        weights = torch.bmm(x_embed, ref_x_embed) / (x_embed.shape[-1]**0.5)
        weights = weights.softmax(dim=2)

        ref_x_new = self.ref_fc(ref_x)
        # [num_attention_blocks, ref_roi_n, C / num_attention_blocks]
        ref_x_new = ref_x_new.view(ref_roi_n, self.num_attention_blocks,
                                   -1).permute(1, 0, 2)

        # [roi_n, num_attention_blocks, C / num_attention_blocks]
        x_new = torch.bmm(weights, ref_x_new).permute(1, 0, 2).contiguous()
        # [roi_n, C]
        x_new = self.fc(x_new.view(roi_n, -1))
        return x_new
