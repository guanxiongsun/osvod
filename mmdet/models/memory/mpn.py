# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from .memory_bank import MemoryBank


from ..builder import NECKS


@NECKS.register_module()
class MPN(BaseModule):
    r"""Memory Pyramid Network.
    Args:
        in_channels (List[int]): Number of input channels per scale.
    """

    def __init__(self, in_channels,
                 scales,
                 before_fpn
                 ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.scales = scales
        self.before_fpn = before_fpn

        # add memory modules for every level
        self.memories = nn.ModuleList()
        for _in_channels in self.in_channels:
            _memory = MemoryBank(_in_channels)
            self.memories.append(_memory)

    # def write_single_level(self, x, stride, bboxes):
    #     """
    #     save pixels within detected boxes into memory
    #     :param stride: stride of feature map
    #     :param x: [N, C, H, W]
    #     :param bboxes: [N, 5]
    #     :return
    #     """
    #     # no obj detected
    #     if all(bboxes) == 0:
    #         # do nothing
    #         return
    #
    #     pred_labels = proposals.get_field("labels").detach().cpu().numpy()
    #     pred_scores = proposals.get_field("scores").detach().cpu().numpy()
    #     boxes = proposals.bbox.detach()
    #     # stride = 16
    #     boxes = (boxes / stride).int().cpu().numpy()
    #     temp_obj_pixels = []
    #
    #     if pred_scores.max() < self.score_thresh:
    #         # no high quality obj -> do nothing
    #         return
    #
    #     for box, pred_score, pred_label in zip(boxes, pred_scores, pred_labels):
    #         if pred_score >= self.score_thresh:
    #             # 1. map pixels in box to new index on x_box [H*W, C]
    #             # box [x1, y1, x2, y2] -> [ind_1, ind_2, ind_3, ... ]
    #             inds = sorted(self.box_to_inds_list(box, width))
    #             # 2. get mem_dict
    #
    #             # save part obj
    #             if len(inds) > PIXEL_NUM:
    #                 inds = np.asarray(inds)
    #                 inds = np.random.choice(inds, PIXEL_NUM, replace=False)
    #
    #             pixels = x[inds]
    #
    #             self.update(pixels)
    #
    #         elif pred_score >= 0.5:
    #             # quality [0.5, 0.9)
    #             inds = sorted(self.box_to_inds_list(box, width))
    #
    #             # save part obj
    #             if len(inds) > PIXEL_NUM:
    #                 inds = np.asarray(inds)
    #                 inds = np.random.choice(inds, PIXEL_NUM, replace=False)
    #
    #             pixels = x[inds]
    #             temp_obj_pixels.append(pixels)
    #
    #     # obj irr pixels
    #     obj_irr_pixels = self.get_obj_irr_pixels(x)
    #     # save part of irr pixels
    #     if len(obj_irr_pixels) > PIXEL_NUM:
    #         inds = np.arange(len(obj_irr_pixels))
    #         inds = np.random.choice(inds, PIXEL_NUM, replace=False)
    #         obj_irr_pixels = obj_irr_pixels[inds]
    #
    #     if len(temp_obj_pixels) > 0:
    #         # low quality obj
    #         obj_temp_pixels = torch.cat(temp_obj_pixels, dim=0)
    #         obj_irr_pixels = torch.cat([obj_temp_pixels, obj_irr_pixels])
    #     self.obj_irr_mem = obj_irr_pixels
    #     return

    def write_single_level_train(self, x, gt_bboxes, level):
        """
        save pixels within detected boxes into memory
        :param x: [N, C, H, W]
        :param gt_bboxes: [n, 5(ind, x, y, x, y)], e.g.,
            tensor([[  0.0000, 329.8406, 247.6415, 591.7313, 428.7736],
                    [  1.0000, 305.7750, 247.6415, 580.4062, 431.6038]], device='cuda:0')
        :param level: the current level
        :return
        """
        # assert len(x) == len(gt_bboxes)
        stride = self.scales[level]
        memory = self.memories[level]

        n, c, _, w = x.size()
        ref_obj_irr_list = []
        ref_obj_list = []
        for i, _x in enumerate(x):
            # [C, H, W] -> [H*W, C]
            _x = _x.view(c, -1).permute(1, 0)

            # get bboxes of this image
            ind = gt_bboxes[:, 0] == i
            _bboxes = gt_bboxes[ind][:, 1:]     # [n0, 4]
            # no object
            if len(_bboxes) == 0:
                # obj irr pixels
                ref_obj_irr_list.append(self.get_obj_irr_pixels(_x))
                # memory.update(self.get_obj_irr_pixels(_x))
                return

            # have objects
            boxes = torch.div(_bboxes, stride).int()

            for box in boxes:
                # 1. map pixels in box to new index on x_box [H*W, C]
                # box [x1, y1, x2, y2] -> [ind_1, ind_2, ind_3, ... ]
                inds = sorted(self.box_to_inds_list(box, w))

                inds = np.asarray(inds)

                # save part obj
                PIXEL_NUM = 100
                if len(inds) > PIXEL_NUM:
                    inds = np.random.choice(inds, PIXEL_NUM, replace=False)

                ref_obj_list.append(_x[inds])
                # memory.update(_x[inds])

        ref_all = ref_obj_list + ref_obj_irr_list

        return torch.cat(ref_all, dim=0)

    @staticmethod
    def box_to_inds_list(box, w):
        inds = []
        for x_i in range(box[0], box[2] + 1):
            for y_j in range(box[1], box[3] + 1):
                inds.append(int(x_i + y_j * w))
        return inds

    @staticmethod
    def get_obj_irr_pixels(x, scale=1.0):
        """
        get object irrelevant features
        :param x: [n, c]
        :param scale: factor to control threshold
        :return: [m, c]
        """
        n, c = x.size()
        l2_norm = x.pow(2).sum(dim=1).sqrt() / np.sqrt(c)
        keep_irrelevant = (F.softmax(l2_norm, dim=0) > scale / n)
        pixels = x[keep_irrelevant]
        return pixels

    @torch.no_grad()
    def prepare_memory_train(self, ref_x, ref_gt_bboxes):
        ref_feats_all = []
        for i in range(len(self.memories)):
            _ref_x = ref_x[i]
            ref_feats_all.append(
                self.write_single_level_train(
                    _ref_x, ref_gt_bboxes[0].clone(), i)
            )
        return ref_feats_all

    @staticmethod
    def filter_with_mask(query, mask=None):
        if mask is None:
            return query
        else:
            return query[mask]

    @staticmethod
    def update_with_query(_input, query_new, mask=None):
        if mask is None:
            return query_new
        else:
            _input[mask] = query_new
            return _input

    # all_x = self.mpn.forward_train(all_x, gt_bboxes, ref_gt_bboxes)
    def forward_train(self, inputs,
                      gt_bboxes=None,
                      ref_gt_bboxes=None):
        assert len(inputs) == len(self.in_channels)

        x = []
        ref_x = []
        for i in range(len(inputs)):
            x.append(inputs[i][[0]])
            ref_x.append(inputs[i][1:])

        # save ref feats to all levels of memory
        ref_feats_all = self.prepare_memory_train(ref_x, ref_gt_bboxes)

        # do aggregation
        outputs = []
        for i, _x in enumerate(x):
            # [1, C, H, W]
            n, c, h, w = _x.size()
            # [n, c, h, w] -> [n*h*w, c]
            _x = _x.permute(0, 2, 3, 1).view(-1, c)

            _query = self.filter_with_mask(_x)
            # query = _x[_mask]
            _key = ref_feats_all[i]
            _query_new = self.memories[i](_query, _key)
            _output = self.update_with_query(_x, _query_new)
            # _x[_mask] = query_new

            # remove feats in memory
            self.memories[i].reset()

            # [n*h*w, c] -> [n, c, h, w]
            _output = _output.view(n, h, w, c).permute(0, 3, 1, 2)
            outputs.append(_output)

        return tuple(outputs)
