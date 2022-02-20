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
                 strides,
                 before_fpn
                 ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.strides = strides
        self.before_fpn = before_fpn

        # add memory modules for every level
        self.memories = nn.ModuleList()
        for _in_channels in self.in_channels:
            _memory = MemoryBank(_in_channels)
            self.memories.append(_memory)

    def get_ref_feats_from_gtbboxes_single_level_train(self, x, gt_bboxes, stride):
        """
        save pixels within detected boxes into memory
        :param x: [N, C, H, W]
        :param gt_bboxes: [n, 5(ind, x, y, x, y)], e.g.,
            tensor([[  0.0000, 329.8406, 247.6415, 591.7313, 428.7736],
                    [  1.0000, 305.7750, 247.6415, 580.4062, 431.6038]], device='cuda:0')
        :param stride: stride of the current level
        :return
        """
        n, c, _, w = x.size()
        ref_obj_irr_list = []
        ref_obj_list = []
        for i, _x in enumerate(x):
            # [C, H, W] -> [H*W, C]
            _x = _x.view(c, -1).permute(1, 0).contiguous()

            # get bboxes of this image
            ind = gt_bboxes[:, 0] == i
            _bboxes = gt_bboxes[ind]     # [n0, 4]
            # no object
            if len(_bboxes) == 0:
                # obj irr pixels
                obj_irr_feats = self.get_obj_irr_pixels(_x)
                OBJ_PIXEL_NUM = 300
                obj_irr_feats = obj_irr_feats[:OBJ_PIXEL_NUM]
                ref_obj_irr_list.append(obj_irr_feats)
                # memory.update(self.get_obj_irr_pixels(_x))
            # have objects
            else:
                _bboxes = _bboxes[:, 1:]
                boxes = torch.div(_bboxes, stride).int()
                for box in boxes:
                    # 1. map pixels in box to new index on x_box [H*W, C]
                    # box [x1, y1, x2, y2] -> [ind_1, ind_2, ind_3, ... ]
                    inds = sorted(self.box_to_inds_list(box, w))
                    inds = np.asarray(inds)
                    # save part obj
                    PIXEL_NUM = 50
                    if len(inds) > PIXEL_NUM:
                        inds = np.random.choice(inds, PIXEL_NUM, replace=False)
                    ref_obj_list.append(_x[inds])
                    # memory.update(_x[inds])
        ref_all = ref_obj_list + ref_obj_irr_list
        ref_all = torch.cat(ref_all, dim=0)

        # objects are too small
        if len(ref_all) == 0:
            # obj irr pixels
            obj_irr_feats = self.get_obj_irr_pixels(_x)
            PIXEL_NUM = 10
            obj_irr_feats = obj_irr_feats[:PIXEL_NUM]
            return obj_irr_feats

        # max num of feats is set to 1000
        return ref_all[:1000]

    @staticmethod
    def box_to_inds_list(box, w):
        inds = []
        for x_i in range(box[0], box[2]):
            for y_j in range(box[1], box[3]):
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
        for lvl in range(len(ref_x)):
            _ref_x = ref_x[lvl]
            _device = _ref_x.device
            _ref_feats = self.get_ref_feats_from_gtbboxes_single_level_train(
                _ref_x.cpu().clone(), ref_gt_bboxes[0].cpu().clone(), self.strides[lvl]
            )
            ref_feats_all.append(_ref_feats.to(_device))
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
        for lvl in range(len(inputs)):
            x.append(inputs[lvl][[0]])
            ref_x.append(inputs[lvl][1:])

        # save ref feats to all levels of memory
        if len(ref_gt_bboxes[0]) < 1:
            print(len(ref_gt_bboxes))
        ref_feats_all = self.prepare_memory_train(ref_x, ref_gt_bboxes)

        # do aggregation
        outputs = []
        for lvl, _x in enumerate(x):
            # [1, C, H, W]
            n, c, h, w = _x.size()
            # [n, c, h, w] -> [n*h*w, c]
            _x = _x.permute(0, 2, 3, 1).view(-1, c).contiguous()

            _query = self.filter_with_mask(_x)
            # query = _x[_mask]
            _key = ref_feats_all[lvl]
            if len(_key) == 0:
                print(_key.shape)
            _query_new = self.memories[lvl](_query, _key)
            _output = self.update_with_query(_x, _query_new)
            # _x[_mask] = query_new

            # [n*h*w, c] -> [n, c, h, w]
            _output = _output.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
            outputs.append(_output)

        return tuple(outputs)

    def get_feats_inside_bboxes_single_level(self, x, bboxes, stride):
        """
        save pixels within detected boxes into memory
        :param x: [N, C, H, W]
        :param bboxes: list of ndarray N x [30, 5]
        :param stride: stride of the current level
        :return
        """
        assert len(x) == len(bboxes)

        n, c, _, w = x.size()
        feats_list = []
        for i, _x in enumerate(x):
            ref_obj_list = []

            # [C, H, W] -> [H*W, C]
            _x = _x.view(c, -1).permute(1, 0).contiguous()
            # get bboxes of this image
            _bboxes = bboxes[i]     # 30 x [n, 5]
            for cls_ind in range(len(_bboxes)):
                _bboxes_of_cls = _bboxes[cls_ind]
                # get feats inside high-quality bboxes
                for box_with_score in _bboxes_of_cls:
                    if box_with_score[-1] > 0.8:
                        box = box_with_score[:4]
                        box = (box / stride).astype(int).tolist()
                        inds = sorted(self.box_to_inds_list(box, w))
                        # inds = np.asarray(inds)
                        # save part obj
                        PIXEL_NUM = 300
                        if len(inds) > PIXEL_NUM:
                            inds = np.random.choice(inds, PIXEL_NUM, replace=False)
                        ref_obj_list.append(_x[inds])

            # no high-quality bbox
            if len(ref_obj_list) == 0 or len(torch.cat(ref_obj_list, dim=0)):
                # obj irr pixels
                obj_irr_feats = self.get_obj_irr_pixels(_x)
                PIXEL_NUM = 50
                obj_feats = obj_irr_feats[:PIXEL_NUM]
            else:
                obj_feats = torch.cat(ref_obj_list, dim=0)

            # max num of feats is set to 1000
            feats_list.append(
                obj_feats[:1000]
            )

        return torch.cat(feats_list, dim=0)

    def write_operation(self, x, bboxes):
        """
        Write features inside bboxes into memory
        :param x: list of feature maps
        :param bboxes: list of bboxes
        :return:
        """
        assert len(x) == len(self.memories)     # number of levels
        assert len(x[0]) == len(bboxes)         # number of frames

        # write for every level
        for lvl in range(len(x)):
            _x = x[lvl]
            _device = _x.device
            _feats = self.get_feats_inside_bboxes_single_level(
                _x.cpu(), bboxes, self.strides[lvl]
            )
            # update memory
            self.memories[lvl].update(_feats.to(_device))
        return

    def forward_test(self, x):
        assert len(x) == len(self.memories)

        # do aggregation
        outputs = []
        for i, _x in enumerate(x):
            # [1, C, H, W]
            n, c, h, w = _x.size()
            # [n, c, h, w] -> [n*h*w, c]
            _x = _x.permute(0, 2, 3, 1).view(-1, c).contiguous()

            _query = self.filter_with_mask(_x)
            # query = _x[_mask]
            _key = self.memories[i].sample()
            _query_new = self.memories[i](_query, _key)
            _output = self.update_with_query(_x, _query_new)
            # _x[_mask] = query_new

            # [n*h*w, c] -> [n, c, h, w]
            _output = _output.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()
            outputs.append(_output)

        return tuple(outputs)

    def reset(self):
        for lvl in range(len(self.memories)):
            self.memories[lvl].reset()
