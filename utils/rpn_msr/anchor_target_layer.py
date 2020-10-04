# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from utils.bbox.bbox import bbox_overlaps

from utils.bbox.bbox_transform import bbox_transform
from utils.rpn_msr.config import Config as cfg
from utils.rpn_msr.generate_anchors import generate_anchors

DEBUG = False


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride=[16, ], anchor_scales=[16, ]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_cls_score: (1, H, W, Ax2) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))  # 生成基本的anchor,一共9个
    _num_anchors = _anchors.shape[0]  # 9个anchor

    if DEBUG:
        print('anchors:')
        print(_anchors)
        print('anchor shapes:')
        print(np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        )))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0
    # map of shape (..., H, W)
    # height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]  # 图像的高宽及通道数
    if DEBUG:
        print("im_info: ", im_info)
    # 在feature-map上定位anchor，并加上delta，得到在实际图像中anchor的真实坐标
    # Algorithm:
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map的高宽

    if DEBUG:
        print('AnchorTargetLayer: height', height, 'width', width)
        print('')
        print('im_size: ({}, {})'.format(im_info[0], im_info[1]))
        print('scale: {}'.format(im_info[2]))
        print('height, width: ({}, {})'.format(height, width))
        print('rpn: gt_boxes.shape', gt_boxes.shape)
        print('rpn: gt_boxes', gt_boxes)

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()  # 生成feature-map和真实image上anchor之间的偏移量
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 9个anchor
    K = shifts.shape[0]  # 50*37，feature-map的宽乘高的大小
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))  # 相当于复制宽高的维度，然后相加
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    # 仅保留那些还在图像内部的anchor，超出图像的都删掉
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    if DEBUG:
        print('total_anchors', total_anchors)
        print('inds_inside', len(inds_inside))

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor
    if DEBUG:
        print('anchors.shape', anchors.shape)

    # 至此，anchor准备好了
    # --------------------------------------------------------------
    # label: 1 is positive, 0 is negative, -1 is dont care
    # (A)
    labels = np.empty((len(inds_inside),), dtype=np.float32)
    labels.fill(-1)  # 初始化label，均为-1

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt), shape is A x G
    # 计算anchor和gt-box的overlap，用来给anchor上标签
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组
    # 存放每一个anchor和每一个gtbox之间的overlap
    argmax_overlaps = overlaps.argmax(axis=1)  # (A)#找到和每一个gtbox，overlap最大的那个anchor
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    gt_argmax_overlaps = overlaps.argmax(axis=0)  # G#找到每个位置上9个anchor中与gtbox，overlap最大的那个
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    if not cfg.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0  # 先给背景上标签，小于0.3overlap的

    # fg label: for each gt, anchor with highest overlap
    labels[gt_argmax_overlaps] = 1  # 每个位置上的9个anchor中overlap最大的认为是前景
    # fg label: above threshold IOU
    labels[max_overlaps >= cfg.RPN_POSITIVE_OVERLAP] = 1  # overlap大于0.7的认为是前景

    if cfg.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    # 对正样本进行采样，如果正样本的数量太多的话
    # 限制正样本的数量不超过128个
    num_fg = int(cfg.RPN_FG_FRACTION * cfg.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
        labels[disable_inds] = -1  # 变为-1

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是256，限制正样本数目最多128，
    # 如果正样本数量小于128，差的那些就用负样本补上，凑齐256个样本
    num_bg = cfg.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    # 至此， 上好标签，开始计算rpn-box的真值
    # --------------------------------------------------------------
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.RPN_BBOX_INSIDE_WEIGHTS)  # 内部权重，前景就给1，其他是0

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    if cfg.RPN_POSITIVE_WEIGHT < 0:  # 暂时使用uniform 权重，也就是正样本是1，负样本是0
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights  # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means:')
        print(means)
        print('stdevs:')
        print(stds)

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)  # 内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)  # 外部权重以0填充

    if DEBUG:
        print('rpn: max max_overlap', np.max(max_overlaps))
        print('rpn: num_positive', np.sum(labels == 1))
        print('rpn: num_negative', np.sum(labels == 0))
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)

    # labels
    labels = labels.reshape((1, height, width, A))  # reshap一下label
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4))  # reshape

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    if DEBUG:
        print("anchor target set")
    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
