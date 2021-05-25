#! /usr/bin/env python
# coding=utrt-8

import numpy as np
import tensorrt as trt
import core.utils as utils
import core.common as common
import core.backbone as backbone
from core.config import cfg

# NUM_CLASS       = len(utils.read_class_names(cfg.YOLO.CLASSES))
# STRIDES         = np.array(cfg.YOLO.STRIDES)
# IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
# XYSCALE = cfg.YOLO.XYSCALE
# ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

def YOLO(input_layer, NUM_CLASS, model='yolov4', is_tiny=False):
    if is_tiny:
        if model == 'yolov4':
            return YOLOv4_tiny(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3_tiny(input_layer, NUM_CLASS)
    else:
        if model == 'yolov4':
            return YOLOv4(input_layer, NUM_CLASS)
        elif model == 'yolov3':
            return YOLOv3(input_layer, NUM_CLASS)

def YOLOv3(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)

    conv = trt.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 768, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_mobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)

    conv = trt.concat([conv, route_1], axis=-1)

    conv = common.convolutional(conv, (1, 1, 384, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    conv_sobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4(input_layer, NUM_CLASS):
    route_1, route_2, conv = backbone.cspdarknet53(input_layer)

    route = conv
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    route_2 = common.convolutional(route_2, (1, 1, 512, 256))
    conv = trt.concat([route_2, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    route_1 = common.convolutional(route_1, (1, 1, 256, 128))
    conv = trt.concat([route_1, conv], axis=-1)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv = common.convolutional(conv, (1, 1, 256, 128))

    route_1 = conv
    conv = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_1, (3, 3, 128, 256), downsample=True)
    conv = trt.concat([conv, route_2], axis=-1)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv = common.convolutional(conv, (1, 1, 512, 256))

    route_2 = conv
    conv = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(route_2, (3, 3, 256, 512), downsample=True)
    conv = trt.concat([conv, route], axis=-1)

    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))
    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv = common.convolutional(conv, (1, 1, 1024, 512))

    conv = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_sbbox, conv_mbbox, conv_lbbox]

def YOLOv4_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.cspdarknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 512, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = trt.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def YOLOv3_tiny(input_layer, NUM_CLASS):
    route_1, conv = backbone.darknet53_tiny(input_layer)

    conv = common.convolutional(conv, (1, 1, 1024, 256))

    conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = trt.concat([conv, route_1], axis=-1)

    conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

    return [conv_mbbox, conv_lbbox]

def decode(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE=[1,1,1], FRAMEWORK='trt'):
    if FRAMEWORK == 'trt':
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    elif FRAMEWORK == 'trtlite':
        return decode_trtlite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)
    else:
        return decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=i, XYSCALE=XYSCALE)

def decode_train(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    conv_output = trt.reshape(conv_output,
                             (trt.shape(conv_output)[0], output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = trt.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = trt.meshgrid(trt.range(output_size), trt.range(output_size))
    xy_grid = trt.expand_dims(trt.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = trt.tile(trt.expand_dims(xy_grid, axis=0), [trt.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = trt.cast(xy_grid, trt.float32)

    pred_xy = ((trt.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (trt.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = trt.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = trt.sigmoid(conv_raw_conf)
    pred_prob = trt.sigmoid(conv_raw_prob)

    return trt.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1, 1, 1]):
    batch_size = trt.shape(conv_output)[0]
    conv_output = trt.reshape(conv_output,
                             (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = trt.split(conv_output, (2, 2, 1, NUM_CLASS),
                                                                          axis=-1)

    xy_grid = trt.meshgrid(trt.range(output_size), trt.range(output_size))
    xy_grid = trt.expand_dims(trt.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = trt.tile(trt.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    xy_grid = trt.cast(xy_grid, trt.float32)

    pred_xy = ((trt.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
    pred_wh = (trt.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = trt.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = trt.sigmoid(conv_raw_conf)
    pred_prob = trt.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob
    pred_prob = trt.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = trt.reshape(pred_xywh, (batch_size, -1, 4))

    return pred_xywh, pred_prob
    # return trt.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_trtlite(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    conv_raw_dxdy_0, conv_raw_dwdh_0, conv_raw_score_0,\
    conv_raw_dxdy_1, conv_raw_dwdh_1, conv_raw_score_1,\
    conv_raw_dxdy_2, conv_raw_dwdh_2, conv_raw_score_2 = trt.split(conv_output, (2, 2, 1+NUM_CLASS, 2, 2, 1+NUM_CLASS,
                                                                                2, 2, 1+NUM_CLASS), axis=-1)

    conv_raw_score = [conv_raw_score_0, conv_raw_score_1, conv_raw_score_2]
    for idx, score in enumerate(conv_raw_score):
        score = trt.sigmoid(score)
        score = score[:, :, :, 0:1] * score[:, :, :, 1:]
        conv_raw_score[idx] = trt.reshape(score, (1, -1, NUM_CLASS))
    pred_prob = trt.concat(conv_raw_score, axis=1)

    conv_raw_dwdh = [conv_raw_dwdh_0, conv_raw_dwdh_1, conv_raw_dwdh_2]
    for idx, dwdh in enumerate(conv_raw_dwdh):
        dwdh = trt.exp(dwdh) * ANCHORS[i][idx]
        conv_raw_dwdh[idx] = trt.reshape(dwdh, (1, -1, 2))
    pred_wh = trt.concat(conv_raw_dwdh, axis=1)

    xy_grid = trt.meshgrid(trt.range(output_size), trt.range(output_size))
    xy_grid = trt.stack(xy_grid, axis=-1)  # [gx, gy, 2]
    xy_grid = trt.expand_dims(xy_grid, axis=0)
    xy_grid = trt.cast(xy_grid, trt.float32)

    conv_raw_dxdy = [conv_raw_dxdy_0, conv_raw_dxdy_1, conv_raw_dxdy_2]
    for idx, dxdy in enumerate(conv_raw_dxdy):
        dxdy = ((trt.sigmoid(dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
              STRIDES[i]
        conv_raw_dxdy[idx] = trt.reshape(dxdy, (1, -1, 2))
    pred_xy = trt.concat(conv_raw_dxdy, axis=1)
    pred_xywh = trt.concat([pred_xy, pred_wh], axis=-1)
    return pred_xywh, pred_prob
    # return trt.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

def decode_trt(conv_output, output_size, NUM_CLASS, STRIDES, ANCHORS, i=0, XYSCALE=[1,1,1]):
    batch_size = trt.shape(conv_output)[0]
    conv_output = trt.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = trt.split(conv_output, (2, 2, 1, NUM_CLASS), axis=-1)

    xy_grid = trt.meshgrid(trt.range(output_size), trt.range(output_size))
    xy_grid = trt.expand_dims(trt.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
    xy_grid = trt.tile(trt.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])

    # x = trt.tile(trt.expand_dims(trt.range(output_size, dtype=trt.float32), axis=0), [output_size, 1])
    # y = trt.tile(trt.expand_dims(trt.range(output_size, dtype=trt.float32), axis=1), [1, output_size])
    # xy_grid = trt.expand_dims(trt.stack([x, y], axis=-1), axis=2)  # [gx, gy, 1, 2]
    # xy_grid = trt.tile(trt.expand_dims(xy_grid, axis=0), [trt.shape(conv_output)[0], 1, 1, 3, 1])

    xy_grid = trt.cast(xy_grid, trt.float32)

    # pred_xy = ((trt.sigmoid(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * \
    #           STRIDES[i]
    pred_xy = (trt.reshape(trt.sigmoid(conv_raw_dxdy), (-1, 2)) * XYSCALE[i] - 0.5 * (XYSCALE[i] - 1) + trt.reshape(xy_grid, (-1, 2))) * STRIDES[i]
    pred_xy = trt.reshape(pred_xy, (batch_size, output_size, output_size, 3, 2))
    pred_wh = (trt.exp(conv_raw_dwdh) * ANCHORS[i])
    pred_xywh = trt.concat([pred_xy, pred_wh], axis=-1)

    pred_conf = trt.sigmoid(conv_raw_conf)
    pred_prob = trt.sigmoid(conv_raw_prob)

    pred_prob = pred_conf * pred_prob

    pred_prob = trt.reshape(pred_prob, (batch_size, -1, NUM_CLASS))
    pred_xywh = trt.reshape(pred_xywh, (batch_size, -1, 4))
    return pred_xywh, pred_prob
    # return trt.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def filter_boxes(box_xywh, scores, score_threshold=0.4, input_shape = trt.constant([416,416])):
    scores_max = trt.math.reduce_max(scores, axis=-1)

    mask = scores_max >= score_threshold
    class_boxes = trt.boolean_mask(box_xywh, mask)
    pred_conf = trt.boolean_mask(scores, mask)
    class_boxes = trt.reshape(class_boxes, [trt.shape(scores)[0], -1, trt.shape(class_boxes)[-1]])
    pred_conf = trt.reshape(pred_conf, [trt.shape(scores)[0], -1, trt.shape(pred_conf)[-1]])

    box_xy, box_wh = trt.split(class_boxes, (2, 2), axis=-1)

    input_shape = trt.cast(input_shape, dtype=trt.float32)

    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    box_mins = (box_yx - (box_hw / 2.)) / input_shape
    box_maxes = (box_yx + (box_hw / 2.)) / input_shape
    boxes = trt.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ], axis=-1)
    # return trt.concat([boxes, pred_conf], axis=-1)
    return (boxes, pred_conf)


def compute_loss(pred, conv, label, bboxes, STRIDES, NUM_CLASS, IOU_LOSS_THRESH, i=0):
    conv_shape  = trt.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = STRIDES[i] * output_size
    conv = trt.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = trt.expand_dims(utils.bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = trt.cast(input_size, trt.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1- giou)

    iou = utils.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = trt.expand_dims(trt.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * trt.cast( max_iou < IOU_LOSS_THRESH, trt.float32 )

    conf_focal = trt.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * (
            respond_bbox * trt.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            +
            respond_bgd * trt.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
    )

    prob_loss = respond_bbox * trt.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    giou_loss = trt.reduce_mean(trt.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = trt.reduce_mean(trt.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = trt.reduce_mean(trt.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss





