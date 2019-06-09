import numpy as np
import math
import scipy
import random
import logging
import tensorflow as tf
from distutils.version import LooseVersion
import skimage.color
import skimage.io
import skimage.transform
import cv2
from keras.utils import Sequence
from functools import reduce
from keras import backend as K
from PIL import Image
import numpy as np
import os
import cv2
import keras
from example.balloon.dataset_balloon import BalloonDataset, BalloonConfig
from myolo import yolo
from myolo import utils
from myolo import visualize

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT_DIR = '/Users/stark/Mask-YOLOv3/Mask-YOLO'

config = BalloonConfig()
# config.display()

balloon_DIR = os.path.join(ROOT_DIR, "datasets/balloon")

# Training dataset
dataset_train = BalloonDataset()
dataset_train.load_balloon(balloon_DIR, "train")
dataset_train.prepare()

num_train = 1
batch_size = 1

# I want test utils.data_generator works or not.
# Data generators
train_info = []
for id in range(num_train):
    image, gt_class_ids, gt_boxes, gt_masks = \
        utils.load_image_gt(dataset_train, config, id,
                            use_mini_mask=config.USE_MINI_MASK)
    # visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids, train_dataset.class_names)
    train_info.append([image, gt_class_ids, gt_boxes, gt_masks])


train_generator = utils.data_generator(train_info, batch_size=batch_size, config=config)

output = train_generator.__next__()
print(output[0][0].shape)
print(output[0][1].shape)
print(output[0][2].shape)
# output1 = tf.convert_to_tensor(output[0][1], np.float32)
# output2 = tf.convert_to_tensor(output[0][2], np.float32)
output = [output[0][1][0], output[0][2][0]]


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)
    # tiny yolo have two output, so two output layers
    # 每个输出的尺度不同，需要不同尺度的anchor_mask
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # 需要首先过滤掉置信度太小的，小于score_threshold的，这里设置好mask
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 使用的内置NMS函数！之前怀疑的NMS出错了，究竟是什么原因？
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''

    # Convert final layer features to bounding box parameters.
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
                                                                anchors, num_classes, input_shape)
    # Scale boxes back to original image shape.
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)

    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3]  # height, width

    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # grid包含grid的位置
    # Tensor("Cast_4:0", shape=(?, ?, 1, 2), dtype=float32) Tensor("Reshape_9:0", shape=(?, ?, ?, 3, 85), dtype=float32) Tensor("truediv_8:0", shape=(?, ?, ?, 3, 2), dtype=float32) Tensor("truediv_9:0", shape=(?, ?, ?, 3, 2), dtype=float32)
    # Tensor("truediv_8:0", shape=(?, ?, ?, 3, 2), dtype=float32) Tensor("truediv_9:0", shape=(?, ?, ?, 3, 2), dtype=float32) Tensor("Sigmoid_4:0", shape=(?, ?, ?, 3, 1), dtype=float32) Tensor("Sigmoid_5:0", shape=(?, ?, ?, 3, 80), dtype=float32)

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    # box_mins, box_maxes. what's that mean?
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


boxes = utils.decode_one_yolo_output(output,
                                     anchors=config.ANCHORS,
                                     nms_threshold=0.3,  # for shapes dataset this could be big
                                     obj_threshold=0.35,
                                     nb_class=config.NUM_CLASSES)

print(boxes[0].label, boxes[0].score)
