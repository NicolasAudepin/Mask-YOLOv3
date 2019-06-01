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

# Validation dataset
dataset_val = BalloonDataset()
dataset_val.load_balloon(balloon_DIR, "val")
dataset_val.prepare()


model = yolo.MaskYOLO(mode="yolo_train",
                      weights_path="/Users/stark/Mask-YOLOv3/Mask-YOLO/model_data/yolo-tiny.h5",
                      config=config,
                      yolo_pretrain_dir=None,
                      yolo_trainable=True)

model.train(dataset_train, dataset_val, num_train=50, batch_size=5, learning_rate=config.LEARNING_RATE)

# image = cv2.imread('/Users/stark/Mask-YOLOv3/Mask-YOLO/datasets/balloon/train/34020010494_e5cb88e1c4_k.jpg')
# image = cv2.cvtColor(cv2.resize(image, (416, 416)), cv2.COLOR_BGR2RGB)
#
# model.infer_yolo(image, '/Users/stark/Mask-YOLOv3/Mask-YOLO/model_data/trained_weights_final.h5')
#



############################################################
#  Training dataset test
############################################################

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# # Load random image and mask.
# image_id = np.random.choice(dataset_train.image_ids)
# image = dataset_train.load_image(image_id)
# mask, class_ids = dataset_train.load_mask(image_id)
# # Compute Bounding box
# bbox = utils.extract_bboxes(mask)
# print(bbox)
# # Display image and additional stats
# print("image_id ", image_id, dataset_train.image_reference(image_id))
#
# # Display image and instances
# visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


# Test Data generators
# train_info = []
# for id in range(0, 50):
#     image, gt_class_ids, gt_boxes, gt_masks = \
#         utils.load_image_gt(dataset_train, config, id,
#                              use_mini_mask=config.USE_MINI_MASK)
#     train_info.append([image, gt_class_ids, gt_boxes, gt_masks])
#
# # print(gt_class_ids) output: [1 1 1 1]
# print(gt_boxes)
#
# val_info = []
# for id in range(0, 6):
#     image, gt_class_ids, gt_boxes, gt_masks = \
#         utils.load_image_gt(dataset_val, config, id,
#                              use_mini_mask=config.USE_MINI_MASK)
#     val_info.append([image, gt_class_ids, gt_boxes, gt_masks])
#
# train_generator = utils.BatchGenerator(train_info, config, mode='training',
#                                                 shuffle=True, jitter=False, norm=True)
#
# val_generator = utils.BatchGenerator(val_info, config, mode='training',
#                                       shuffle=False, jitter=False, norm=False)
#
#
# img = train_generator[0][0][0][0]
# plt.imshow(img.astype('float'))
# gt_masks = train_generator[0][0][5][0]
# gt_boxes = train_generator[0][0][4][0]
# gt_class_ids = train_generator[0][0][3][0]
# # print(gt_boxes.shape)  # (10, 4)
# # print(gt_masks.shape)  # (224, 224, 100) when MAX_GT_INSTANCES = 100
# # print(gt_class_ids.shape)    # (10,)
# visualize.display_instances(img, gt_boxes, gt_masks, gt_class_ids, dataset_train.class_names)
#
# print('ss')
