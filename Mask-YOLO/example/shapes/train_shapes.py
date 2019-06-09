import numpy as np
from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
from myolo import yolo


config = ShapesConfig()
config.display()

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(1000, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# image, gt_class_ids, gt_boxes, gt_masks = mutils.load_image_gt(dataset_train, config, image_id=440, augment=None,
#                                                                augmentation=None,
#                                                                use_mini_mask=config.USE_MINI_MASK)
# config.BATCH_SIZE = 1

model = yolo.MaskYOLO(mode="yolo_train",
                      weights_path="/Users/stark/Mask-YOLOv3/Mask-YOLO/model_data/yolo-tiny.h5",
                      config=config,
                      pretrained=None,
                      yolo_trainable=True)

model.train(dataset_train, dataset_val, num_train=10, batch_size=1, stage1epochs=50, stage2epochs=150)
