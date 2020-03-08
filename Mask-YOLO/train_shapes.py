import numpy as np
from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
from myolo import visualize
from myolo import utils
from myolo import yolo


config = ShapesConfig()
config.display()

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

image, gt_class_ids, gt_boxes, gt_masks = utils.load_image_gt(dataset_train, config, image_id=440, augment=None,
                                                               augmentation=None,
                                                               use_mini_mask=config.USE_MINI_MASK)
config.BATCH_SIZE = 1

# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Load random image and mask.
image_id = np.random.choice(dataset_train.image_ids)
image = dataset_train.load_image(image_id)
mask, class_ids = dataset_train.load_mask(image_id)
# Compute Bounding box
bbox = utils.extract_bboxes(mask)
print(bbox)
# Display image and additional stats
print("image_id ", image_id, dataset_train.image_reference(image_id))

# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)

model = yolo.MaskYOLO(mode="yolo_train",
                      weights_path="/Users/stark/Mask-YOLOv3/Mask-YOLO/model_data/yolo-tiny.h5",
                      config=config,
                      pretrained=None,
                      yolo_trainable=True)

model.train(dataset_train, dataset_val, num_train=10, batch_size=1, stage1epochs=50, stage2epochs=150)
