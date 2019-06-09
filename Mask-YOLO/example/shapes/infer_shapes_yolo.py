import numpy as np
from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
import myolo.model as modellib
import random
from myolo import utils
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from myolo.yolo import MaskYOLO
from myolo.config import Config
from example.balloon.dataset_balloon import BalloonDataset, BalloonConfig
from PIL import Image


ROOT_DIR = '/Users/stark/Mask-YOLOv3/Mask-YOLO'


config = ShapesConfig()


def detect_img(yolo):

    img = '/Users/stark/Mask-YOLOv3/Mask-YOLO/example/shapes/1006.jpeg'
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        # continue
    else:
        r_image = yolo.detect_image(image)
        r_image.show()
    yolo.close_session()


if __name__ == '__main__':
    detect_img(MaskYOLO(mode="yolo_detect",
                      weights_path="/Users/stark/Mask-YOLOv3/Mask-YOLO/example/shapes/logs/000/trained_weights_stage_1.h5",
                      config=config))





# print(original_image.shape)
# input_image = original_image / 255.
# plt.imshow(input_image)
# plt.show()
# # input_image = np.expand_dims(input_image, axis=0)
# # dummy_true_boxes = np.zeros((1, 1, 1, 1, config.TRUE_BOX_BUFFER, 4))
# #
# # model.detect_for_one()