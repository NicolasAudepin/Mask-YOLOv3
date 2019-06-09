
from myolo.yolo import MaskYOLO
from myolo.config import Config
from example.balloon.dataset_balloon import BalloonDataset, BalloonConfig
from PIL import Image

ROOT_DIR = '/Users/stark/Mask-YOLOv3/Mask-YOLO'


config = BalloonConfig()


def detect_img(yolo):
    # while True:
    # img = input('Input image filename:')
    img = '/Users/stark/Mask-YOLOv3/Mask-YOLO/datasets/balloon/train/34020010494_e5cb88e1c4_k.jpg'
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
                      weights_path="/Users/stark/Mask-YOLOv3/Mask-YOLO/example/balloon/logs/000/weights_final.h5",
                      config=config))
