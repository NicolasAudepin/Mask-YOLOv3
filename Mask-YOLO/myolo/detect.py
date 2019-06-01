
from myolo.yolo import MaskYOLO
from myolo.config import Config
from example.balloon.dataset_balloon import BalloonDataset, BalloonConfig
from PIL import Image

ROOT_DIR = '/Users/stark/Mask-YOLOv3/Mask-YOLO'


config = BalloonConfig()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


if __name__ == '__main__':
    detect_img(MaskYOLO(mode="yolo_detect",
                      weights_path="/Users/stark/Mask-YOLOv3/Mask-YOLO/example/balloon/logs/000/trained_weights_final.h5",
                      config=config))
