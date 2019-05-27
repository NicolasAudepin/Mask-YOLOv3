import datetime
import multiprocessing
import os
import re
import numpy as np
import keras
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from myolo import utils
from myolo.model import tiny_yolo_body, yolo_loss
import matplotlib.pyplot as plt
from myolo import visualize
import cv2


# what's the different between class MaskYOLO(object): and class MaskYOLO:
class MaskYOLO:
    """ Build the overall structure of MaskYOLO class
    which generate bbox and class label on the YOLO side based on that then added with a Mask branch
    Note to myself: all the operations have to be built with Tensor and Layer so as to generate TF Graph
    """

    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    def __init__(self,
                 mode,
                 config,
                 model_dir=None,
                 yolo_pretrain_dir=None,
                 yolo_trainable=True):
        """
        mode: Either "training" or "inference"
        self.config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference', 'yolo']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        # self.weights_path = weights_path
        self.yolo_pretrain_dir = yolo_pretrain_dir
        self.yolo_trainable = yolo_trainable
        # self.keras_model = self.build(mode=mode, self.config=self.config)
        self.keras_model = self.build(mode=mode)
        self.epoch = 0

        # TODO freeze_body need be clear
    def build(self, mode, load_pretrained=True, freeze_body=2):
        '''create the training model, for Tiny YOLOv3'''
        K.clear_session()  # get a new session
        image_input = Input(shape=(None, None, 3))

        h, w = self.config.INPUT_SHAPE

        weights_path = self.model_dir

        # TODO
        anchors = np.array(self.config.ANCHORS)
        num_classes = self.config.NUM_CLASSES
        num_anchors = len(anchors)

        if mode == 'yolo':
            # tiny yolo have two output y1, y2. input image after 32x down sample
            # to get y1, so y1's shape = input.shape // 32
            # the same way to get y2's shape below.
            # y_true = [y1, y2]
            # why anchors box have six anchors but we use three of them
            # yolo have nine anchors we use three of them too
            y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l],
                                   num_anchors // 2, num_classes + 5)) for l in range(2)]

            # model_body input: image_input, output: [y1, y2]
            model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
            print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

            if load_pretrained:
                model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print('Load weights {}.'.format(weights_path))
                if freeze_body in [1, 2]:
                    # Freeze the darknet body or freeze all but 2 output layers.
                    num = (20, len(model_body.layers) - 2)[freeze_body - 1]
                    for i in range(num): model_body.layers[i].trainable = False
                    print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

            model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                                arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
                [*model_body.output, *y_true])
            # model = Model([model_body.input, *y_true], model_loss)
            model = Model([model_body.input, *y_true], [model_loss, *model_body.output])

            return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
	    custom_callbacks: Optional. Add custom callbacks to be called
	        with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        # assert self.mode == "training", "Create model in training mode.", "yolo"

        # Pre-defined layer regular expressions
        # layer_regex = {
        #     # all layers but the backbone
        #     # "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        #     # # From a specific Resnet stage and up
        #     # "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        #     # "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        #     # "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
        #     # All layers
        #     "all": ".*",
        # }
        # if layers in layer_regex.keys():
        #     layers = layer_regex[layers]

        # Data generators
        train_info = []
        for id in range(0, 50):
            image, gt_class_ids, gt_boxes, gt_masks = \
                utils.load_image_gt(train_dataset, self.config, id,
                                     use_mini_mask=self.config.USE_MINI_MASK)
            train_info.append([image, gt_class_ids, gt_boxes, gt_masks])

        val_info = []
        for id in range(0, 10):
            image, gt_class_ids, gt_boxes, gt_masks = \
                utils.load_image_gt(val_dataset, self.config, id,
                                     use_mini_mask=self.config.USE_MINI_MASK)
            val_info.append([image, gt_class_ids, gt_boxes, gt_masks])

        # train_generator = utils.BatchGenerator(train_info, self.config, mode=self.mode,
        #                                         shuffle=True, jitter=False, norm=True)

        # val_generator = utils.BatchGenerator(val_info, self.config, mode=self.mode,
        #                                       shuffle=True, jitter=False, norm=True)

        train_generator = utils.data_generator(train_info, 50, self.config)
        val_generator = utils.data_generator(val_info, 10, self.config)

        # Create log_dir if it does not exist
        if not os.path.exists(self.config.LOG_DIR):
            os.makedirs(self.config.LOG_DIR)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir='./', histogram_freq=0, write_graph=True, write_images=False),
            ModelCheckpoint(self.config.LOG_DIR + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                            monitor='val_loss', save_weights_only=True, save_best_only=True, period=50),
        ]

        # Train with frozen layers first, to get a stable loss.
        # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.

        # Add Losses
        # First, clear previously set losses to avoid duplication

        self.keras_model.compile(optimizer=Adam(lr=1e-3),
                                 loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred}
                                 )

        num_train = 50
        num_val = 12
        batch_size = 5
        self.keras_model.fit_generator(train_generator,
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=val_generator,
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=0,
            callbacks=callbacks)
        self.epoch = max(self.epoch, epochs)

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

    def infer_yolo(self, image, weights_dir, save_path='', display=True):
        """ decode yolo output to boxes, confidence score, class label, with visualization.
        :param image: original input image with the same image shape in self.config. single image. dtype=uint8
        :param display: True for visualizing the yolo result on the input image
        :param save_path
        """
        assert list(image.shape) == self.config.IMAGE_SHAPE
        assert image.dtype == 'uint8'
        assert self.mode == 'yolo'

        # now = datetime.datetime.now()
        # tz = timezone('US/Eastern')
        # fmt = '%b-%d-%H-%M'
        # now = tz.localize(now)

        normed_image = image / 255.  # normalize the image to 0~1

        # form the inputs as model required
        normed_image = np.expand_dims(normed_image, axis=0)
        dummy_y2 = np.zeros((1, 26, 26, 3, 7))
        dummy_y1 = np.zeros((1, 13, 13, 3, 7))
        dummy_y = [dummy_y1, dummy_y2]
        # load weights
        self.load_weights(weights_dir)

        # model predict for single input image
        netout = self.keras_model.predict([normed_image, *dummy_y])
        # print(netout[0])
        #
        # print(netout[1].shape, netout[2].shape)
        # decode network output
        # netout = [loss, y1, y2]
        boxes = utils.decode_one_yolo_output(netout[1:],
                                              anchors=self.config.ANCHORS,
                                              nms_threshold=0.3,  # for shapes dataset this could be big
                                              obj_threshold=0.35,
                                              nb_class=self.config.NUM_CLASSES)

        normed_image = utils.draw_boxes(normed_image[0], boxes, labels=self.config.LABELS)

        plt.imshow(normed_image[:, :, ::-1])
        plt.savefig(save_path + 'InferYOLO.png')

    def detect(self, image, weights_dir, save_path='./img_results/', cs_threshold=0.35, display=True):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert list(image.shape) == self.config.IMAGE_SHAPE
        assert image.dtype == 'uint8'
        assert self.mode == 'inference'

        # now = datetime.datetime.now()
        # tz = timezone('US/Eastern')
        # fmt = '%b-%d-%H-%M'
        # now = tz.localize(now)

        normed_image = image / 255.  # normalize the image to 0~1

        # form the inputs as model required
        normed_image = np.expand_dims(normed_image, axis=0)
        dummy_true_boxes = np.zeros((1, 1, 1, 1, self.config.TRUE_BOX_BUFFER, 4))
        # dummy_true_boxes' shape is (1, 1, 1, 1, 10, 4)

        # load weights
        self.load_weights(weights_dir)

        # model predict for single input image
        self.config.BATCH_SIZE = 1
        yolo_output, detections, myolo_mask = self.keras_model.predict([normed_image, dummy_true_boxes], verbose=0)
        # yolo_output = (1, 7, 7, 5, 7)
        # detections = (1, 245, 6)
        # myolo_mask = (1, 245, 28, 28, 2)
        # 245 = 7*7*5
        # test if detections align with results of yolo_output
        for detection in detections[0]:
            # print("score is ", detection[4])
            if detection[4] >= cs_threshold:
                print(detection)

        # decode network output
        yolo_boxes = utils.decode_one_yolo_output(yolo_output[0],
                                                   anchors=self.config.ANCHORS,
                                                   nms_threshold=0.3,  # for shapes dataset this could be big
                                                   obj_threshold=0.2,
                                                   nb_class=self.self.config.NUM_CLASSES)
        # print(yolo_output)
        # if display:
        #     image = utils.draw_boxes(image, yolo_boxes, labels=self.self.config.LABELS)
        #     plt.imshow(image)
        #     plt.show()

        # Decode masks
        results = []
        boxes, class_ids, scores, full_masks = self.decode_masks(detections, myolo_mask, image.shape)

        top10_indices = np.argsort(scores)[::-1][:10]
        # print(top10_indices)
        # print(boxes[top10_indices])
        # if display:
        #     image = utils.draw_boxes(image, boxes[top10_indices], labels=self.self.config.LABELS)
        #     plt.imshow(image)
        #     plt.show()
        list_to_remove = []
        for index in top10_indices:
            if scores[index] < cs_threshold:
                list_to_remove.append(np.where(top10_indices == index)[0][0])
        removed_indices = np.delete(top10_indices, list_to_remove)

        boxes_temp = boxes[removed_indices]
        class_ids_temp = class_ids[removed_indices]
        # scores = scores[removed_indices]
        # full_masks = full_masks[removed_indices]

        nmb_indices = utils.NMB(boxes_temp, class_ids_temp, removed_indices, self.config.IMAGE_SHAPE, nms_threshold=0.7)
        # bug in here
        nmb_indices = top10_indices
        boxes = np.array([i * 224 for i in boxes[nmb_indices]])
        class_ids = class_ids[nmb_indices]
        scores = scores[nmb_indices]
        full_masks = full_masks[:, :, nmb_indices]

        # for i in range(0, full_masks.shape[-1]):
        #     full_masks[:, :, i] = np.transpose(full_masks[:, :, i])
        #     full_masks[:, :, i] = np.rot90(full_masks[:, :, i], 3)

        results.append({
            "bboxes": boxes,
            "class_ids": class_ids,
            "confidence_scores": scores,
            "full_masks": full_masks,
        })

        save_path += 'InferMaskYOLO-' + now.strftime(fmt) + '.png'

        if display:
            visualize.display_instances(image, boxes, full_masks, class_ids, self.config.LABELS, scores, save_path)

        return results

    def decode_masks(self, detections, myolo_mask, image_shape):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (x1, y1, x2, y2, score, class_id)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [H, W, C] Shape of the image (no resizing or padding for now)

        Returns:
        boxes: [N, (x1, y1, x2, y2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """

        assert len(detections) == 1    # only detect for one image per time
        assert len(myolo_mask) == 1
        assert list(image_shape) == self.config.IMAGE_SHAPE

        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        # zero_ix = np.where(detections[:, 4] == 0)[0]
        # N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        detection = detections[0]
        myolo_mask = myolo_mask[0]
        N = len(detection)

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detection[:N, :4]
        scores = detection[:N, 4]
        class_ids = detection[:N, 5].astype(np.int32)

        masks = myolo_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        # Convert boxes to pixel coordinates on the original image
        # boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # print(image_data.shape)
        # 图像预处理结束，输出图像大小（1, 416， 416， 3）

        # 将图像输入模型进行预测
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        # 预测结束，得到out_boxes, out_scores, out_classes
        # 输出找到了一个物体，下一步使用结果数据在图像上画框，展示结果
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        # print(self.yolo_model.output[0].shape)
        # 最后feature map的通道数是255=（3*（5+80）），每个格子有3个anchor

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymax - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5e-3 * image_h,
                    (0, 255, 0), 1)

    return image


