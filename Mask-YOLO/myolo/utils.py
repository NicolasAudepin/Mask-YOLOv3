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

def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


############################################################
#  Dataset
############################################################

class Dataset(object):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        logging.warning("You are using the default load_mask(), maybe you need to define your own one.")
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_image(image, net_image_shape):
    """Resizes an image keeping the aspect ratio changed.

    Returns:
    image: the resized image
    scale: The scale factor used to resize the image
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (x1, y1, x2, y2) and default scale == 1.
    h, w = image.shape[:2]
    scale = [1, 1]

    # Scale
    scale[0], scale[1] = net_image_shape[0] / h, net_image_shape[1] / w

    # Resize image using bilinear interpolation
    if scale != [1, 1]:
        image = resize(image, (round(h * scale[0]), round(w * scale[1])),
                       preserve_range=True)

    return image.astype(image_dtype), scale


def resize_mask(mask, scale):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    mask = scipy.ndimage.zoom(mask, zoom=[scale[0], scale[1], 1], order=0)
    # if crop is not None:
    #     y, x, h, w = crop
    #     mask = mask[y:y + h, x:x + w]
    # else:
    #     mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        x1, y1, x2, y2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (x1, y1, x2, y2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    network_image_shape = config.IMAGE_SHAPE
    original_shape = image.shape
    image, scale = resize_image(image, network_image_shape)
    mask = resize_mask(mask, scale)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    return image, class_ids, bbox, mask


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


class BatchGenerator(Sequence):
    def __init__(self,
                 all_info,
                 config,
                 mode,
                 shuffle=True,
                 jitter=False,
                 norm=False):

        # self.generator = None
        self.config = config
        self.mode = mode
        self.all_info = all_info
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        assert mode in ['yolo', 'training']

        self.anchors = [BoundBox(0, 0, self.config.ANCHORS[2 * i], self.config.ANCHORS[2 * i + 1]) for i in
                        range(int(len(self.config.ANCHORS) // 2))]

        if shuffle:
            np.random.shuffle(self.all_info)   # image, gt_class_ids, gt_boxes, gt_masks
        # self.images = [item[0] for item in all_info]

    def __len__(self):
        return int(np.ceil(float(len(self.all_info)) / self.config.BATCH_SIZE))

    def num_classes(self):
        return self.config.NUM_CLASSES

    def size(self):
        return len(self.all_info)

    def load_image(self, i):
        return cv2.imread(self.all_info[i][0])

    def __getitem__(self, idx):
        l_bound = idx * self.config.BATCH_SIZE
        r_bound = (idx + 1) * self.config.BATCH_SIZE

        if r_bound > len(self.all_info):
            r_bound = len(self.all_info)
            l_bound = max(0, r_bound - self.config.BATCH_SIZE)

        instance_count = 0

        batch_images = np.zeros((r_bound - l_bound,) + (224, 224, 3), dtype=np.float32)
        batch_yolo_target = np.zeros((r_bound - l_bound, self.config.GRID_H, self.config.GRID_W,
                                      self.config.N_BOX, 4 + 1 + self.config.NUM_CLASSES))

        batch_yolo_true_boxes = np.zeros((r_bound - l_bound, 1, 1, 1, self.config.TRUE_BOX_BUFFER, 4))

        batch_gt_class_ids = np.zeros((r_bound - l_bound, self.config.TRUE_BOX_BUFFER), dtype=np.int32)
        batch_gt_boxes = np.zeros((r_bound - l_bound, self.config.TRUE_BOX_BUFFER, 4), dtype=np.int32)
        batch_gt_masks = np.zeros((r_bound - l_bound, 224, 224,
                                   self.config.MAX_GT_INSTANCES), dtype=np.bool)

        # x_batch = np.zeros((r_bound - l_bound, self.config.IMAGE_SHAPE[1], self.config.IMAGE_SHAPE[0], 3))  # input images
        # b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config.TRUE_BOX_BUFFER,
        #                     4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        # y_batch = np.zeros((r_bound - l_bound, self.config.GRID_H, self.config.GRID_W, self.config.N_BOX,
        #                     4 + 1 + self.config.NUM_CLASSES))  # desired network output

        for train_instance in self.all_info[l_bound:r_bound]:

            image = train_instance[0]
            gt_class_ids = train_instance[1]
            gt_boxes = train_instance[2]
            gt_masks = train_instance[3]

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.TRUE_BOX_BUFFER:
                print('find instances more than ' + str(self.config.TRUE_BOX_BUFFER) + ' in an image')
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.TRUE_BOX_BUFFER, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            ### YOLO
            true_box_index = 0
            for i in range(0, gt_boxes.shape[0]):
                # gt_boxes: [instance, (x1, y1, x2, y2)]
                xmin = gt_boxes[i][0]
                ymin = gt_boxes[i][1]
                xmax = gt_boxes[i][2]
                ymax = gt_boxes[i][3]

                center_x = .5 * (xmin + xmax)
                center_x = center_x / (float(self.config.IMAGE_SHAPE[0]) / self.config.GRID_W)
                center_y = .5 * (ymin + ymax)
                center_y = center_y / (float(self.config.IMAGE_SHAPE[1]) / self.config.GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < self.config.GRID_W and grid_y < self.config.GRID_H:
                    obj_indx = gt_class_ids[i]

                    center_w = (xmax - xmin) / (float(self.config.IMAGE_SHAPE[0]) / self.config.GRID_W)
                    center_h = (ymax - ymin) / (float(self.config.IMAGE_SHAPE[1]) / self.config.GRID_H)

                    yolo_box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           center_w,
                                           center_h)

                    for j in range(0, len(self.anchors)):
                        anchor = self.anchors[j]
                        iou = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = j
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    batch_yolo_target[instance_count, grid_y, grid_x, best_anchor, 0:4] = yolo_box
                    batch_yolo_target[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                    batch_yolo_target[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1
                    # assign the true box to b_batch
                    batch_yolo_true_boxes[instance_count, 0, 0, 0, true_box_index] = yolo_box

                    true_box_index += 1
                    true_box_index = true_box_index % self.config.TRUE_BOX_BUFFER

            # assign input image to x_batch
            if self.norm:
                batch_images[instance_count] = image / 255.
            else:
                # plot image and bounding boxes for sanity check
                img = image[:, :, ::-1].astype(np.uint8).copy()
                for i in range(0, gt_boxes.shape[0]):
                    # if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    if train_instance[2][i][2] > train_instance[2][i][0] \
                            and train_instance[2][i][3] > train_instance[2][i][1]:
                        cv2.rectangle(img, (gt_boxes[i][0], gt_boxes[i][1]),
                                      (gt_boxes[i][2], gt_boxes[i][3]),
                                      (255, 0, 0), 2)
                        cv2.putText(img, str(gt_class_ids[i]),
                                    (gt_boxes[i][0] + 2, gt_boxes[i][1] + 12),
                                    0, 1.2e-3 * image.shape[0],
                                    (0, 255, 0), 1)

                batch_images[instance_count] = img

            batch_gt_class_ids[instance_count, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[instance_count, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[instance_count, :, :, :gt_masks.shape[-1]] = gt_masks

            # increase instance counter in current batch
            instance_count += 1

        if self.mode == 'yolo':
            inputs = [batch_images, batch_yolo_true_boxes, batch_yolo_target]
            outputs = []
        elif self.mode == 'training':
            inputs = [batch_images, batch_yolo_true_boxes, batch_yolo_target,
                      batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            outputs = []
        else:
            raise NotImplementedError

        # return [x_batch, b_batch], y_batch
        return inputs, outputs
