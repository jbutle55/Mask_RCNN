import argparse
import sys
import os
import numpy as np
import cv2
import datetime
import skimage
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

from samples.coco.coco import CocoDataset

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2014"

MODEL_DIR = ''


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = False


class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 6)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(
            np.array(boxes), np.arange(N), 0.4)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes


class WESCAMConfig(Config):
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    USE_MINI_MASK = False


class CocoConfig(Config):
    """Configuration for training on MS COCO.
        Derives from the base Config class and overrides values specific
        to the COCO dataset.
        """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


def main(args):
    command = args.command
    weights = args.weights  # mask_rcnn_coco.h5'
    dataset = args.dataset
    logs = args.logs
    video = args.video
    output = args.output
    config_arg = args.config

    # Setup configs
    if config_arg == 'coco':
        config = CocoConfig()
    elif config_arg == 'shapes':
        config = ShapesConfig()
    elif config_arg == 'wescam':
        config = WESCAMConfig
    config.display()


    # Create model
    if command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs)

    # Select weights file to load
    if model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    model.keras_model.summary()

    # Train or evaluate
    if command == "train":
        if config == 'coco:'
            # Training dataset. Use the training set and 35K from the
            # validation set, as as in the Mask RCNN paper.
            dataset_train = CocoDataset()
            dataset_train.load_coco(dataset, "train", year=year, auto_download=False)
            dataset_train.load_coco(dataset, "valminusminival", year=DEFAULT_DATASET_YEAR, auto_download=False)
            dataset_train.prepare()

            # Validation dataset
            dataset_val = CocoDataset()
            val_type = "val" if year in '2017' else "minival"
            dataset_val.load_coco(dataset, val_type, year=year, auto_download=download)
            dataset_val.prepare()

            # Image Augmentation
            # Right/Left flip 50% of the time
            augmentation = imgaug.augmenters.Fliplr(0.5)

            # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+',
                    augmentation=augmentation)

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all',
                    augmentation=augmentation)

        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(command))

    if command == 'detect_vid':
        vcapture = cv2.VideoCapture(args.video)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])

                # Draw Bboxes
                for count, box in enumerate(r['rois']):
                    # Shape (y1, x1, y2, x2, class_id)
                    splash = cv2.rectangle(splash, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                    splash = cv2.addText(splash, r['class_ids'][count], (box[3], box[2]), 2)

                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

    if command == 'evaluate':
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=config,
                                  model_dir=logs)

        # Get path to saved weights
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, "mask_rcnn_shapes")
        # model_path = model.find_last()

        # Load trained weights
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        confidence_thresholds = [0.5]
        tp_rates = {}
        fp_rates = {}

        all_fp_rates = []
        all_tp_rates = []

        # Compute ROCs for above range of thresholds
        # Compute one for each class vs. the other classes
        for index, conf in enumerate(confidence_thresholds):

            tp_of_img = []
            fp_of_img = []

            all_classes = []

            print('Creating model with confidence threshold: {}'.format(conf))

            config.DETECTION_MIN_CONFIDENCE = conf


            # Recreate the model in inference mode
            model = modellib.MaskRCNN(mode="inference",
                                      config=config,
                                      model_dir=MODEL_DIR)
            # Load trained weights
            print("Loading weights from ", model_path)
            model.load_weights(model_path, by_name=True)

            image_ids = np.random.choice(dataset_val.image_ids, 10)
            for image_id in image_ids:
                # Load image and ground truth data
                image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                    modellib.load_image_gt(dataset_val, config,
                                           image_id)
                molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
                # Run object detection
                results = model.detect([image], verbose=0)
                r = results[0]

                # Detect returns:
                # "rois" []
                # "class_ids" [N]
                # "scores" [N]

                classes = list(set(r['class_ids']))  # All unique class ids
                for c in classes:
                    if c not in all_classes:
                        all_classes.append(c)

                # Need TPR and FPR rates for each class versus the other classes
                # Recall == TPR
                #_, _, tpr, _ = utils.compute_ap_indiv_class(gt_bbox, gt_class_id, gt_mask,
                #                                            r["rois"], r["class_ids"], r["scores"], r['masks'])

                # Display Test Image
                original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                    modellib.load_image_gt(dataset_val, config,
                                           image_id)

                visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                            dataset_train.class_names, figsize=(8, 8))

                tpr = utils.compute_ap_indiv_class(gt_bbox, gt_class_id, gt_mask,
                                                            r["rois"], r["class_ids"], r["scores"], r['masks'])
                total_fpr = utils.compute_fpr_indiv_class(gt_bbox, gt_class_id, gt_mask,
                                                            r["rois"], r["class_ids"], r["scores"], r['masks'])

                tp_of_img.append(tpr)
                fp_of_img.append(total_fpr)

            # Need to get average TPR and FPR for number of images used
            for c in all_classes:
                tp_s = 0
                for item in tp_of_img:
                    if c in item.keys():
                        tp_s += item[c]
                    else:
                        tp_s += 0
                tp_rates[c] = tp_s / len(image_ids)

            for c in all_classes:
                fp_s = 0
                for item in fp_of_img:
                    if c in item.keys():
                        fp_s += item[c]
                    else:
                        fp_s += 0
                fp_rates[c] = fp_s / len(image_ids)

            all_fp_rates.append(fp_rates)
            all_tp_rates.append(tp_rates)

        # Plot roc curves
        utils.compute_roc_curve(all_tp_rates, all_fp_rates, confidence_thresholds)

    return


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', required=True)
    parser.add_argument('dataset', required=False)
    parser.add_argument('logs', required=False)
    parser.add_argument('video', required=False)
    parser.add_argument('output', default='output')
    parser.add_argument('config', default='coco')
    parser.add_argument('weights', default='mask_rcnn_coco.h5')
    arguments = parser.parse_args()
    main(arguments)

