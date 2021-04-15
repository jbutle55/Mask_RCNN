import argparse
import sys
import os
import numpy as np
import cv2
import json
import skimage
import imgaug
import random
import math
from tensorflow.keras.models import Model

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
    NAME = "wescam"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 80 + 1  # background + 3 shapes

    DETECTION_MIN_CONFIDENCE = 0.1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 1024
    #IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 5

    USE_MINI_MASK = False

    CLASS_DICT = {0: u'__background__',
                  1: u'person',
                  2: u'bicycle',
                  3: u'car',
                  4: u'motorcycle',
                  5: u'airplane',
                  6: u'bus',
                  7: u'train',
                  8: u'truck',
                  9: u'boat',
                  10: u'traffic light',
                  11: u'fire hydrant',
                  12: u'stop sign',
                  13: u'parking meter',
                  14: u'bench',
                  15: u'bird',
                  16: u'cat',
                  17: u'dog',
                  18: u'horse',
                  19: u'sheep',
                  20: u'cow',
                  21: u'elephant',
                  22: u'bear',
                  23: u'zebra',
                  24: u'giraffe',
                  25: u'backpack',
                  26: u'umbrella',
                  27: u'handbag',
                  28: u'tie',
                  29: u'suitcase',
                  30: u'frisbee',
                  31: u'skis',
                  32: u'snowboard',
                  33: u'sports ball',
                  34: u'kite',
                  35: u'baseball bat',
                  36: u'baseball glove',
                  37: u'skateboard',
                  38: u'surfboard',
                  39: u'tennis racket',
                  40: u'bottle',
                  41: u'wine glass',
                  42: u'cup',
                  43: u'fork',
                  44: u'knife',
                  45: u'spoon',
                  46: u'bowl',
                  47: u'banana',
                  48: u'apple',
                  49: u'sandwich',
                  50: u'orange',
                  51: u'broccoli',
                  52: u'carrot',
                  53: u'hot dog',
                  54: u'pizza',
                  55: u'donut',
                  56: u'cake',
                  57: u'chair',
                  58: u'couch',
                  59: u'potted plant',
                  60: u'bed',
                  61: u'dining table',
                  62: u'toilet',
                  63: u'tv',
                  64: u'laptop',
                  65: u'mouse',
                  66: u'remote',
                  67: u'keyboard',
                  68: u'cell phone',
                  69: u'microwave',
                  70: u'oven',
                  71: u'toaster',
                  72: u'sink',
                  73: u'refrigerator',
                  74: u'book',
                  75: u'clock',
                  76: u'vase',
                  77: u'scissors',
                  78: u'teddy bear',
                  79: u'hair drier',
                  80: u'toothbrush'}


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


class CocoDataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


class StanfordDataset(utils.Dataset):
    def load_stanford(self):
        self.add_class("stanford", 1, "car")

        dataset_dir = '/home/justin/Data/Stanford/sdd/Annotations_json/nexus_video8_.json'

        image_dir = '/home/justin/Data/Stanford/sdd/images'

        annotations = json.load(open(dataset_dir))
        annotations = list(annotations.values())  # don't need the dict keys

        print(annotations[1][0])

        image_base = annotations[1][0]['file_name'][:-4]

        width = annotations[1][0]['width']
        height = annotations[1][0]['height']

        print(f'image base: {image_base}')

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations[2]]

        # Add images
        for a in annotations:

            id = a['image_id']
            category = a['category_id']

            if category != 1:
                # Only include 1 class
                continue
            bbox = a['bbox']

            filepath = os.path.join(image_dir, f'{image_base}{id}.jpg')
            print(f'file path: {filepath}')
            polygons = bbox


            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            # if type(a['regions']) is dict:
            #     polygons = [r['shape_attributes'] for r in a['regions'].values()]
            # else:
            #     polygons = [r['shape_attributes'] for r in a['regions']]

            self.add_image(
                "stanford",
                image_id=a['image_id'],  # use file name as a unique image id
                path=filepath,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


class StanfordConfig(Config):
    # Give the configuration a recognizable name
    NAME = "stanford"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    DETECTION_MIN_CONFIDENCE = 0.5

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    #IMAGE_MIN_DIM = 1024
    #IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    #TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    #STEPS_PER_EPOCH = 300

    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 5

    USE_MINI_MASK = False

    CLASS_DICT = {0: u'__background__',
                  1: u'car',
                  }


def main(args):
    command = args.command
    weights = args.weights  # mask_rcnn_coco.h5'
    dataset = args.dataset
    logs = args.logs
    video = args.video
    output = args.output
    config_arg = args.config

    print(config_arg)

    # Setup configs
    if config_arg == 'coco':
        config = CocoConfig()
    elif config_arg == 'shapes':
        config = ShapesConfig()
    elif config_arg == 'wescam':
        config = WESCAMConfig()
    elif config_arg == 'stanford':
        config = StanfordConfig()
    config.display()

    if args.roi_layer:
        config.POST_NMS_ROIS_INFERENCE = 1000

    # Create model
    if command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.weights.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.weights

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    #model.load_weights(model_path, by_name=True)

    model.keras_model.summary()

    if args.roi_layer:
        layer_name = 'ROI'
        layer_model = Model(inputs=model.keras_model.input,
                            outputs=model.keras_model.get_layer(layer_name).output)
        model.keras_model = layer_model

    elif args.detection_layer:
        layer_name = 'mrcnn_detection'
        layer_model = Model(inputs=model.keras_model.input,
                            outputs=model.keras_model.get_layer(layer_name).output)
        model.keras_model = layer_model

    # Train or evaluate
    if command == "train":
        if config == 'coco':
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
        file_name = output
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
                if args.roi_layer:
                    # Predicts different shape than what model.detect expects
                    # Predicts shape [1, 1000, 4]
                    # Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]

                    # Mold inputs to format expected by the neural network
                    molded_images, image_metas, windows = model.mold_inputs([image])

                    # Validate image sizes
                    # All images in a batch MUST be of the same size
                    image_shape = molded_images[0].shape
                    for g in molded_images[1:]:
                        assert g.shape == image_shape, \
                            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

                    # Anchors
                    anchors = model.get_anchors(image_shape)
                    # Duplicate across the batch dimension because Keras requires it
                    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)


                    r = model.keras_model.predict([molded_images, image_metas, anchors])

                    window = windows[0]
                    original_image_shape = image.shape
                    boxes = r[0, :, :]
                    # original_image_shape = [height, width, 3]

                    # Translate normalized coordinates in the resized image to pixel
                    # coordinates in the original image before resizing
                    window = utils.norm_boxes(window, image_shape[:2])
                    wy1, wx1, wy2, wx2 = window
                    shift = np.array([wy1, wx1, wy1, wx1])
                    wh = wy2 - wy1  # window height
                    ww = wx2 - wx1  # window width
                    scale = np.array([wh, ww, wh, ww])
                    # Convert boxes to normalized coordinates on the window
                    boxes = np.divide(boxes - shift, scale)
                    # Convert boxes to pixel coordinates on the original image
                    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

                    splash = np.asarray(image).astype(np.uint8)
                    # Draw Bboxes
                    for index, box in enumerate(boxes):
                        splash = cv2.rectangle(splash, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

                elif args.detection_layer:
                    # Output of shape [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
                    # coordinates are normalized.

                    # Mold inputs to format expected by the neural network
                    molded_images, image_metas, windows = model.mold_inputs([image])

                    # Validate image sizes
                    # All images in a batch MUST be of the same size
                    image_shape = molded_images[0].shape
                    for g in molded_images[1:]:
                        assert g.shape == image_shape, \
                            "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

                    # Anchors
                    anchors = model.get_anchors(image_shape)
                    # Duplicate across the batch dimension because Keras requires it
                    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

                    r = model.keras_model.predict([molded_images, image_metas, anchors])

                    window = windows[0]
                    original_image_shape = image.shape
                    boxes = r[0, :, :4]
                    # original_image_shape = [height, width, 3]

                    # Translate normalized coordinates in the resized image to pixel
                    # coordinates in the original image before resizing
                    window = utils.norm_boxes(window, image_shape[:2])
                    wy1, wx1, wy2, wx2 = window
                    shift = np.array([wy1, wx1, wy1, wx1])
                    wh = wy2 - wy1  # window height
                    ww = wx2 - wx1  # window width
                    scale = np.array([wh, ww, wh, ww])
                    # Convert boxes to normalized coordinates on the window
                    boxes = np.divide(boxes - shift, scale)
                    # Convert boxes to pixel coordinates on the original image
                    boxes = utils.denorm_boxes(boxes, original_image_shape[:2])


                    # Un-normalize
                    # r[0, :, 0] = r[0, :, 0] * height
                    # r[0, :, 2] = r[0, :, 2] * height
                    # r[0, :, 1] = r[0, :, 1] * width
                    # r[0, :, 3] = r[0, :, 3] * width

                    splash = np.asarray(image).astype(np.uint8)
                    # Draw Bboxes
                    #for index, box in enumerate(r[0]):
                    for index, box in enumerate(boxes):
                        splash = cv2.rectangle(splash, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                        splash = cv2.putText(splash,
                                             '{:.2f}'.format(r[0, index, 5]),
                                             (box[3], box[2]), cv2.FONT_HERSHEY_COMPLEX,
                                             1, (255, 0, 0), 2)

                else:
                    r = model.detect([image], verbose=0)[0]
                    # Color splash
                    splash = color_splash(image, r['masks'])

                    # Draw Bboxes
                    for index, box in enumerate(r['rois']):
                        # Shape (y1, x1, y2, x2, class_id)
                        splash = cv2.rectangle(splash, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                        class_id = r['class_ids'][index]
                        splash = cv2.putText(splash, '{} - {:.2f}'.format(config.CLASS_DICT[class_id], r['scores'][index]),
                                             (box[3], box[2]), cv2.FONT_HERSHEY_COMPLEX,
                                             1, (255, 0, 0), 2)

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

    if command == 'roc':
        confidence_thresholds = np.linspace(0.1, 1, 15)
        confidence_thresholds = [0.5]
        all_tp_rates = []
        all_fp_rates = []

        # Validation dataset
        dataset_val = StanfordDataset()
        dataset_val.load_stanford()
        dataset_val.prepare()

        # Compute ROCs for above range of thresholds
        # Compute one for each class vs. the other classes
        for index, conf in enumerate(confidence_thresholds):
            tp_of_img = []
            fp_of_img = []
            all_classes = []

            tp_rates = {}
            fp_rates = {}

            print('Creating model with confidence threshold: {}'.format(conf))
            config.DETECTION_MIN_CONFIDENCE = conf

            # Recreate the model in inference mode
            model = modellib.MaskRCNN(mode="inference",
                                      config=config,
                                      model_dir=MODEL_DIR)

            # Load trained weights
            #model.load_weights(model_path, by_name=True)
            model.load_weights(model_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])

            image_ids = np.random.choice(dataset_val.image_ids, 10)
            image_ids = dataset_val.image_ids
            for image_id in image_ids:
                # Load image and ground truth data
                image, image_meta, gt_class_id, gt_bbox, gt_mask = \
                    modellib.load_image_gt(dataset_val, config,
                                           image_id)
                molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

                # print('OG Image')
                # visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset_val.class_names, figsize=(8, 8))

                # Run object detection
                results = model.detect([image], verbose=0)
                r = results[0]
                # Detect returns:
                # "rois" []
                # "class_ids" [N]
                # "scores" [N]

                # print('Pred Image')
                # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_val.class_names,
                #                            r['scores'], figsize=(8, 8))

                classes = list(set(r['class_ids']))  # All unique class ids
                for c in classes:
                    if c not in all_classes:
                        all_classes.append(c)

                complete_classes = dataset_val.class_ids[1:]
                print(f'complete_classes: {complete_classes}')

                complete_classes = ['car']

                # Need TPR and FPR rates for each class versus the other classes
                # Recall == TPR
                tpr = utils.compute_ap_indiv_class(gt_bbox, gt_class_id, gt_mask,
                                                   r["rois"], r["class_ids"], r["scores"], r['masks'], complete_classes)
                total_fpr = utils.compute_fpr_indiv_class(gt_bbox, gt_class_id, gt_mask,
                                                          r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                          complete_classes)

                # print(f'For Image: TPR: {tpr} -- FPR: {total_fpr}')

                tp_of_img.append(tpr)
                fp_of_img.append(total_fpr)

            #all_classes = dataset_val.class_ids[1:]

            # Need to get average TPR and FPR for number of images used
            #for c in all_classes:
            for c in complete_classes:
                tp_s = 0
                for item in tp_of_img:
                    if c in item.keys():
                        tp_s += item[c]
                    else:
                        tp_s += 0

                tp_rates[c] = tp_s / len(image_ids)
                # tp_rates[c] = tp_s

            # print(tp_rates)

            # for c in all_classes:
            for c in complete_classes:
                fp_s = 0
                for item in fp_of_img:
                    if c in item.keys():
                        fp_s += item[c]
                    else:
                        fp_s += 0
                fp_rates[c] = fp_s / len(image_ids)
                # fp_rates[c] = fp_s

            all_fp_rates.append(fp_rates)
            all_tp_rates.append(tp_rates)

        print(f'TP Rates: {all_tp_rates}')
        print(f'FP Rates: {all_fp_rates}')

        # Plot roc curves
        utils.compute_roc_curve(all_tp_rates, all_fp_rates, save_fig=True)

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
    parser.add_argument('--command', choices=['detect_vid', 'train', 'evaluate', 'roc'])
    parser.add_argument('--dataset', default='')
    parser.add_argument('--logs', default='')
    parser.add_argument('--video', default='')
    parser.add_argument('--output', default='output')
    parser.add_argument('--config', default='coco')
    parser.add_argument('--weights', default='mask_rcnn_coco.h5')
    parser.add_argument('--roi_layer', action='store_true', default=False)
    parser.add_argument('--detection_layer', action='store_true', default=False)
    arguments = parser.parse_args()
    main(arguments)

