import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 80 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.0


def detect_and_color_splash(model, image_path=None, video_path=None, outpath=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        file_name = outpath
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
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


command = 'splash'
weights = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Mask_RCNN/mask_rcnn_coco.h5'  # coco
dataset = ''
logs = ''
image = None
video = '/Users/justinbutler/Desktop/school/Calgary/ML_Work/Datasets/WESCAMdata20210218171306/MFOV - EOW.avi'
output = '/Users/justinbutler/Desktop/test/MFOV_EOW_mask_0conf.avi'
viz_feat_map = False
print_model = False

# Validate arguments
if command == "train":
    assert dataset, "Argument --dataset is required for training"
elif command == "splash":
    pass
print("Weights: ", weights)
print("Dataset: ", dataset)
print("Logs: ", logs)
# Configurations
if command == "train":
    config = BalloonConfig()
else:
    class InferenceConfig(BalloonConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
config.display()
# Create model
if command == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=logs)
else:
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=logs)

# Select weights file to load
if weights.lower() == "coco":
    weights_path = COCO_WEIGHTS_PATH
    # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)
elif weights.lower() == "last":
    # Find last trained weights
    weights_path = model.find_last()
elif weights.lower() == "imagenet":
    # Start from ImageNet trained weights
    weights_path = model.get_imagenet_weights()
else:
    weights_path = weights
# Load weights
print("Loading weights ", weights_path)
if weights.lower() == "coco":
    # Exclude the last layers because they require a matching
    # number of classes
    #model.load_weights(weights_path, by_name=True, exclude=[
    #    "mrcnn_class_logits", "mrcnn_bbox_fc",
    #    "mrcnn_bbox", "mrcnn_mask"])
    model.load_weights(weights_path, by_name=True)
else:
    model.load_weights(weights_path, by_name=True)

if print_model:
    model.keras_model.summary()
    for i in range(len(model.keras_model.layers)):
        layer = model.keras_model.layers[i]
        if 'conv' not in layer.name:
            continue
        print(i, layer.name, layer.output_shape)

if viz_feat_map:
    feat_model = tf.keras.Model(inputs=model.keras_model.input, outputs=model.keras_model.layers)


# Train or evaluate
if command == "splash":
    detect_and_color_splash(model, image_path=image,
                            video_path=video, outpath=output)
else:
    print("'{}' is not recognized. "
          "Use 'train' or 'splash'".format(command))