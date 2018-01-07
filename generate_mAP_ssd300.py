from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
import csv

from matplotlib import pyplot as plt
from keras_ssd300 import ssd_300
from keras_ssd_loss import SSDLoss
from keras_layer_AnchorBoxes import AnchorBoxes
from keras_layer_L2Normalization import L2Normalization
from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2
from ssd_batch_generator import BatchGenerator

img_height = 300  # Height of the input images
img_width = 300  # Width of the input images
img_channels = 3  # Number of color channels of the input images
n_classes = 2  # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88,
          1.05]  # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0 / 3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]]  # The anchor box aspect ratios used in the original SSD300
two_boxes_for_ar1 = True
limit_boxes = False  # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2,
             0.2]  # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids'  # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model

K.clear_session()  # Clear previous models from memory.

model, predictor_sizes = ssd_300(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=None,
                                 # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=None,
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)

# 2: Load the trained VGG-16 weights into the model.

# TODO: Set the path to the VGG-16 weights.
# vgg16_path = 'vgg-16_ssd-fcn_ILSVRC-CLS-LOC.h5'
#
# model.load_weights(vgg16_path, by_name=True)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-04)

ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = 'ssd300.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session()  # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'compute_loss': ssd_loss.compute_loss})

# 1: Instantiate to `BatchGenerator` objects: One for training, one for validation.

val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.

# TODO: Set the paths to the datasets here.

# The directories that contain the images.

road_test_images_path = 'datasets/LaneMarkings/1/'

# The directories that contain the annotations.
road_test_annotations_path = 'datasets/LaneMarkings/11/'

# The paths to the image sets.
road_test_image_set_path = 'datasets/LaneMarkings/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'lane marking']
#
# train_dataset.parse_xml(images_paths=[VOC_2007_images_path,
#                                       VOC_2007_test_images_path,
#                                       VOC_2012_images_path],
#                         annotations_paths=[VOC_2007_annotations_path,
#                                            VOC_2007_test_annotations_path,
#                                            VOC_2012_annotations_path],
#                         image_set_paths=[VOC_2007_trainval_image_set_path,
#                                          VOC_2007_test_image_set_path,
#                                          VOC_2012_train_image_set_path],
#                         classes=classes,
#                         include_classes='all',
#                         exclude_truncated=False,
#                         exclude_difficult=False,
#                         ret=False)
#
val_dataset.parse_xml(images_paths=[road_test_images_path],
                      annotations_paths=[road_test_annotations_path],
                      image_set_paths=[road_test_image_set_path],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

# 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

# 4: Set the batch size.

batch_size = 32  # Change the batch size if you like, or if you run into memory issues with your GPU.

### Make predictions

# 1: Set the generator

predict_generator = val_dataset.generate(batch_size=batch_size,
                                         shuffle=False,
                                         train=False,
                                         equalize=False,
                                         brightness=False,
                                         flip=False,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(300, 300, 1, 3),
                                         full_crop_and_resize=(300, 300, 1, 3, 0.5),
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True,
                                         include_thresh=0.4,
                                         diagnostics=False,
                                         one_time=True)
# 2: Generate samples


true_csv_file = open('true.csv', 'w', newline="")
pred_csv_file = open('pred.csv', 'w', newline="")

true_writer = csv.writer(true_csv_file)
pred_writer = csv.writer(pred_csv_file)

true_writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'score'])
pred_writer.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'score'])

for i, (X, y_true, filenames) in enumerate(predict_generator):

    # 3: Make a prediction

    # model.predict_generator()
    y_pred = model.predict(X)
    # 4: Decode the raw prediction `y_pred`

    y_pred_decoded = decode_y2(y_pred,
                               confidence_thresh=0.7,
                               iou_threshold=0.2,
                               top_k='all',
                               input_coords='centroids',
                               normalize_coords=normalize_coords,
                               img_height=img_height,
                               img_width=img_width)

    for j, (filename, y_true_item, y_pred_item) in enumerate(zip(filenames, y_true, y_pred_decoded)):
        id = i * batch_size + j
        for box in y_true_item:
            y_true_item_data = [id, box[1], box[3], box[2] - box[1], box[4] - box[3], 1]
            true_writer.writerow(y_true_item_data)

        for box in y_pred_item:
            y_pred_item_data = [id, box[2], box[4], box[3] - box[2], box[5] - box[4], box[1]]
            pred_writer.writerow(y_pred_item_data)

true_csv_file.close()
pred_csv_file.close()
print('Done.')
