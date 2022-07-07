# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import imgaug as ia
import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
ia.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Model arch name
model_arch = "YOLOv1Tiny"
# Model parameters setting
model_num_grid = 7
model_num_bboxes = 2
model_num_classes = 20
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "YOLOv1Tiny"

if mode == "train":
    # Dataset setting
    train_file_index_path = "./data/VOC0712/train.txt"
    train_images_dir = "./data/VOC0712/images/train"
    train_labels_dir = "./data/VOC0712/labels/train"

    test_file_index_path = "./data/VOC0712/test.txt"
    test_images_dir = "./data/VOC0712/images/test"
    test_labels_dir = "./data/VOC0712/labels/test"

    image_size = 448
    batch_size = 64
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_path = "./results/pretrained_models/YOLOv1TinyFeature-ImageNet_1K(mini)-c9be2973.pth.tar"

    # Incremental training and migration training
    resume = ""

    # Total num epochs (i.e 40,000 iters)
    epochs = 155

    # Optimizer parameter
    optim_lr = 5e-4
    optim_momentum = 0.9
    optim_weight_decay = 5e-4

    # Detection parameters
    iou_threshold = 0.5
    confidence_threshold = 0.4

    # How many iterations to print the training result
    print_frequency = 10

if mode == "test":
    # Dataset setting
    file_index_path = "./data/YOLO/test.txt"
    images_dir = "./data/YOLO/images"
    annotations_dir = "./data/YOLO/annotations"

    # Test parameters setting
    image_size = 448
    batch_size = 128
    num_workers = 4

    # Detection parameters
    iou_threshold = 0.5
    confidence_threshold = 0.0

    # model_path = "./results/pretrained_models/YOLOv1Tiny-VOC0712-xxxxxxxx.pth.tar"
    model_path = "./samples/YOLOv1Tiny/epoch_3.pth.tar"
