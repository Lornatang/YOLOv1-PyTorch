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
import os

# Process `VOC2007` train dataset
print(f"Process `VOC2007-train` dataset")
os.system("python ./convert_voc_to_yolo.py "
          "--images_index_path ./VOCdevkit/VOC2007/ImageSets/Main/train.txt "
          "--input_images_dir ./VOCdevkit/VOC2007/JPEGImages "
          "--input_labels_dir ./VOCdevkit/VOC2007/Annotations "
          "--output_images_dir ./VOC0712/images "
          "--output_labels_dir ./VOC0712/labels "
          "--classes_txt_file_path ./voc_classes.txt "
          "--output_mode_dir ./VOC0712 "
          "--mode train")
# Process `VOC2007` valid dataset
print(f"Process `VOC2007-valid` dataset")
os.system("python ./convert_voc_to_yolo.py "
          "--images_index_path ./VOCdevkit/VOC2007/ImageSets/Main/val.txt "
          "--input_images_dir ./VOCdevkit/VOC2007/JPEGImages "
          "--input_labels_dir ./VOCdevkit/VOC2007/Annotations "
          "--output_images_dir ./VOC0712/images "
          "--output_labels_dir ./VOC0712/labels "
          "--classes_txt_file_path ./voc_classes.txt "
          "--output_mode_dir ./VOC0712 "
          "--mode train")
# Process `VOC2007` test dataset
print(f"Process `VOC2007-test` dataset")
os.system("python ./convert_voc_to_yolo.py "
          "--images_index_path ./VOCdevkit/VOC2007/ImageSets/Main/test.txt "
          "--input_images_dir ./VOCdevkit/VOC2007/JPEGImages "
          "--input_labels_dir ./VOCdevkit/VOC2007/Annotations "
          "--output_images_dir ./VOC0712/images "
          "--output_labels_dir ./VOC0712/labels "
          "--classes_txt_file_path ./voc_classes.txt "
          "--output_mode_dir ./VOC0712 "
          "--mode test")
# Process `VOC2012` train dataset
print(f"Process `VOC2012-train` dataset")
os.system("python ./convert_voc_to_yolo.py "
          "--images_index_path ./VOCdevkit/VOC2012/ImageSets/Main/train.txt "
          "--input_images_dir ./VOCdevkit/VOC2012/JPEGImages "
          "--input_labels_dir ./VOCdevkit/VOC2012/Annotations "
          "--output_images_dir ./VOC0712/images "
          "--output_labels_dir ./VOC0712/labels "
          "--classes_txt_file_path ./voc_classes.txt "
          "--output_mode_dir ./VOC0712 "
          "--mode train")
# Process `VOC2012` valid dataset
print(f"Process `VOC2012-valid` dataset")
os.system("python ./convert_voc_to_yolo.py "
          "--images_index_path ./VOCdevkit/VOC2012/ImageSets/Main/val.txt "
          "--input_images_dir ./VOCdevkit/VOC2012/JPEGImages "
          "--input_labels_dir ./VOCdevkit/VOC2012/Annotations "
          "--output_images_dir ./VOC0712/images "
          "--output_labels_dir ./VOC0712/labels "
          "--classes_txt_file_path ./voc_classes.txt "
          "--output_mode_dir ./VOC0712 "
          "--mode train")
