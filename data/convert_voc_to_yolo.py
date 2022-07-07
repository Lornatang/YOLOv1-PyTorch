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
import argparse
import os
import shutil
from xml.etree import ElementTree


def main(args) -> None:
    # Create dataset output directory
    if not os.path.exists(args.output_mode_dir):
        os.makedirs(args.output_mode_dir)
    if not os.path.exists(os.path.join(args.output_images_dir, args.mode)):
        os.makedirs(os.path.join(args.output_images_dir, args.mode))
    if not os.path.exists(os.path.join(args.output_labels_dir, args.mode)):
        os.makedirs(os.path.join(args.output_labels_dir, args.mode))

    # Read the image object category in the VOC dataset
    classes = read_classes_from_txt(args.classes_txt_file_path)

    # Read the name of the image in the directory and annotation file name
    images_index = open(args.images_index_path, "r").read().strip("\n ").split()
    write_file = open(os.path.join(args.output_mode_dir, args.mode + ".txt"), "a")

    for file_name in os.listdir(args.input_images_dir):
        base_file_name = os.path.basename(file_name).split(".")[0]
        if base_file_name in images_index:
            # Write the label to the specified path
            convert_labels_to_txt(os.path.join(args.input_labels_dir, base_file_name + ".xml"),
                                  os.path.join(args.output_labels_dir, args.mode, base_file_name + ".txt"),
                                  classes)

            # Copy the image file to the specified path
            shutil.copyfile(os.path.join(args.input_images_dir, file_name),
                            os.path.join(args.output_images_dir, args.mode, file_name))

            # Write the image file names in the same mode to the same txt file
            write_file.write(f"{args.output_images_dir}/{args.mode}/{file_name}\n")

    write_file.close()


def read_classes_from_txt(txt_file_path: str) -> list:
    """Read the tag information from the file to form a category list

    Args:
        txt_file_path (str): Txt file path containing dataset class name

    Returns:
        classes (list): Object detection datasets classes

    """
    classes = []

    # The category labels are read sequentially, and the index cannot be changed at will!
    with open(txt_file_path, "r") as f:
        for line in f.readlines():
            classes.append(line.strip("\n"))

    return classes


def convert_labels_to_txt(input_labels_file_path: str,
                          output_labels_file_path: str,
                          classes: list) -> None:
    """

    Args:
        input_labels_file_path (str): The object location is marked with the file path, xml format
        output_labels_file_path (str): YOLO training data set annotation file path, txt format
        classes (list): Object detection datasets classes

    """
    # Read the label file and open the txt file with write permissions
    annotation_file = open(input_labels_file_path, "r")
    txt_file = open(output_labels_file_path, "w")

    # Parse XML document into element tree.
    labels_tree = ElementTree.parse(annotation_file)
    # Return root element of this tree
    labels_root = labels_tree.getroot()

    # Get the height and width of the entire image
    image_size = labels_root.find("size")
    image_width = int(image_size.find("width").text)
    image_height = int(image_size.find("height").text)

    # Iterate over each annotation object in the annotation file
    for object_information in labels_root.iter("object"):
        # How easily the object can be found, 0 is easy, 1 is hard
        difficult = object_information.find("difficult").text
        # The class name of the current object
        classes_name = object_information.find("name").text

        # If the current annotation file has additional object annotation information
        # or is difficult to identify, skip it
        if classes_name not in classes or int(difficult) == 1:
            continue

        # Get the class name and position in the bounding boxes
        classes_index = classes.index(classes_name)
        xml_bboxes = object_information.find("bndbox")
        bboxes = (float(xml_bboxes.find("xmin").text),
                  float(xml_bboxes.find("xmax").text),
                  float(xml_bboxes.find("ymin").text),
                  float(xml_bboxes.find("ymax").text))

        # YOLO annotation format [classes_index, pos_x, pos_y, bboxes_width, bboxes_height]
        div_width = 1. / image_width
        div_height = 1. / image_height

        pos_x = (bboxes[0] + bboxes[1]) / 2.0
        pos_y = (bboxes[2] + bboxes[3]) / 2.0
        bboxes_width = bboxes[1] - bboxes[0]
        bboxes_height = bboxes[3] - bboxes[2]

        pos_x = pos_x * div_width
        bboxes_width = bboxes_width * div_width
        pos_y = pos_y * div_height
        bboxes_height = bboxes_height * div_height

        txt_file.write(f"{str(classes_index)} {pos_x} {pos_y} {bboxes_width} {bboxes_height}\n")

    # Close all file op
    annotation_file.close()
    txt_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VOC data format to YOLO data format.")
    parser.add_argument("--images_index_path", type=str, help="Dataset input images index path.")
    parser.add_argument("--input_images_dir", type=str, help="Dataset input images directory path.")
    parser.add_argument("--input_labels_dir", type=str, help="Dataset input labels directory path.")
    parser.add_argument("--output_images_dir", type=str, help="Dataset output images directory.")
    parser.add_argument("--output_labels_dir", type=str, help="Dataset output labels directory.")
    parser.add_argument("--classes_txt_file_path", type=str, help="Txt file path containing dataset class name.")
    parser.add_argument("--output_mode_dir", type=str, help="Output directory path.")
    parser.add_argument("--mode", type=str, help="Dataset mode, which can be `train`, `valid` or `test`.")
    args = parser.parse_args()

    main(args)
