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

import torch
from torch.utils.data import DataLoader

import config
from dataset import ImageDataset, CUDAPrefetcher
from model import YOLOv1Tiny, YOLOv1
from utils import calculate_map, nms, convert_cell_boxes_to_boxes


def main() -> None:
    # Load test datasets
    data_datasets = ImageDataset(config.file_index_path,
                                 config.images_dir,
                                 config.annotations_dir,
                                 config.image_size,
                                 config.model_num_grid,
                                 config.model_num_bboxes,
                                 config.model_num_classes,
                                 "test")

    # Generator all dataloader
    data_dataloader = DataLoader(data_datasets,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    data_prefetcher = CUDAPrefetcher(data_dataloader, config.device)

    # Build model
    if config.model_arch == "YOLOv1":
        model = YOLOv1(config.model_num_grid, config.model_num_bboxes, config.model_num_classes)
    elif config.model_arch == "YOLOv1Tiny":
        model = YOLOv1Tiny(config.model_num_grid, config.model_num_bboxes, config.model_num_classes)
    else:
        print("unrecognized model schema name, calling `YOLOv1Tiny` model")
        model = YOLOv1Tiny(config.model_num_grid, config.model_num_bboxes, config.model_num_classes)

    # Transfer to CUDA
    model = model.to(device=config.device, memory_format=torch.channels_last)

    # Load model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Put the YOLO network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    total_index = 0

    # Initialize all bboxes
    predictions_bboxes_list = []
    annotations_bboxes_list = []

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            images = batch_data["image"].to(device=config.device,
                                            memory_format=torch.channels_last,
                                            non_blocking=True)
            annotations = batch_data["annotation"].to(device=config.device,
                                                      memory_format=torch.channels_last,
                                                      non_blocking=True)

            # Use the generator model to generate a fake sample
            predictions = model(images)

            predictions_bboxes = convert_cell_boxes_to_boxes(predictions, config.model_num_grid)
            annotations_bboxes = convert_cell_boxes_to_boxes(annotations, config.model_num_grid)

            for index in range(images.size(0)):
                nms_predictions_bboxes = nms(predictions_bboxes[index],
                                             config.iou_threshold,
                                             config.confidence_threshold)

                for nms_prediction_bboxes in nms_predictions_bboxes:
                    predictions_bboxes_list.append([total_index] + nms_prediction_bboxes)

                for annotation_bboxes in annotations_bboxes[index]:
                    if annotation_bboxes[1] > config.confidence_threshold:
                        annotations_bboxes_list.append([total_index] + annotation_bboxes)

                total_index += 1

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

        # Calculate mAP value for dataset
        map_value = calculate_map(predictions_bboxes_list,
                                  annotations_bboxes_list,
                                  config.iou_threshold,
                                  config.model_num_classes)

        # Print metrics
        print(f"* mAP: {map_value:.8f}")


if __name__ == "__main__":
    main()
