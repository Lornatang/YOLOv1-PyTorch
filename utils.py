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
from collections import Counter

import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib import pyplot as plt

__all__ = [
    "convert_xywh_to_x1y1x2y2",
    "calculate_iou", "nms", "calculate_map", "convert_cell_boxes_to_boxes", "plot_image"
]


def convert_xywh_to_x1y1x2y2(xywh: list) -> np.ndarray:
    x1y1x2y2 = np.zeros_like(xywh)

    x1y1x2y2[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    x1y1x2y2[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    x1y1x2y2[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    x1y1x2y2[..., 3] = xywh[..., 1] + xywh[..., 3] / 2

    return x1y1x2y2


def calculate_iou(inputs_bboxes: torch.Tensor, target_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.float:
    """Calculate intersection over union

    Args:
        inputs_bboxes (torch.Tensor): Inputs of bounding boxes (Batch_size, 4)
        target_bboxes (torch.Tensor): Target of bounding boxes (Batch_size, 4)
        eps (optional, float): Prevent numeric overflow. Default: 1e-9

    Returns:
        iou (torch.Tensor): Intersection over union

    """
    # Get boxes shape
    inputs_bboxes_x1 = inputs_bboxes[..., 0:1] - inputs_bboxes[..., 2:3] / 2
    inputs_bboxes_y1 = inputs_bboxes[..., 1:2] - inputs_bboxes[..., 3:4] / 2
    inputs_bboxes_x2 = inputs_bboxes[..., 0:1] + inputs_bboxes[..., 2:3] / 2
    inputs_bboxes_y2 = inputs_bboxes[..., 1:2] + inputs_bboxes[..., 3:4] / 2

    target_bboxes_x1 = target_bboxes[..., 0:1] - target_bboxes[..., 2:3] / 2
    target_bboxes_y1 = target_bboxes[..., 1:2] - target_bboxes[..., 3:4] / 2
    target_bboxes_x2 = target_bboxes[..., 0:1] + target_bboxes[..., 2:3] / 2
    target_bboxes_y2 = target_bboxes[..., 1:2] + target_bboxes[..., 3:4] / 2

    # Get intersection area
    x1 = torch.max(inputs_bboxes_x1, target_bboxes_x1)
    y1 = torch.max(inputs_bboxes_y1, target_bboxes_y1)
    x2 = torch.min(inputs_bboxes_x2, target_bboxes_x2)
    y2 = torch.min(inputs_bboxes_y2, target_bboxes_y2)

    x_diff = torch.sub(x2, x1)
    y_diff = torch.sub(y2, y1)
    x_diff = torch.clamp_(x_diff, 0)
    y_diff = torch.clamp_(y_diff, 0)
    intersect_area = torch.mul(x_diff, y_diff)

    # Get bboxes area
    inputs_area = torch.abs((inputs_bboxes_x2 - inputs_bboxes_x1) * (inputs_bboxes_y2 - inputs_bboxes_y1))
    target_area = torch.abs((target_bboxes_x2 - target_bboxes_x1) * (target_bboxes_y2 - target_bboxes_y1))

    # Get union area
    union_area = (inputs_area + target_area - intersect_area + eps)

    iou = intersect_area / union_area

    return iou


def nms(bboxes: list, iou_threshold: float, confidence_threshold: float) -> list:
    """Non Max Suppression given bboxes

    Args:
        bboxes (list): List of lists containing all bboxes.
            bounding_boxes shape: [pred_classes, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): Threshold where predicted bboxes is correct
        confidence_threshold (float): Threshold to remove predicted bboxes

    Returns:
        nms_bounding_boxes (list): Bboxes after performing NMS given a specific IoU threshold

    """
    assert type(bboxes) == list, "bounding_boxes must shape is `list`"

    bboxes = [box for box in bboxes if box[1] > confidence_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    nms_bbox = []

    while bboxes:
        current_boxes = bboxes.pop(0)

        bboxes = [boxes for boxes in bboxes if
                  boxes[0] != current_boxes[0] or calculate_iou(torch.Tensor(current_boxes[2:]),
                                                                torch.Tensor(boxes[2:])) < iou_threshold]

        nms_bbox.append(current_boxes)

    return nms_bbox


def calculate_map(predictions_bboxes: list,
                  annotations_bboxes: list,
                  iou_threshold: float,
                  num_classes: int,
                  eps: float = 1e-9) -> float:
    """Calculate mean average precision

    Args:
        predictions_bboxes (list): Prediction. Shape: [pred_classes, confidence_score, x1, y1, x2, y2]
        annotations_bboxes (list): Annotations. Shape: [pred_classes, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): Threshold where predicted bboxes is correct
        num_classes (int): number of classes
        eps (optional, float): Prevent numeric overflow. Default: 1e-9

    Returns:
        map_value (float): mAP value

    """
    # list storing all AP for respective classes
    average_precisions = []

    for current_classes in range(num_classes):
        predictions = []
        annotations = []

        # Loop through all predictions and targets, add when the category name is correct
        for prediction_bboxes in predictions_bboxes:
            if prediction_bboxes[1] == current_classes:
                predictions.append(prediction_bboxes)

        for annotation_bboxes in annotations_bboxes:
            if annotation_bboxes[1] == current_classes:
                annotations.append(annotation_bboxes)

        # Find the number of bboxes per training example
        amount_bboxes = Counter([annotation[0] for annotation in annotations])
        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        # Sort by box probabilities which is index 2
        predictions.sort(key=lambda x: x[2], reverse=True)
        true_positive = torch.zeros((len(predictions)))
        false_positive = torch.zeros((len(predictions)))
        total_annotations_bboxes = len(annotations)

        # If none exists for this class then we can safely skip
        if total_annotations_bboxes == 0:
            continue

        for prediction_index, prediction in enumerate(predictions):
            # Only take out the ground_truths that have the same training idx as detection
            annotations_image = [bbox for bbox in annotations if bbox[0] == prediction[0]]

            # Record the best IOU value
            best_iou = 0
            best_annotations_index = 0

            for annotation_index, annotation in enumerate(annotations_image):
                iou = calculate_iou(torch.tensor(prediction[3:]), torch.tensor(annotation[3:]))

                if iou > best_iou:
                    best_iou = iou
                    best_annotations_index = annotation_index

            if best_iou > iou_threshold:
                # Only detect ground truth detection once
                if amount_bboxes[prediction[0]][best_annotations_index] == 0:
                    # true positive and add this bounding box to seen
                    true_positive[prediction_index] = 1
                    amount_bboxes[prediction[0]][best_annotations_index] = 1
                else:
                    false_positive[prediction_index] = 1

            # If IOU is smaller the detection is a false positive
            else:
                false_positive[prediction_index] = 1

        # Calculate the cumulative value of each row of an array
        true_positive_cumsum = torch.cumsum(true_positive, dim=0)
        false_positive_cumsum = torch.cumsum(false_positive, dim=0)

        recalls = true_positive_cumsum / (total_annotations_bboxes + eps)
        precisions = true_positive_cumsum / (true_positive_cumsum + false_positive_cumsum + eps)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    map_value = sum(average_precisions) / len(average_precisions)

    return map_value


def convert_cell_boxes_to_boxes(bboxes: torch.Tensor, num_grid: int, num_classes: int) -> list:
    """Convert YOLO-specific meshes to computational meshes

    Args:
        bboxes (torch.Tensor): YOLO predictions out
        num_grid (int): Number of grids in YOLO
        num_classes (int): Number of class

    Returns:
        new_bounding_boxes (list): Computational bounding boxes

    """
    # Converts bounding boxes output from Yolo
    bboxes = bboxes.cpu()
    batch_size = bboxes.shape[0]
    bboxes = bboxes.reshape(batch_size, 7, 7, num_classes + 10)
    bboxes1 = bboxes[..., num_classes + 1:num_classes + 5]
    bboxes2 = bboxes[..., num_classes + 6:num_classes + 10]
    scores = torch.cat([bboxes[..., num_classes].unsqueeze(0), bboxes[..., num_classes+5].unsqueeze(0)], 0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / num_grid * (best_boxes[..., :1] + cell_indices)
    y = 1 / num_grid * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / num_grid * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = bboxes[..., :num_classes].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(bboxes[..., num_classes], bboxes[..., num_classes+5]).unsqueeze(-1)
    converted_predictions = torch.cat([predicted_class, best_confidence, converted_bboxes], -1)

    # Convert YOLO-specific grid to computational grid
    converted_predictions = converted_predictions.reshape(bboxes.shape[0], num_grid * num_grid, -1)
    converted_predictions[..., 0] = converted_predictions[..., 0].long()
    new_bounding_boxes = []

    for boxes_index in range(bboxes.shape[0]):
        bboxes = []

        for bbox_idx in range(num_grid * num_grid):
            bboxes.append([x.item() for x in converted_predictions[boxes_index, bbox_idx, :]])
        new_bounding_boxes.append(bboxes)

    return new_bounding_boxes


def plot_image(image: np.ndarray, boxes: list, index):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "The boxes must is x, y, w, h!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle((upper_left_x * width, upper_left_y * height),
                                 box[2] * width,
                                 box[3] * height,
                                 linewidth=1,
                                 edgecolor="r",
                                 facecolor="none")
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig(f"{index}.png")
