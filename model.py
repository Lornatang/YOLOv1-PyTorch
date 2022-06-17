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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

__all__ = [
    "YOLOv1",
    "calculate_map", "get_bounding_boxes",
    "YOLOLoss",
]


class _BasicConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple, padding: tuple) -> None:
        super(_BasicConvBlock, self).__init__()
        self.bcb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bcb(x)

        return out


class YOLOv1(nn.Module):
    def __init__(self, num_grid: int, num_bounding_boxes: int, num_classes: int) -> None:
        super(YOLOv1, self).__init__()
        self.num_grid = num_grid
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # 448*448*3 -> 224*224*64
            _BasicConvBlock(3, 64, (7, 7), (2, 2), (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 224*224*64 -> 112*112*192
            _BasicConvBlock(64, 192, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 112*112*256 -> 56*56*512
            _BasicConvBlock(192, 128, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(128, 256, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(256, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 56*56*512 -> 28*28*1024
            _BasicConvBlock(512, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(512, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(512, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(512, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(512, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 28*28*1024 -> 7*7*1024
            _BasicConvBlock(1024, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 1024, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(1024, 1024, (3, 3), (2, 2), (1, 1)),
            _BasicConvBlock(1024, 1024, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(1024, 1024, (3, 3), (1, 1), (1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_grid * num_grid * (num_bounding_boxes * 5 + num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def _calculate_iou(inputs_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.float:
    """Calculate intersection over union

    Args:
        inputs_boxes (torch.Tensor): Inputs of bounding boxes (Batch_size, 4)
        target_boxes (torch.Tensor): Target of bounding boxes (Batch_size, 4)

    Returns:
        iou (torch.Tensor): Intersection over union

    """
    epsilon = 1e-6

    # Get boxes shape
    inputs_boxes_x1 = inputs_boxes[..., 0:1] - torch.div(inputs_boxes[..., 2:3], 2, "trunc")
    inputs_boxes_y1 = inputs_boxes[..., 1:2] - torch.div(inputs_boxes[..., 3:4], 2, "trunc")
    inputs_boxes_x2 = inputs_boxes[..., 0:1] + torch.div(inputs_boxes[..., 2:3], 2, "trunc")
    inputs_boxes_y2 = inputs_boxes[..., 1:2] + torch.div(inputs_boxes[..., 3:4], 2, "trunc")

    target_boxes_x1 = target_boxes[..., 0:1] - torch.div(target_boxes[..., 2:3], 2, "trunc")
    target_boxes_y1 = target_boxes[..., 1:2] - torch.div(target_boxes[..., 3:4], 2, "trunc")
    target_boxes_x2 = target_boxes[..., 0:1] + torch.div(target_boxes[..., 2:3], 2, "trunc")
    target_boxes_y2 = target_boxes[..., 1:2] + torch.div(target_boxes[..., 3:4], 2, "trunc")

    # Get boxes area
    inputs_boxes_area = torch.abs((inputs_boxes_x2 - inputs_boxes_x1) * (inputs_boxes_y2 - inputs_boxes_y1))
    target_boxes_area = torch.abs((target_boxes_x2 - target_boxes_x1) * (target_boxes_y2 - target_boxes_y1))

    # Get x1y1x2y2 diff
    x1 = torch.max(inputs_boxes_x1, target_boxes_x1)
    y1 = torch.min(inputs_boxes_y1, target_boxes_y1)
    x2 = torch.max(inputs_boxes_x2, target_boxes_x2)
    y2 = torch.min(inputs_boxes_y2, target_boxes_y2)

    x_diff = torch.sub(x2, x1)
    y_diff = torch.sub(y2, y1)
    x_diff = torch.clamp_(x_diff, 0)
    y_diff = torch.clamp_(y_diff, 0)
    intersect_area = torch.mul(x_diff, y_diff)

    iou = intersect_area / (inputs_boxes_area + target_boxes_area - intersect_area + epsilon)

    return iou


def _nms(bounding_boxes: list, iou_threshold: float, threshold: float) -> list:
    """Non Max Suppression given bboxes

    Args:
        bounding_boxes (list): List of lists containing all bboxes.
            bounding_boxes shape: [pred_classes, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): Threshold where predicted bboxes is correct
        threshold (float): Threshold to remove predicted bboxes

    Returns:
        nms_bounding_boxes (list): Bboxes after performing NMS given a specific IoU threshold

    """
    assert type(bounding_boxes) == list, "bounding_boxes must shape is `list`"

    bounding_boxes = [box for box in bounding_boxes if box[1] > threshold]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    nms_bounding_boxes = []

    while bounding_boxes:
        current_boxes = bounding_boxes.pop(0)

        bounding_boxes = [boxes for boxes in bounding_boxes if
                          boxes[0] != current_boxes[0] or _calculate_iou(torch.Tensor(current_boxes[2:]),
                                                                         torch.Tensor(boxes[2:])) < iou_threshold]

        nms_bounding_boxes.append(current_boxes)

    return nms_bounding_boxes


def calculate_map(inputs_boxes: list, target_boxes: list, iou_threshold: float, num_classes: int) -> float:
    """Calculate mean average precision

    Args:
        inputs_boxes (list): Prediction. Shape: [pred_classes, confidence_score, x1, y1, x2, y2]
        target_boxes (list): Target. Shape: [pred_classes, confidence_score, x1, y1, x2, y2]
        iou_threshold (float): Threshold where predicted bboxes is correct
        num_classes (int): number of classes

    Returns:
        map_value (float): mAP value

    """
    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for current_classes in range(num_classes):
        predictions = []
        targets = []

        # Loop through all predictions and targets, add when the category name is correct
        for detection in inputs_boxes:
            if detection[1] == current_classes:
                predictions.append(detection)

        for true_box in target_boxes:
            if true_box[1] == current_classes:
                targets.append(true_box)

        # Find the number of bboxes per training example
        amount_bboxes = Counter([gt[0] for gt in targets])
        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        # Sort by box probabilities which is index 2
        predictions.sort(key=lambda x: x[2], reverse=True)
        true_positive = torch.zeros((len(predictions)))
        false_positive = torch.zeros((len(predictions)))
        total_true_bboxes = len(targets)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for prediction_index, prediction in enumerate(predictions):
            # Only take out the ground_truths that have the same training idx as detection
            target_image = [bbox for bbox in targets if bbox[0] == prediction[0]]

            # Record the best IOU value
            best_iou = 0
            best_target_index = 0

            for target_index, target in enumerate(target_image):
                iou = _calculate_iou(torch.tensor(prediction[3:]), torch.tensor(target[3:]))

                if iou > best_iou:
                    best_iou = iou
                    best_target_index = target_index

            if best_iou > iou_threshold:
                # Only detect ground truth detection once
                if amount_bboxes[prediction[0]][best_target_index] == 0:
                    # true positive and add this bounding box to seen
                    true_positive[prediction_index] = 1
                    amount_bboxes[prediction[0]][best_target_index] = 1
                else:
                    false_positive[prediction_index] = 1

            # If IOU is smaller the detection is a false positive
            else:
                false_positive[prediction_index] = 1

        # Calculate the cumulative value of each row of an array
        true_positive_cumsum = torch.cumsum(true_positive, dim=0)
        false_positive_cumsum = torch.cumsum(false_positive, dim=0)

        recalls = true_positive_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(true_positive_cumsum, (true_positive_cumsum + false_positive_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    map_value = sum(average_precisions) / len(average_precisions)

    return map_value


def _convert_cell_boxes_to_boxes(predictions: torch.Tensor, grid_size: int) -> list:
    """Convert YOLO-specific meshes to computational meshes

    Args:
        predictions (torch.Tensor): YOLO predictions out
        grid_size (int): Number of grids in YOLO

    Returns:
        new_bounding_boxes (list): Computational bounding boxes

    """
    # Converts bounding boxes output from Yolo
    predictions = predictions.cpu()
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat([predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)], 0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / grid_size * (best_boxes[..., :1] + cell_indices)
    y = 1 / grid_size * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / grid_size * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_predictions = torch.cat([predicted_class, best_confidence, converted_bboxes], -1)

    # Convert YOLO-specific grid to computational grid
    converted_predictions = converted_predictions.reshape(predictions.shape[0], grid_size * grid_size, -1)
    converted_predictions[..., 0] = converted_predictions[..., 0].long()
    new_bounding_boxes = []

    for boxes_index in range(predictions.shape[0]):
        bboxes = []

        for bbox_idx in range(grid_size * grid_size):
            bboxes.append([x.item() for x in converted_predictions[boxes_index, bbox_idx, :]])
        new_bounding_boxes.append(bboxes)

    return new_bounding_boxes


def get_bounding_boxes(data_loader: DataLoader,
                       model: YOLOv1,
                       grid_size: int,
                       iou_threshold: float,
                       threshold: float,
                       device: torch.device):
    """Get bounding boxes from YOLo output

    Args:
        data_loader (DataLoader): Image loader provided for YOLO training
        model (YOLOv1): YOLOv1 model
        grid_size (int): Number of grids in YOLO
        iou_threshold (float): Threshold where predicted bboxes is correct
        threshold (float): Threshold to remove predicted bboxes
        device (torch.Tensor): Specify the operating device model

    Returns:
        prediction_boxes (list): All prediction boxes
        target_boxes (list): All target boxes

    """
    prediction_boxes = []
    target_boxes = []

    # Set the model to validation mode without taking gradients
    model.eval()
    total_index = 0

    for batch_index, (images, target) in enumerate(data_loader):
        # Get batch size
        batch_size = images.size(0)

        # It will be faster to move data to the CUDA side
        images = images.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        with torch.no_grad():
            predictions = model(images)

        predictions_bounding_boxes = _convert_cell_boxes_to_boxes(predictions, grid_size)
        target_bounding_boxes = _convert_cell_boxes_to_boxes(target, grid_size)

        for index in range(batch_size):
            nms_boxes = _nms(predictions_bounding_boxes[index], iou_threshold, threshold)

            for nms_box in nms_boxes:
                prediction_boxes.append([total_index] + nms_box)

            for box in target_bounding_boxes[index]:
                if box[1] > threshold:
                    target_boxes.append([total_index] + box)

            total_index += 1

    # Restore model training mode
    model.train()

    return prediction_boxes, target_boxes


class YOLOLoss(nn.Module):
    def __init__(self,
                 criterion: nn.MSELoss,
                 num_grid: int,
                 num_bounding_boxes: int,
                 num_classes: int,
                 epsilon: float = 1e-6) -> None:
        super().__init__()
        self.criterion = criterion
        self.num_grid = num_grid
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes
        self.epsilon = epsilon

        # These Loss values are set in the YOLO paper
        self.boxes_coefficient_loss = 5
        self.non_object_coefficient_loss = 0.5

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert output shape to [batch_size, num_grid, num_grid, num_classes+num_bounding_boxes*5]
        inputs = inputs.view([-1, self.num_grid, self.num_grid, self.num_classes + self.num_bounding_boxes * 5])

        # Calculate IoU for the two predicted bounding boxes with target bbox
        inputs_iou = _calculate_iou(inputs[..., 21:25], target[..., 21:25])
        target_iou = _calculate_iou(inputs[..., 26:30], target[..., 21:25])
        iou = torch.cat([inputs_iou.unsqueeze(0), target_iou.unsqueeze(0)], 0)

        # Take the box with highest IoU out of the two prediction
        max_iou, best_boxes = torch.max(iou, 0)
        # When object boxes is exists
        exists_boxes = target[..., 20].unsqueeze(3)

        # Calculate bounding boxes loss
        # Set boxes with no object in them to 0
        box_predictions = exists_boxes * (best_boxes * inputs[..., 26:30] + (1 - best_boxes) * inputs[..., 21:25])
        box_targets = exists_boxes * target[..., 21:25]

        # Make sure value >= 0.
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + self.epsilon))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        boxes_loss = self.criterion(torch.flatten(box_predictions, end_dim=-2),
                                    torch.flatten(box_targets, end_dim=-2))
        boxes_loss = torch.mul(boxes_loss, self.boxes_coefficient_loss)

        # Calculate object loss
        # The inputs_boxes is the confidence score for the bbox with highest IoU
        inputs_boxes = (best_boxes * inputs[..., 25:26]) + (1 - best_boxes) * inputs[..., 20:21]

        object_loss = self.criterion(torch.flatten(exists_boxes * inputs_boxes),
                                     torch.flatten(exists_boxes * target[..., 20:21]))

        # Calculate non-object loss
        inputs_non_object_loss = self.criterion(torch.flatten((1 - exists_boxes) * inputs[..., 20:21], start_dim=1),
                                                torch.flatten((1 - exists_boxes) * target[..., 20:21], start_dim=1))

        target_non_object_loss = self.criterion(torch.flatten((1 - exists_boxes) * inputs[..., 25:26], start_dim=1),
                                                torch.flatten((1 - exists_boxes) * target[..., 20:21], start_dim=1))
        non_object_loss = torch.add(inputs_non_object_loss, target_non_object_loss)
        non_object_loss = torch.mul(non_object_loss, self.non_object_coefficient_loss)

        # Calculate classes loss
        class_loss = self.criterion(torch.flatten(exists_boxes * inputs[..., :20], end_dim=-2),
                                    torch.flatten(exists_boxes * target[..., :20], end_dim=-2))

        # Four loss count is YOLO loss!
        loss = boxes_loss + object_loss + non_object_loss + class_loss

        return loss
