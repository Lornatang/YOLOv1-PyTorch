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
import torch
from torch import nn

from utils import calculate_iou

__all__ = [
    "YOLOv1Tiny", "YOLOv1",
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


class YOLOv1Tiny(nn.Module):
    def __init__(self, num_grid: int, num_bounding_boxes: int, num_classes: int) -> None:
        super(YOLOv1Tiny, self).__init__()
        self.num_grid = num_grid
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # 448*448*3 -> 224*224*16
            _BasicConvBlock(3, 16, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 224*224*16 -> 112*112*32
            _BasicConvBlock(16, 32, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 112*112*32 -> 56*56*64
            _BasicConvBlock(32, 64, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 56*56*64 -> 28*28*128
            _BasicConvBlock(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 28*28*128 -> 14*14*256
            _BasicConvBlock(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 14*14*256 -> 7*7*1024
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
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


class YOLOv1(nn.Module):
    def __init__(self, num_grid: int, num_bounding_boxes: int, num_classes: int) -> None:
        super(YOLOv1, self).__init__()
        self.num_grid = num_grid
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # 448*448*3 -> 112*112*64
            _BasicConvBlock(3, 64, (7, 7), (2, 2), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 112*112*64 -> 56*56*192
            _BasicConvBlock(64, 192, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 56*56*192 -> 28*28*512
            _BasicConvBlock(192, 128, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(128, 256, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(256, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 28*28*512 -> 14*14*1024
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

            # 14*14*1024 -> 7*7*1024
            _BasicConvBlock(1024, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (2, 2), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (1, 1), (1, 1)),
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


class YOLOLoss(nn.Module):
    def __init__(self,
                 criterion: nn.MSELoss,
                 num_grid: int,
                 num_bounding_boxes: int,
                 num_classes: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.criterion = criterion
        self.num_grid = num_grid
        self.num_bounding_boxes = num_bounding_boxes
        self.num_classes = num_classes
        self.epsilon = eps

        # These Loss values are set in the YOLO paper
        self.boxes_coefficient_loss = 5
        self.non_object_coefficient_loss = 0.5

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert output shape to [batch_size, num_grid, num_grid, num_classes+num_bounding_boxes*5]
        inputs = inputs.view([-1, self.num_grid, self.num_grid, self.num_classes + self.num_bounding_boxes * 5])

        # Calculate IoU for the two predicted bounding boxes with target bbox
        inputs_iou = calculate_iou(inputs[..., 21:25], target[..., 21:25])
        target_iou = calculate_iou(inputs[..., 26:30], target[..., 21:25])
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
