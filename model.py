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
from torchvision.models.feature_extraction import create_feature_extractor

from utils import calculate_iou

__all__ = [
    "YOLOv1TinyFeature", "YOLOv1Feature",
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


class YOLOv1TinyFeature(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(YOLOv1TinyFeature, self).__init__()
        self.features = nn.Sequential(
            # 224*224*3 -> 112*112*16
            _BasicConvBlock(3, 16, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 112*112*16 -> 56*56*32
            _BasicConvBlock(16, 32, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 56*56*32 -> 28*28*64
            _BasicConvBlock(32, 64, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 28*28*64 -> 14*14*128
            _BasicConvBlock(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 14*14*128 -> 7*7*256
            _BasicConvBlock(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 7*7*256 -> 3*3*1024
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 1024, 4096),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class YOLOv1Feature(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(YOLOv1Feature, self).__init__()
        self.features = nn.Sequential(
            # 224*224*3 -> 56*56*64
            _BasicConvBlock(3, 64, (7, 7), (2, 2), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 56*56*64 -> 28*28*192
            _BasicConvBlock(64, 192, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 28*28*192 -> 14*14*512
            _BasicConvBlock(192, 128, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(128, 256, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(256, 256, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.MaxPool2d((2, 2), (2, 2)),

            # 14*14*512 -> 7*7*1024
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

            # 7*7*1024 -> 3*3*1024
            _BasicConvBlock(1024, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 512, (1, 1), (1, 1), (0, 0)),
            _BasicConvBlock(512, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (2, 2), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (1, 1), (1, 1)),
            _BasicConvBlock(1024, 1024, (3, 3), (1, 1), (1, 1)),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Linear(3 * 3 * 1024, 4096),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class YOLOv1Tiny(nn.Module):
    def __init__(self,
                 num_grid: int,
                 num_bboxes: int,
                 num_classes: int,
                 feature_node_name: str = "features.12.bcb.2") -> None:
        super(YOLOv1Tiny, self).__init__()
        self.feature_node_name = feature_node_name

        self.features = create_feature_extractor(YOLOv1TinyFeature(), [feature_node_name])
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_grid * num_grid * (num_bboxes * 5 + num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)[self.feature_node_name]
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class YOLOv1(nn.Module):
    def __init__(self,
                 num_grid: int,
                 num_bboxes: int,
                 num_classes: int,
                 feature_node_name: str = "features.27.bcb.2") -> None:
        super(YOLOv1, self).__init__()
        self.feature_node_name = feature_node_name

        self.features = create_feature_extractor(YOLOv1Feature(), [feature_node_name])
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_grid * num_grid * (num_bboxes * 5 + num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)[self.feature_node_name]
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class YOLOLoss(nn.Module):
    def __init__(self,
                 num_grid: int,
                 num_bboxes: int,
                 num_classes: int,
                 eps: float = 1e-9) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()
        self.num_grid = num_grid
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.eps = eps

        # These Loss values are set in the YOLO paper
        self.boxes_coefficient_loss = 5
        self.non_object_coefficient_loss = 0.5

    def forward(self,
                inputs: torch.Tensor,
                target: torch.Tensor) -> [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # Convert output shape to [batch_size, num_grid, num_grid, num_classes+num_bounding_boxes*5]
        inputs = inputs.view([-1, self.num_grid, self.num_grid, self.num_classes + self.num_bboxes * 5])

        # Calculate IoU for the two predicted bounding boxes with target bbox
        inputs_iou = calculate_iou(inputs[..., self.num_classes + 1:self.num_classes + 5],
                                   target[..., self.num_classes + 1:self.num_classes + 5])
        target_iou = calculate_iou(inputs[..., self.num_classes + 6:self.num_classes + 10],
                                   target[..., self.num_classes + 1:self.num_classes + 5])

        iou = torch.cat([inputs_iou.unsqueeze(0), target_iou.unsqueeze(0)], dim=0)

        # Take the box with highest IoU out of the two prediction
        _, best_boxes = torch.max(iou, 0)
        # When object boxes is exists
        exists_boxes = target[..., self.num_classes].unsqueeze(3)

        # Calculate bounding boxes loss
        # Set boxes with no object in them to 0
        inputs_box1 = best_boxes * inputs[..., self.num_classes + 6:self.num_classes + 10]
        inputs_box2 = (1 - best_boxes) * inputs[..., self.num_classes + 1:self.num_classes + 5]
        inputs_box = exists_boxes * (inputs_box1 + inputs_box2)
        target_box = exists_boxes * target[..., self.num_classes + 1:self.num_classes + 5]

        # Make sure value >= 0.
        inputs_box[..., 2:4] = torch.sign(inputs_box[..., 2:4]) * torch.sqrt(torch.abs(inputs_box[..., 2:4] + self.eps))
        target_box[..., 2:4] = torch.sqrt(target_box[..., 2:4])

        bboxes_loss = self.criterion(torch.flatten(inputs_box, end_dim=-2), torch.flatten(target_box, end_dim=-2))
        bboxes_loss = torch.mul(bboxes_loss, self.boxes_coefficient_loss)

        # Calculate object loss
        # The inputs_boxes is the confidence score for the bbox with highest IoU
        inputs_boxes1 = (best_boxes * inputs[..., self.num_classes + 5:self.num_classes + 6])
        inputs_boxes2 = (1 - best_boxes) * inputs[..., self.num_classes:self.num_classes + 1]
        inputs_boxes = inputs_boxes1 + inputs_boxes2

        object_loss = self.criterion(torch.flatten(exists_boxes * inputs_boxes),
                                     torch.flatten(exists_boxes * target[..., self.num_classes:self.num_classes + 1]))

        # Calculate non-object loss
        inputs_non_object_loss = self.criterion(
            torch.flatten((1 - exists_boxes) * inputs[..., self.num_classes:self.num_classes + 1], start_dim=1),
            torch.flatten((1 - exists_boxes) * target[..., self.num_classes:self.num_classes + 1], start_dim=1))

        target_non_object_loss = self.criterion(
            torch.flatten((1 - exists_boxes) * inputs[..., self.num_classes + 5:self.num_classes + 6], start_dim=1),
            torch.flatten((1 - exists_boxes) * target[..., self.num_classes:self.num_classes + 1], start_dim=1))
        non_object_loss = torch.add(inputs_non_object_loss, target_non_object_loss)
        non_object_loss = torch.mul(non_object_loss, self.non_object_coefficient_loss)

        # Calculate classes loss
        class_loss = self.criterion(torch.flatten(exists_boxes * inputs[..., :self.num_classes], end_dim=-2),
                                    torch.flatten(exists_boxes * target[..., :self.num_classes], end_dim=-2))

        # Four loss count is YOLO loss!
        loss = bboxes_loss + object_loss + non_object_loss + class_loss

        return bboxes_loss, object_loss, non_object_loss, class_loss, loss
