# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
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
import cv2
import numpy as np
import torch
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision.transforms import functional as F

from utils import convert_xywh_to_x1y1x2y2

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "ImageAugment", "RelativeLabels", "AbsoluteLabels", "PadSquare", "RGBToHSV", "HSVToRGB",
    "AdjustBrightness", "AdjustSaturation", "Resize", "ToTensor",
]


def image_to_tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (torch.Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, False, False)

    """
    # Convert image data type to Tensor data type
    tensor = F.to_tensor(image)

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: torch.Tensor, range_norm: bool, half: bool) -> np.ndarray:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (torch.Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_tensor = torch.randn([1, 3, 448, 448])
        >>> example_image = tensor_to_image(example_tensor, False, False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


class ImageAugment(object):
    def __init__(self, augment_functions: iaa.Sequential) -> None:
        self.augment_functions = augment_functions

    def __call__(self, image: np.ndarray, target: np.ndarray) -> [np.ndarray, np.ndarray]:
        # Convert bboxes shape to x1y1x2y2
        target = np.array(target)
        target[:, 1:] = convert_xywh_to_x1y1x2y2(target[:, 1:])

        # Convert bboxes to image augment
        target_bboxes = BoundingBoxesOnImage(
            [BoundingBox(boxes[1], boxes[2], boxes[3], boxes[4], boxes[0]) for boxes in target],
            image.shape)

        # Apply augmentations
        image, target_bboxes = self.augment_functions(image=image, bounding_boxes=target_bboxes)

        # Clip out of image boxes
        target_bboxes = target_bboxes.clip_out_of_image()

        # Convert bounding boxes back to numpy
        target = np.zeros((len(target_bboxes), 5))
        for target_bboxes_index, bboxes in enumerate(target_bboxes):
            # Extract coordinates for un-padded + unscaled image
            x1 = bboxes.x1
            y1 = bboxes.y1
            x2 = bboxes.x2
            y2 = bboxes.y2

            # Returns (x, y, w, h)
            target[target_bboxes_index, 0] = bboxes.label
            target[target_bboxes_index, 1] = ((x1 + x2) / 2)
            target[target_bboxes_index, 2] = ((y1 + y2) / 2)
            target[target_bboxes_index, 3] = (x2 - x1)
            target[target_bboxes_index, 4] = (y2 - y1)

        return image, target


class RelativeLabels(object):
    def __call__(self, image: np.ndarray, target: np.ndarray) -> [np.ndarray, np.ndarray]:
        image_height, image_width, _ = image.shape
        target[:, [1, 3]] /= image_width
        target[:, [2, 4]] /= image_height

        return image, target


class AbsoluteLabels(object):
    def __call__(self, image: np.ndarray, target: np.ndarray) -> [np.ndarray, np.ndarray]:
        image_height, image_width, _ = image.shape
        target[:, [1, 3]] *= image_width
        target[:, [2, 4]] *= image_height

        return image, target


class PadSquare(ImageAugment):
    def __init__(self) -> None:
        super(PadSquare).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.PadToAspectRatio(1.0, position="center-center").to_deterministic()
        ])


class RGBToHSV(ImageAugment):
    def __init__(self) -> None:
        super(RGBToHSV).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.ChangeColorspace("HSV", "RGB")
        ])


class HSVToRGB(ImageAugment):
    def __init__(self) -> None:
        super(HSVToRGB).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.ChangeColorspace("RGB", "HSV")
        ])


class AdjustBrightness(ImageAugment):
    def __init__(self, add: tuple, from_colorspace: str, to_colorspace: str) -> None:
        super(AdjustBrightness).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.AddToBrightness(add, from_colorspace, to_colorspace)
        ])


class AdjustSaturation(ImageAugment):
    def __init__(self, add: tuple, from_colorspace: str) -> None:
        super(AdjustSaturation).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.AddToSaturation(add, from_colorspace)
        ])


class Resize(ImageAugment):
    def __init__(self, size: list, interpolation: str) -> None:
        super(Resize).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.Resize(size, interpolation)
        ])


class ToTensor(object):
    def __call__(self, image: np.ndarray, target: np.ndarray) -> [torch.Tensor, torch.Tensor]:
        # Convert Numpy format to PyTorch format
        image_tensor = F.to_tensor(image)

        target_tensor = torch.zeros((len(target), 6))
        target_tensor[:, 1:] = F.to_tensor(target)

        return image_tensor, target_tensor


class DefaultAugment(ImageAugment):
    def __init__(self):
        super(DefaultAugment, self).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-10, 10)),
            iaa.Fliplr(0.5),
        ])


class ImproveAugment(ImageAugment):
    def __init__(self):
        super(ImproveAugment, self).__init__()
        self.augment_functions = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])
