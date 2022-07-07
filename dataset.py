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
import queue
import threading

import cv2
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset, DataLoader

from imgproc import image_to_tensor


class ImageDataset(Dataset):
    def __init__(self,
                 file_index_path: str,
                 images_dir: str,
                 labels_dir: str,
                 image_size: int,
                 num_grid: int,
                 num_bboxes: int,
                 num_classes: int,
                 mode: str):
        # Get all image file names from directory
        self.files_index = []
        with open(file_index_path, "r") as f:
            for file_name in f.readlines():
                self.files_index.append(os.path.basename(file_name.strip("\n")))

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.num_grid = num_grid
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Get image and annotation path
        image_path = os.path.join(self.images_dir, self.files_index[batch_index])
        label_path = os.path.join(self.labels_dir, self.files_index[batch_index].split(".")[0] + ".txt")
        target = []
        with open(label_path) as f:
            for line in f.readlines():
                class_index, pos_x, pos_y, width, height = [float(x) if float(x) != int(float(x)) else int(x)
                                                            for x in line.replace("\n", "").split()]

                target.append([class_index, pos_x, pos_y, width, height])

        # Read a batch of image data
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Image and annotation processing operations
        if self.mode == "train":
            image_augment_functions = iaa.Sequential([
                iaa.ChangeColorspace("HSV", "RGB"),
                iaa.AddToBrightness((-50, 50), "HSV", "HSV"),
                iaa.AddToSaturation((-50, 50), "HSV"),
                iaa.ChangeColorspace("RGB", "HSV"),
                iaa.Resize([self.image_size, self.image_size], "cubic"),
            ])
        elif self.mode == "valid" or self.mode == "test":
            image_augment_functions = iaa.Sequential([
                iaa.Resize([self.image_size, self.image_size], "cubic"),
            ])
        else:
            raise ValueError("Unsupported data processing model, please use `train` or `valid`.")

        # Apply image augment
        image = image_augment_functions.augment_image(image)

        # Convert image to `torch.Tensor` format
        image = image_to_tensor(image, False, False)

        # Convert To Cells
        target_matrix = torch.zeros((self.num_grid, self.num_grid, self.num_classes + 5 * self.num_bboxes))
        for class_index, pos_x, pos_y, width, height in target:
            class_index = int(class_index)

            # i,j represents the cell row and cell column
            i, j = int(self.num_grid * pos_y), int(self.num_grid * pos_x)
            pos_x_cell, pos_y_cell = self.num_grid * pos_x - j, self.num_grid * pos_y - i
            width_cell, height_cell = (width * self.num_grid, height * self.num_grid)

            # If no object already found for specific cell i,j
            if target_matrix[i, j, 20] == 0:
                # Set that there exists an object
                target_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor([pos_x_cell, pos_y_cell, width_cell, height_cell])

                target_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                target_matrix[i, j, class_index] = 1

        return {"image_path": image_path, "image": image, "target": target_matrix}

    def __len__(self):
        return len(self.files_index)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
