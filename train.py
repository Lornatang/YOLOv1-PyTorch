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
import shutil
import time
from enum import Enum

import torch
import torch.optim as optim
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import ImageDataset, CUDAPrefetcher
from model import YOLOv1Tiny, YOLOv1, YOLOLoss
from utils import calculate_map, nms, convert_cell_boxes_to_boxes


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_map = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    model = build_model()
    print("Build model successfully.")

    yolo_criterion = define_loss()
    print("Define YOLO loss functions successfully.")

    optimizer = define_optimizer(model)
    print("Define optimizer functions successfully.")

    print("Check whether the resume model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_map = checkpoint["best_mAP"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        print("Loaded resume model weights.")

    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    for epoch in range(start_epoch, config.epochs):
        train(model,
              train_prefetcher,
              yolo_criterion,
              optimizer,
              epoch,
              scaler,
              writer)
        map_value = validate(model, test_prefetcher, epoch, writer, "test")
        print("\n")

        # Automatically save the model with the highest index
        is_best = map_value > best_map
        best_map = max(map_value, best_map)
        torch.save({"epoch": epoch + 1,
                    "best_mAP": best_map,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "best.pth.tar"))

        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "last.pth.tar"))


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = ImageDataset(config.train_file_index_path,
                                  config.images_dir,
                                  config.annotations_dir,
                                  config.image_size,
                                  config.model_num_grid,
                                  config.model_num_bboxes,
                                  config.model_num_classes,
                                  "train")
    test_datasets = ImageDataset(config.test_file_index_path,
                                 config.images_dir,
                                 config.annotations_dir,
                                 config.image_size,
                                 config.model_num_grid,
                                 config.model_num_bboxes,
                                 config.model_num_classes,
                                 "test")

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    if config.model_arch == "YOLOv1":
        model = YOLOv1(config.model_num_grid, config.model_num_bboxes, config.model_num_classes)
    elif config.model_arch == "YOLOv1Tiny":
        model = YOLOv1Tiny(config.model_num_grid, config.model_num_bboxes, config.model_num_classes)
    else:
        print("unrecognized model schema name, calling `YOLOv1Tiny` model")
        model = YOLOv1Tiny(config.model_num_grid, config.model_num_bboxes, config.model_num_classes)

    # Transfer to CUDA
    model = model.to(device=config.device, memory_format=torch.channels_last)

    return model


def define_loss() -> YOLOLoss:
    criterion = nn.MSELoss(reduction="sum")
    # Transfer to CUDA
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)

    yolo_criterion = YOLOLoss(criterion, config.model_num_grid, config.model_num_bboxes, config.model_num_classes)
    # Transfer to CUDA
    yolo_criterion = yolo_criterion.to(device=config.device, memory_format=torch.channels_last)

    return yolo_criterion


def define_optimizer(model: nn.Module) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(),
                           config.model_lr,
                           config.model_betas,
                           weight_decay=config.model_weight_decay)

    return optimizer


def train(model: nn.Module,
          train_prefetcher: CUDAPrefetcher,
          yolo_criterion: YOLOLoss,
          optimizer: optim.Adam,
          epoch: int,
          scaler: amp.GradScaler,
          writer: SummaryWriter) -> None:
    """Training main program

    Args:
        model (nn.Module): YOLO model
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        yolo_criterion (YOLOLoss): Calculate the feature difference between real samples and fake samples by the feature extraction model
        optimizer (optim.Adam): an optimizer for optimizing generator models in YOLO networks
        epoch (int): number of training epochs during training the YOLO network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the YOLO network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device,
                                        memory_format=torch.channels_last,
                                        non_blocking=True)
        target = batch_data["annotation"].to(device=config.device,
                                             memory_format=torch.channels_last,
                                             non_blocking=True)
        # Initialize the YOLO model gradients
        model.zero_grad(set_to_none=True)

        # Back-propagate and update gradient
        with amp.autocast():
            output = model(images)
            loss = yolo_criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), images.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("train/Loss", loss.item(), iters)
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(model: nn.Module,
             data_prefetcher: CUDAPrefetcher,
             epoch: int,
             writer: SummaryWriter,
             mode: str) -> [float, float]:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the YOLO network
        writer (SummaryWriter): log file management function
        mode (str): test validation dataset accuracy or test dataset mAP

    """
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
            with amp.autocast():
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
    print(f"* mAP: {map_value:.2f}")

    if mode == "valid" or mode == "test":
        writer.add_scalar(f"{mode}/mAP", map_value, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `valid` or `test`.")

    return map_value


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == "__main__":
    main()
