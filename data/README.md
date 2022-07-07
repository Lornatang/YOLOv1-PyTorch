# Usage

## Step1: Download datasets

Contains VOC2007, VOC2012, COCO2014, COCO2017 and more.

- [Google Driver](https://drive.google.com/drive/folders/1kUTpfNUP8C3lKH_mSHAyIxS8auVzaSYQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1UsLQvMLbm1uhv-tYTL2q-w?pwd=llot)

## Step2: Prepare the dataset in the following format (e.g VOC2007, 2012)

Unzip the tar file to the `VOCdevkit` folder.

```text
# Dataset struct
- VOCdevkit
    - VOC2007
        - Annotations
            - 000001.xml
            ...
        - ImageSets
            - Layout
            - Main
                - train.txt
            - Segmentation
        - JPEGImages
            - 000001.jpg
            ...
        - SegmentationClass
        - SegmentationObject
    - VOC2012
        - Annotations
            - 000001.xml
            ...
        - ImageSets
            - Layout
            - Main
                - train.txt
            - Segmentation
        - JPEGImages
            - 000001.jpg
            ...
        - SegmentationClass
        - SegmentationObject
```

## Step3: Preprocess the dataset

```bash
python run.py
```

## Step4: Check that the final dataset directory schema is completely correct

```text
# Train dataset
- YOLO
    - images
        - 2008_000008.jpg
        - 2008_000015.jpg
        ...
    - annotations
        - 2008_000008.txt
        - 2008_000015.txt
        - ...
    - train.txt
    - test.txt
        - train
        - valid
        - original

- VOCdevkit
    - ...
- voc_classes.txt
- voc_colors.txt
```
