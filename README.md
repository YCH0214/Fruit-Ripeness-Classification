# Fruit Ripeness Classification

This project trains YOLOv11 models for fruit ripeness detection/classification.
The current system focuses on five fruit types: apple, orange, tomato, banana, and mango.
Each fruit is classified by ripeness level: unripe, ripe, and overripe.

The repository includes two training scripts for comparing model performance with and without data augmentation.

## Demo

Try the demo on Hugging Face Spaces:

https://huggingface.co/spaces/jerrychen76/fruit-ripeness-recognition

## Requirements

Install the required Python package before training:

```bash
pip install ultralytics
```

## Dataset

The original dataset was collected from Hugging Face and Kaggle.
Due to dataset licensing and repository size limits, the full dataset is not included in this repository.

Only sample images are kept in the `datasets` folder for reference, with about 20 images per sample group.
To train the model, download the full dataset from the original sources and place it inside the `datasets` folder.

The training scripts read the dataset configuration from:

```text
datasets/data.yaml
```

Example `data.yaml` format:

```yaml
path: datasets
train: train/images
val: valid/images
test: test/images

names:
  0: class_name_0
  1: class_name_1
  2: class_name_2
```

Adjust the class names and paths according to the actual dataset.

## Training

Train YOLOv11 with data augmentation:

```bash
python train_yolo_with_da.py
```

Train YOLOv11 without data augmentation:

```bash
python train_yolo_without_da.py
```

## Experiments

Training results are saved under the `runs` directory.
