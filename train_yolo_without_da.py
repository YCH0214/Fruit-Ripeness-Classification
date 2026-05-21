# -*- coding: utf-8 -*-
"""Train YOLOv11 without data augmentation."""

from pathlib import Path
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parent
    yaml_path = project_root / "datasets" / "data.yaml"
    save_project_path = project_root / "runs"

    model = YOLO("yolo11s.pt")

    model.train(
        data=yaml_path,
        project=save_project_path,
        epochs=100,
        imgsz=640,
        batch=64,
        workers=8,
        device=0,
        name="Baseline",

        # Disable data augmentation
        mosaic=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        copy_paste=0.0,
        mixup=0.0,
        auto_augment=None,
        erasing=0.0,
    )


if __name__ == "__main__":
    main()
