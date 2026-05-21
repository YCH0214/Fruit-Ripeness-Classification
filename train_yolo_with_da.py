# -*- coding: utf-8 -*-
"""Train YOLOv11 with data augmentation."""

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
        epochs=200,
        imgsz=640,
        batch=64,
        workers=8,
        device=0,
        name="DA_200",

        # Data augmentation
        mosaic=1.0,      # 100% use mosaic augmentation
        mixup=0.1,       # Mix two images to improve robustness
        degrees=20.0,    # Random rotation
        translate=0.1,   # Random translation
        scale=0.5,       # Random scaling
        fliplr=0.5,      # 50% horizontal flip

        # Color and lighting augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
