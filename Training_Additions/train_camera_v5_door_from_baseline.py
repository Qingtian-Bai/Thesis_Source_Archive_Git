"""GPU-only two-stage fine-tuning for the five-class door model.

Stage 1 warms up the new five-class detection head while most of the
pretrained backbone is frozen. Stage 2 unfreezes the model and fine-tunes at a
lower learning rate. The frozen baseline and camera_v2 dataset are untouched.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics_camera_v5_config"))

from ultralytics import YOLO


DATA = ROOT / "ultimate_exam_dataset_camera_v5_door" / "dataset.yaml"
BASELINE = ROOT / "Model_Vault" / "yolov10n_4class_v1_baseline_epoch78.pt"
RUNS = ROOT / "runs" / "detect"
STAGE1_NAME = "camera_v5_door_headwarm_20260729"
STAGE2_NAME = "camera_v5_door_finetune_20260729"


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. CPU training is intentionally disabled.")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def common_args() -> dict[str, object]:
    return {
        "data": str(DATA),
        "imgsz": 640,
        "batch": 16,
        "workers": 2,
        "device": 0,
        "project": str(RUNS),
        "exist_ok": False,
        "patience": 25,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 3.0,
        "amp": True,
        "seed": 20260729,
        "deterministic": True,
        "plots": True,
        "cls_remap": True,
        # The door source already has offline flip/lighting variants.
        # Keep online augmentation useful but conservative.
        "hsv_h": 0.01,
        "hsv_s": 0.35,
        "hsv_v": 0.25,
        "degrees": 5.0,
        "translate": 0.08,
        "scale": 0.30,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "erasing": 0.1,
        "close_mosaic": 10,
    }


def main() -> None:
    require_gpu()
    for required in (DATA, BASELINE):
        if not required.exists():
            raise FileNotFoundError(required)
    for name in (STAGE1_NAME, STAGE2_NAME):
        if (RUNS / name).exists():
            raise SystemExit(f"Training output already exists: {RUNS / name}")

    print(f"Dataset: {DATA}")
    print(f"Starting weights: {BASELINE}")
    print("Stage 1/2: warm up the new five-class head")
    stage1 = YOLO(str(BASELINE))
    args1 = common_args()
    stage1.train(
        **args1,
        name=STAGE1_NAME,
        epochs=15,
        freeze=10,
        lr0=0.0015,
        lrf=0.05,
    )

    stage1_best = RUNS / STAGE1_NAME / "weights" / "best.pt"
    if not stage1_best.exists():
        raise RuntimeError(f"Stage-1 checkpoint missing: {stage1_best}")

    print("Stage 2/2: unfreeze and fine-tune all layers")
    stage2 = YOLO(str(stage1_best))
    args2 = common_args()
    stage2.train(
        **args2,
        name=STAGE2_NAME,
        epochs=100,
        freeze=None,
        lr0=0.0008,
        lrf=0.05,
    )

    stage2_best = RUNS / STAGE2_NAME / "weights" / "best.pt"
    if not stage2_best.exists():
        raise RuntimeError(f"Final checkpoint missing: {stage2_best}")
    print(f"FINAL_BEST={stage2_best}")


if __name__ == "__main__":
    main()
