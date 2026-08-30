"""GPU-only conservative refinement on the reviewed phone hard cases.

The frozen paper/note v1.3 success weight is used as the starting point.
The class order remains unchanged.  This experiment writes to new run
directories and never replaces a frozen checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(ROOT / ".ultralytics_camera_v7_phone_config")
)

from ultralytics import YOLO


DATA = ROOT / "ultimate_exam_dataset_camera_v7_phone_hardcases" / "dataset.yaml"
STARTING_WEIGHT = (
    ROOT
    / "Baselines"
    / "baseline_20260730_v6_paper_note_success"
    / "weights"
    / "yolov10n_5class_paper_note_v1_success_frozen.pt"
)
RUNS = ROOT / "runs" / "detect"
STAGE1_NAME = "phone_hardcases_v1_head_refine_20260730"
STAGE2_NAME = "phone_hardcases_v1_full_refine_20260730"


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; CPU training is intentionally disabled.")
    print(f"CUDA={torch.version.cuda}")
    print(f"GPU={torch.cuda.get_device_name(0)}")
    properties = torch.cuda.get_device_properties(0)
    print(f"VRAM_MiB={properties.total_memory / 1024**2:.0f}")


def common_args() -> dict[str, object]:
    return {
        "data": str(DATA),
        "imgsz": 960,
        "batch": 6,
        "workers": 2,
        "device": 0,
        "project": str(RUNS),
        "exist_ok": False,
        "patience": 8,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 1.0,
        "amp": True,
        "seed": 20260730,
        "deterministic": True,
        "plots": True,
        # Keep augmentation gentle: the 100 reviewed frames are one
        # development scene, while v6 still supplies the broad base data.
        "hsv_h": 0.006,
        "hsv_s": 0.20,
        "hsv_v": 0.16,
        "degrees": 3.0,
        "translate": 0.04,
        "scale": 0.12,
        "shear": 0.0,
        "perspective": 0.0002,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "erasing": 0.03,
    }


def main() -> None:
    require_gpu()
    for required in (DATA, STARTING_WEIGHT):
        if not required.is_file():
            raise FileNotFoundError(required)
    for run_name in (STAGE1_NAME, STAGE2_NAME):
        if (RUNS / run_name).exists():
            raise SystemExit(
                f"Training output exists: {RUNS / run_name}; "
                "refusing to overwrite an earlier experiment"
            )

    print(f"DATA={DATA}")
    print(f"STARTING_WEIGHT={STARTING_WEIGHT}")
    print("Stage 1/2: six-epoch head refinement")
    stage1 = YOLO(str(STARTING_WEIGHT))
    stage1.train(
        **common_args(),
        name=STAGE1_NAME,
        epochs=6,
        freeze=10,
        lr0=0.00030,
        lrf=0.10,
    )
    stage1_best = RUNS / STAGE1_NAME / "weights" / "best.pt"
    if not stage1_best.is_file():
        raise RuntimeError(f"Missing stage-1 checkpoint: {stage1_best}")

    print("Stage 2/2: low-rate whole-model refinement")
    stage2 = YOLO(str(stage1_best))
    stage2.train(
        **common_args(),
        name=STAGE2_NAME,
        epochs=24,
        freeze=None,
        lr0=0.00015,
        lrf=0.08,
    )
    final_best = RUNS / STAGE2_NAME / "weights" / "best.pt"
    if not final_best.is_file():
        raise RuntimeError(f"Missing final checkpoint: {final_best}")
    print(f"FINAL_BEST={final_best}")


if __name__ == "__main__":
    main()
