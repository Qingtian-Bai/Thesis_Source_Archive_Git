"""GPU-only conservative fine-tuning for paper/note discrimination.

The successful frozen five-class model is the starting point. The class order
does not change, so this is a low-learning-rate refinement rather than a new
detection head. Existing phone and door knowledge remains present in every
training batch.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(ROOT / ".ultralytics_camera_v6_paper_note_config")
)

from ultralytics import YOLO


DATA = ROOT / "ultimate_exam_dataset_camera_v6_paper_note" / "dataset.yaml"
STARTING_WEIGHT = (
    ROOT
    / "Baselines"
    / "baseline_20260729_v5_door_success"
    / "weights"
    / "yolov10n_5class_door_v1_frozen.pt"
)
RUNS = ROOT / "runs" / "detect"
STAGE1_NAME = "paper_note_v1_head_refine_20260729"
STAGE2_NAME = "paper_note_v1_full_refine_20260729"


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
        # Small notes benefit from more detector input pixels than the 640
        # phone baseline. Runtime uses a matching 960 note-only pass.
        "imgsz": 960,
        "batch": 6,
        "workers": 2,
        "device": 0,
        "project": str(RUNS),
        "exist_ok": False,
        "patience": 12,
        "optimizer": "AdamW",
        "weight_decay": 0.0005,
        "cos_lr": True,
        "warmup_epochs": 2.0,
        "amp": True,
        "seed": 20260729,
        "deterministic": True,
        "plots": True,
        # Offline v6 already supplies flipped and distance-simulated contrast
        # pairs. Keep online augmentation conservative to protect phone/door.
        "hsv_h": 0.008,
        "hsv_s": 0.25,
        "hsv_v": 0.20,
        "degrees": 4.0,
        "translate": 0.06,
        "scale": 0.18,
        "shear": 0.0,
        "perspective": 0.0003,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "erasing": 0.05,
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
                "do not overwrite an earlier experiment"
            )

    print(f"DATA={DATA}")
    print(f"STARTING_WEIGHT={STARTING_WEIGHT}")
    print("Stage 1/2: refine detection head while freezing the backbone")
    stage1 = YOLO(str(STARTING_WEIGHT))
    stage1.train(
        **common_args(),
        name=STAGE1_NAME,
        epochs=8,
        freeze=10,
        lr0=0.0005,
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
        epochs=45,
        freeze=None,
        lr0=0.00025,
        lrf=0.08,
    )
    final_best = RUNS / STAGE2_NAME / "weights" / "best.pt"
    if not final_best.is_file():
        raise RuntimeError(f"Missing final checkpoint: {final_best}")
    print(f"FINAL_BEST={final_best}")


if __name__ == "__main__":
    main()
