"""Run the frozen 2026-07-29 five-class success baseline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


SNAPSHOT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SNAPSHOT_ROOT.parents[1]
sys.dont_write_bytecode = True
APP_PATH = SNAPSHOT_ROOT / "code" / "monitor.py"
MODEL_PATH = (
    SNAPSHOT_ROOT / "weights" / "yolov10n_5class_door_v1_frozen.pt"
)

os.environ["CUSTOM_OBJECT_CONFIDENCE"] = "0.50"
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(WORKSPACE_ROOT / ".ultralytics_frozen_v5_config")
)


def load_monitor_class():
    spec = importlib.util.spec_from_file_location("frozen_v5_monitor", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen monitor: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep runtime evidence, tracker files and caches outside the frozen
    # snapshot so launching it cannot mutate its own archived contents.
    module.PROJECT_ROOT = WORKSPACE_ROOT
    return module.DualCoreAntiCheatingSystem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen five-class door success baseline"
    )
    parser.add_argument("--source", default="0")
    parser.add_argument("--save-output")
    parser.add_argument("--save-raw")
    arguments = parser.parse_args()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    # The frozen app loads the shared YOLO pose/COCO weights by filename.
    os.chdir(WORKSPACE_ROOT)
    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    monitor_class = load_monitor_class()
    system = monitor_class(custom_model_path=MODEL_PATH)
    if system.DOOR_CLASS_ID != 4:
        raise RuntimeError(f"Expected door class 4, got {system.DOOR_CLASS_ID}")

    print(f"[FROZEN V5] snapshot={SNAPSHOT_ROOT}")
    print(f"[FROZEN V5] model={MODEL_PATH}")
    print("[FROZEN V5] confidence=0.50; door=neutral context")
    system.run_forever(
        source=source,
        save_output=arguments.save_output,
        save_raw=arguments.save_raw,
    )


if __name__ == "__main__":
    main()
