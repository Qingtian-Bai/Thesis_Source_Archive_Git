"""Run the immutable paper/note v1.3 success baseline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[1]
APP_PATH = EXPERIMENT_ROOT / "monitor.py"
MODEL_PATH = (
    EXPERIMENT_ROOT
    / "weights"
    / "yolov10n_5class_paper_note_v1_success_frozen.pt"
)

sys.dont_write_bytecode = True
os.environ.setdefault(
    "YOLO_CONFIG_DIR", str(WORKSPACE_ROOT / ".ultralytics_paper_note_v6_frozen_config")
)
os.environ["CUSTOM_OBJECT_CONFIDENCE"] = "0.50"
os.environ["PAPER_NEUTRAL_CONFIDENCE"] = "0.55"
os.environ["PAPER_COUNTER_CONFIDENCE"] = "0.20"
os.environ["NOTE_DETECTION_CONFIDENCE"] = "0.30"
os.environ["NOTE_INFERENCE_SIZE"] = "960"
os.environ["NOTE_CONFIRM_FRAMES"] = "4"


def load_monitor_class():
    spec = importlib.util.spec_from_file_location("paper_note_v6_frozen_monitor", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import experiment app: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Runtime evidence and tracker configuration stay in the workspace, not
    # inside the experiment's archived source directory.
    module.PROJECT_ROOT = WORKSPACE_ROOT
    return module.DualCoreAntiCheatingSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen paper/note v1.3 success baseline")
    parser.add_argument("--source", default="0")
    parser.add_argument("--save-output")
    parser.add_argument("--save-raw")
    arguments = parser.parse_args()

    for required in (APP_PATH, MODEL_PATH):
        if not required.is_file():
            raise FileNotFoundError(required)
    os.chdir(WORKSPACE_ROOT)
    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    monitor_class = load_monitor_class()
    system = monitor_class(custom_model_path=MODEL_PATH)
    if system.yolo_custom.names.get(3) != "note":
        raise RuntimeError(f"Unexpected model classes: {system.yolo_custom.names}")

    print(f"[FROZEN PAPER/NOTE V1.3] model={MODEL_PATH}")
    print("[FROZEN PAPER/NOTE V1.3] paper=neutral context")
    print(
        "[FROZEN PAPER/NOTE V1.3] note=0.30 proposal + hand association + "
        "4-frame confirmation; red evidence requires transfer between "
        "two students"
    )
    system.run_forever(
        source=source,
        save_output=arguments.save_output,
        save_raw=arguments.save_raw,
    )


if __name__ == "__main__":
    main()
