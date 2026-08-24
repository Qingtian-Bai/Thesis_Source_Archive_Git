"""Fast tests for model-role wiring and spatial evidence cooldown keys."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
APP = RELEASE_ROOT / "code" / "hybrid_monitor.py"
TEST_OUTPUT_ROOT = RELEASE_ROOT / "runtime_outputs" / "unit_test"
TEST_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
YOLO_CONFIG_ROOT = TEST_OUTPUT_ROOT / ".ultralytics_config"
(YOLO_CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_ROOT)
os.environ["TRACKER_CONFIG_PATH"] = str(TEST_OUTPUT_ROOT / "auto_exam_tracker.yaml")


def load_class():
    spec = importlib.util.spec_from_file_location("hybrid_logic_test", APP)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import monitor: {APP}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DualCoreAntiCheatingSystem


def main():
    monitor_class = load_class()
    monitor = object.__new__(monitor_class)
    monitor.EVIDENCE_SPATIAL_CELL_PIXELS = 96
    monitor.EVIDENCE_SPATIAL_MATCH_PIXELS = 128
    monitor.CAPTURE_COOLDOWN = 5.0
    monitor.object_evidence_cooldowns = []

    phone_a = {
        "class_id": 1,
        "bbox_xyxy": [100, 100, 140, 180],
        "track_ids": [7],
    }
    same_phone = {
        "class_id": 1,
        "bbox_xyxy": [105, 102, 145, 182],
        "track_ids": [7],
    }
    other_student = {
        "class_id": 1,
        "bbox_xyxy": [105, 102, 145, 182],
        "track_ids": [8],
    }
    other_location = {
        "class_id": 1,
        "bbox_xyxy": [400, 300, 440, 380],
        "track_ids": [7],
    }
    note_same_place = {
        "class_id": 3,
        "bbox_xyxy": [105, 102, 145, 182],
        "track_ids": [7],
    }

    key = monitor._object_cooldown_key
    assert key(1, phone_a) == key(1, same_phone)
    assert key(1, phone_a) != key(1, other_student)
    assert key(1, phone_a) != key(1, other_location)
    assert key(1, phone_a) != key(3, note_same_place)
    monitor._remember_object_evidence(1, phone_a, 10.0)
    assert monitor._object_is_in_spatial_cooldown(1, same_phone, 11.0)
    assert monitor._object_is_in_spatial_cooldown(1, other_student, 11.0)
    assert not monitor._object_is_in_spatial_cooldown(1, other_location, 11.0)
    assert not monitor._object_is_in_spatial_cooldown(3, note_same_place, 11.0)
    assert not monitor._object_is_in_spatial_cooldown(1, same_phone, 16.0)
    print("UT-01 hybrid cooldown-key tests: PASS")


if __name__ == "__main__":
    main()
