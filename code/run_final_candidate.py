"""Reproducible entry point for the frozen pre-test dissertation candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "final_config.yaml"
APP_PATH = ROOT / "code" / "final_monitor.py"


ENVIRONMENT_MAP = {
    "runtime.pose_confidence": "POSE_CONFIDENCE",
    "runtime.pose_track_confidence": "POSE_TRACK_CONFIDENCE",
    "runtime.max_students": "MAX_STUDENTS",
    "runtime.camera_frame_queue_size": "CAMERA_FRAME_QUEUE_SIZE",
    "runtime.camera_reopen_interval": "CAMERA_REOPEN_INTERVAL",
    "runtime.state_max_step_seconds": "STATE_MAX_STEP_SECONDS",
    "runtime.crossing_trigger_seconds": "CROSSING_TRIGGER_SECONDS",
    "runtime.occlusion_grace_seconds": "OCCLUSION_GRACE_SECONDS",
    "custom_detector.confidence": "CUSTOM_OBJECT_CONFIDENCE",
    "custom_detector.phone_person_margin_pixels": "PHONE_PERSON_MARGIN",
    "custom_detector.phone_max_frame_area_ratio": "PHONE_MAX_FRAME_AREA_RATIO",
    "custom_detector.door_neutral_confidence": "DOOR_NEUTRAL_CONFIDENCE",
    "custom_detector.paper_neutral_confidence": "PAPER_NEUTRAL_CONFIDENCE",
    "custom_detector.paper_counter_confidence": "PAPER_COUNTER_CONFIDENCE",
    "note_detector.confidence": "NOTE_DETECTION_CONFIDENCE",
    "note_detector.inference_size": "NOTE_INFERENCE_SIZE",
    "note_detector.confirm_seconds": "NOTE_CONFIRM_SECONDS",
    "note_detector.track_gap_seconds": "NOTE_TRACK_GAP_SECONDS",
    "note_detector.transfer_window_seconds": "NOTE_TRANSFER_WINDOW_SECONDS",
    "note_detector.minimum_hand_observations": "NOTE_MIN_HAND_OBSERVATIONS",
    "note_detector.hand_distance_factor": "NOTE_HAND_DISTANCE_FACTOR",
    "note_detector.max_person_area_ratio": "NOTE_MAX_PERSON_AREA_RATIO",
    "note_detector.max_shoulder_width_ratio": "NOTE_MAX_SHOULDER_WIDTH_RATIO",
    "note_detector.max_shoulder_height_ratio": "NOTE_MAX_SHOULDER_HEIGHT_RATIO",
    "coco_assist.mode": "COCO_ASSIST_MODE",
    "coco_assist.phone_confidence": "COCO_PHONE_CONFIDENCE",
    "coco_assist.inference_size": "COCO_PHONE_INFERENCE_SIZE",
    "coco_assist.minimum_observations": "COCO_MIN_OBERVATIONS",
    "coco_assist.confirm_seconds": "COCO_CONFIRM_SECONDS",
    "coco_assist.max_gap_seconds": "COCO_MAX_GAP_SECONDS",
    "coco_assist.distractor_confidence": "COCO_DISTRACTOR_CONFIDENCE",
    "coco_assist.distractor_iou": "COCO_DISTRACTOR_IOU",
    "coco_assist.person_margin_ratio": "COCO_PERSON_MARGIN_RATIO",
    "coco_assist.person_margin_min_pixels": "COCO_PERSON_MARGIN_MIN",
    "coco_assist.minimum_aspect_ratio": "COCO_MIN_ASPECT_RATIO",
    "coco_assist.strong_confidence": "COCO_STRONG_CONFIDENCE",
    "coco_assist.rotated_support_angle_degrees": "COCO_ROTATED_SUPPORT_ANGLE",
    "coco_assist.rotated_support_confidence": "COCO_ROTATED_SUPPORT_CONFIDENCE",
    "coco_assist.rotated_support_iou": "COCO_ROTATED_SUPPORT_IOU",
    "evidence.capture_cooldown_seconds": "CAPTURE_COOLDOWN_SECONDS",
    "evidence.event_pre_seconds": "EVENT_PRE_SECONDS",
    "evidence.event_post_seconds": "EVENT_POST_SECONDS",
    "evidence.minimum_free_space_mb": "EVENT_MIN_FREE_MB",
}


def nested_value(data, dotted_key):
    value = data
    for part in dotted_key.split("."):
        value = value[part]
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions():
    names = ["ultralytics", "torch", "torchvision", "opencv-python", "numpy", "PyYAML"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_details():
    details = {"cuda_available": False}
    try:
        import torch

        details["cuda_available"] = bool(torch.cuda.is_available())
        details["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            details["device_count"] = torch.cuda.device_count()
            details["device_names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as error:
        details["probe_error"] = repr(error)
    return details


def load_monitor_class():
    spec = importlib.util.spec_from_file_location("final_candidate_monitor", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import monitor: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DualCoreAntiCheatingSystem


def write_manifest(manifest, unique_path):
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    unique_path.write_text(text, encoding="utf-8")
    (ROOT / "run_manifest.json").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Frozen dissertation candidate")
    parser.add_argument("--source", default="0")
    parser.add_argument("--save-output")
    parser.add_argument("--save-raw")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--exit-on-eof", action="store_true")
    parser.add_argument("--max-frames", type=int)
    arguments = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    model_path = ROOT / config["model"]["custom_weight"]
    for required in (CONFIG_PATH, APP_PATH, model_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    for dotted_key, environment_name in ENVIRONMENT_MAP.items():
        os.environ[environment_name] = str(nested_value(config, dotted_key))
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics_config")
    sys.dont_write_bytecode = True

    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    source_path = None if isinstance(source, int) else Path(source).expanduser().resolve()
    started = datetime.now(timezone.utc)
    manifest_path = ROOT / "manifests" / f"run_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    manifest = {
        "release_id": config["release_id"],
        "status": "starting",
        "started_at_utc": started.isoformat(),
        "command_arguments": vars(arguments),
        "source": {
            "value": str(source),
            "kind": "camera" if isinstance(source, int) else "file",
            "sha256": sha256(source_path) if source_path and source_path.is_file() else None,
        },
        "artifacts": {
            "config": {"path": str(CONFIG_PATH), "sha256": sha256(CONFIG_PATH)},
            "code": {"path": str(APP_PATH), "sha256": sha256(APP_PATH)},
            "weight": {"path": str(model_path), "sha256": sha256(model_path)},
        },
        "expected_classes": config["model"]["expected_classes"],
        "resolved_configuration": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "device": device_details(),
        },
    }
    write_manifest(manifest, manifest_path)

    os.chdir(WORKSPACE_ROOT)
    try:
        monitor_class = load_monitor_class()
        system = monitor_class(custom_model_path=model_path)
        manifest["status"] = "running"
        write_manifest(manifest, manifest_path)
        system.run_forever(
            source=source,
            save_output=arguments.save_output,
            save_raw=arguments.save_raw,
            display=not arguments.no_display,
            exit_on_eof=arguments.exit_on_eof,
            max_frames=arguments.max_frames,
        )
        manifest["status"] = "completed"
    except Exception:
        manifest["status"] = "failed"
        manifest["exception"] = traceback.format_exc()
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest, manifest_path)


if __name__ == "__main__":
    main()
