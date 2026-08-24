"""Reproducible entry point for the post-final hybrid v2 experiment."""

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


CODE_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CODE_DIR.parent
CONFIG_PATH = RELEASE_ROOT / "config" / "hybrid_config.yaml"
APP_PATH = CODE_DIR / "hybrid_monitor.py"
RUNTIME_OUTPUT_ROOT = RELEASE_ROOT / "runtime_outputs"
MANIFEST_DIR = RUNTIME_OUTPUT_ROOT / "manifests"
LATEST_MANIFEST_PATH = RUNTIME_OUTPUT_ROOT / "run_manifest.json"


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
    "evidence.output_directory": "EVIDENCE_OUTPUT_DIR",
    "evidence.spatial_cell_pixels": "EVIDENCE_SPATIAL_CELL_PIXELS",
    "evidence.spatial_match_pixels": "EVIDENCE_SPATIAL_MATCH_PIXELS",
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
    spec = importlib.util.spec_from_file_location("post_final_hybrid_v2_monitor", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import monitor: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DualCoreAntiCheatingSystem


def write_manifest(manifest, unique_path):
    unique_path.parent.mkdir(parents=True, exist_ok=True)
    LATEST_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    unique_path.write_text(text, encoding="utf-8")
    LATEST_MANIFEST_PATH.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Frozen 6FE329 post-test regression release")
    parser.add_argument("--source", default="0")
    parser.add_argument("--save-output")
    parser.add_argument("--save-raw")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--exit-on-eof", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--evidence-dir")
    arguments = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    phone_model_path = RELEASE_ROOT / config["model"]["phone_weight"]
    note_context_model_path = RELEASE_ROOT / config["model"]["note_context_weight"]
    pose_model_path = RELEASE_ROOT / config["model"]["pose_weight"]
    coco_model_path = RELEASE_ROOT / config["model"]["coco_weight"]
    for required in (
        CONFIG_PATH,
        APP_PATH,
        phone_model_path,
        note_context_model_path,
        pose_model_path,
        coco_model_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    for dotted_key, environment_name in ENVIRONMENT_MAP.items():
        os.environ[environment_name] = str(nested_value(config, dotted_key))
    if arguments.evidence_dir:
        evidence_path = Path(arguments.evidence_dir).expanduser()
        if not evidence_path.is_absolute():
            evidence_path = RELEASE_ROOT / evidence_path
        os.environ["EVIDENCE_OUTPUT_DIR"] = str(evidence_path.resolve())
    os.environ["POSE_MODEL_PATH"] = str(pose_model_path)
    os.environ["COCO_MODEL_PATH"] = str(coco_model_path)
    os.environ["TRACKER_CONFIG_PATH"] = str(
        RUNTIME_OUTPUT_ROOT / "auto_exam_tracker.yaml"
    )
    yolo_config_root = RUNTIME_OUTPUT_ROOT / ".ultralytics_config"
    (yolo_config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_root)
    sys.dont_write_bytecode = True

    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    source_path = None
    if not isinstance(source, int):
        source_path = Path(source).expanduser()
        if not source_path.is_absolute():
            release_candidate = RELEASE_ROOT / source_path
            if release_candidate.exists():
                source_path = release_candidate
        source_path = source_path.resolve()
        source = str(source_path)
    started = datetime.now(timezone.utc)
    manifest_path = MANIFEST_DIR / f"run_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
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
            "phone_weight": {
                "path": str(phone_model_path),
                "sha256": sha256(phone_model_path),
            },
            "note_context_weight": {
                "path": str(note_context_model_path),
                "sha256": sha256(note_context_model_path),
            },
            "pose_weight": {
                "path": str(pose_model_path),
                "sha256": sha256(pose_model_path),
            },
            "coco_weight": {
                "path": str(coco_model_path),
                "sha256": sha256(coco_model_path),
            },
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

    os.chdir(RELEASE_ROOT)
    try:
        monitor_class = load_monitor_class()
        system = monitor_class(
            phone_model_path=phone_model_path,
            note_context_model_path=note_context_model_path,
        )
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
