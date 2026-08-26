"""Deterministic fault-injection tests for the frozen hybrid release.

The tests use synthetic frames, fake capture devices, temporary directories,
and mocked model objects.  They do not load participant recordings or execute
detector inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MONITOR_PATH = RELEASE_ROOT / "code" / "hybrid_monitor.py"
LAUNCHER_PATH = RELEASE_ROOT / "code" / "run_hybrid_v2.py"
DEFAULT_OUTPUT_DIR = (
    RELEASE_ROOT
    / "runtime_outputs"
    / "fault_test_runs"
    / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
)
TEST_RUNTIME_ROOT = RELEASE_ROOT / "runtime_outputs" / "fault_test"
YOLO_CONFIG_ROOT = TEST_RUNTIME_ROOT / ".ultralytics_config"
(YOLO_CONFIG_ROOT / "Ultralytics").mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(YOLO_CONFIG_ROOT)
os.environ["TRACKER_CONFIG_PATH"] = str(TEST_RUNTIME_ROOT / "auto_exam_tracker.yaml")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MONITOR = load_module("frozen_hybrid_fault_test", MONITOR_PATH)


@dataclass
class TestResult:
    test_id: str
    name: str
    status: str
    duration_seconds: float
    detail: str
    exception: str | None = None


class RecordingWriter:
    """Small cv2.VideoWriter replacement that records written frame IDs."""

    instances = []

    def __init__(self, path, _fourcc, _fps, _shape):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"synthetic writer placeholder")
        self.frame_ids = []
        self.released = False
        RecordingWriter.instances.append(self)

    def isOpened(self):
        return True

    def write(self, frame):
        self.frame_ids.append(int(frame[0, 0, 0]))

    def release(self):
        self.released = True


def test_low_disk_degradation(work_root: Path):
    output_dir = work_root / "low_disk"
    buffer = MONITOR.EventVideoBuffer(
        output_dir,
        nominal_fps=10.0,
        pre_seconds=1.0,
        post_seconds=2.0,
        min_free_mb=200.0,
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    with patch.object(
        MONITOR.shutil,
        "disk_usage",
        return_value=SimpleNamespace(total=1_000_000, used=999_999, free=1),
    ), patch.object(MONITOR.cv2, "VideoWriter") as writer_factory:
        assert buffer._has_disk_space() is False
        assert buffer.disabled_for_space is True
        assert buffer.trigger("phone", 3.0, frame.shape) is None
        writer_factory.assert_not_called()
    return "Event-video creation was disabled while the monitoring path remained callable."


class FakeCameraCapture:
    def __init__(self, read_results):
        self.read_results = list(read_results)
        self.released = False

    def isOpened(self):
        return True

    def set(self, _property, _value):
        return True

    def get(self, property_id):
        if property_id == MONITOR.cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        if self.read_results:
            return self.read_results.pop(0)
        return False, None

    def release(self):
        self.released = True


def test_camera_disconnect_and_recovery(_work_root: Path):
    recovered_frame = np.full((3, 3, 3), 17, dtype=np.uint8)
    first = FakeCameraCapture([(False, None), (False, None)])
    second = FakeCameraCapture([(True, recovered_frame)])
    opened = []

    def factory(_source):
        capture = first if not opened else second
        opened.append(capture)
        return capture

    capture = None
    with patch.object(MONITOR.cv2, "VideoCapture", side_effect=factory):
        capture = MONITOR.RecentFrameCapture(
            0,
            reopen_interval=2,
            queue_size=3,
        )
        ok, frame, _captured_at, sequence, _dropped = capture.read_recent(
            0,
            timeout=2.0,
        )
        capture.stop()
    assert ok is True
    assert sequence >= 1
    assert int(frame[0, 0, 0]) == 17
    assert len(opened) >= 2
    assert first.released is True
    assert capture is not None and capture.thread.is_alive() is False
    return "Two failed reads triggered reopen; a frame from the replacement capture was delivered."


class TimestampCapture:
    def __init__(self):
        self.timestamps_ms = [0.0, 0.0, -1.0]
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, property_id):
        if property_id == MONITOR.cv2.CAP_PROP_FPS:
            return 10.0
        if property_id == MONITOR.cv2.CAP_PROP_FRAME_WIDTH:
            return 8.0
        if property_id == MONITOR.cv2.CAP_PROP_FRAME_HEIGHT:
            return 8.0
        if property_id == MONITOR.cv2.CAP_PROP_POS_MSEC:
            return self.timestamps_ms[max(0, self.index - 1)]
        return 0.0

    def read(self):
        if self.index >= len(self.timestamps_ms):
            return False, None
        frame = np.full((8, 8, 3), self.index, dtype=np.uint8)
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class EmptyBoxes:
    id = None

    def __len__(self):
        return 0


class EmptyModel:
    def __init__(self):
        self.result = SimpleNamespace(boxes=EmptyBoxes(), keypoints=None)

    def track(self, *_args, **_kwargs):
        return [self.result]

    def predict(self, *_args, **_kwargs):
        return [self.result]


class TimestampEventBuffer:
    latest = None

    def __init__(self, output_dir, **_kwargs):
        self.output_dir = Path(output_dir)
        self.event_times = []
        self.closed = False
        TimestampEventBuffer.latest = self

    def add_frame(self, _frame, event_time):
        self.event_times.append(float(event_time))

    def trigger(self, *_args, **_kwargs):
        return None

    def close(self):
        self.closed = True


def minimal_monitor(work_root: Path):
    monitor = object.__new__(MONITOR.DualCoreAntiCheatingSystem)
    monitor.note_tracks = {}
    monitor.next_note_track_id = 1
    monitor.crossing_states = {}
    monitor.track_states = {}
    monitor.frame_index = 0
    monitor.last_capture_time = {}
    monitor.object_evidence_cooldowns = []
    monitor.event_video_buffer = None
    monitor._active_capture = None
    monitor._active_writer = None
    monitor._active_raw_writer = None
    monitor.manual_exit_requested = False
    monitor.session_event_origin = None
    monitor.evidence_dir = work_root / "timestamp_evidence"
    monitor.CAMERA_REOPEN_INTERVAL = 2
    monitor.CAMERA_FRAME_QUEUE_SIZE = 3
    monitor.FPS_ESTIMATE = 10.0
    monitor.EVENT_PRE_SECONDS = 1.0
    monitor.EVENT_POST_SECONDS = 2.0
    monitor.EVENT_MIN_FREE_MB = 0.0
    monitor.TIME_THRESHOLD_SECONDS = 0.5
    monitor.NOTE_CONFIRM_SECONDS = 0.15
    monitor.NOTE_TRANSFER_WINDOW_SECONDS = 1.2
    monitor.MAX_STUDENTS = 5
    monitor.POSE_TRACK_CONFIDENCE = 0.25
    monitor.POSE_CONFIDENCE = 0.45
    monitor.tracker_path = "synthetic_tracker.yaml"
    monitor.DOOR_CLASS_ID = None
    monitor.CUSTOM_OBJECT_CONFIDENCE = 0.5
    monitor.PAPER_CLASS_ID = 2
    monitor.NOTE_CLASS_ID = 3
    monitor.PAPER_COUNTER_CONFIDENCE = 0.2
    monitor.NOTE_DETECTION_CONFIDENCE = 0.3
    monitor.NOTE_INFERENCE_SIZE = 960
    monitor.COCO_PHONE_CLASS = 67
    monitor.COCO_PHONE_CONFIDENCE = 0.15
    monitor.COCO_PHONE_INFERENCE_SIZE = 1280
    monitor.COCO_ASSIST_MODE = "counter_evidence"
    monitor.yolo_pose = EmptyModel()
    monitor.yolo_phone = EmptyModel()
    monitor.yolo_note_context = EmptyModel()
    monitor.yolo_coco = EmptyModel()
    monitor.update_scene_context = lambda *_args, **_kwargs: None
    monitor.update_note_candidates = lambda *_args, **_kwargs: []
    monitor.draw_note_candidates = lambda *_args, **_kwargs: []
    monitor.update_coco_phone_confirmation = lambda *_args, **_kwargs: None
    return monitor


def test_timestamp_fallback(work_root: Path):
    fake_capture = TimestampCapture()
    monitor = minimal_monitor(work_root)
    with patch.object(MONITOR.cv2, "VideoCapture", return_value=fake_capture), patch.object(
        MONITOR,
        "EventVideoBuffer",
        TimestampEventBuffer,
    ), patch.object(MONITOR.cv2, "putText", return_value=None), patch.object(
        MONITOR.cv2,
        "destroyAllWindows",
        return_value=None,
    ):
        monitor.run_live(
            source="synthetic_timestamp_input.mp4",
            display=False,
            exit_on_eof=True,
            max_frames=3,
        )
    assert TimestampEventBuffer.latest is not None
    observed = TimestampEventBuffer.latest.event_times
    assert len(observed) == 3
    assert np.allclose(observed, [0.0, 0.1, 0.2], atol=1e-9)
    assert TimestampEventBuffer.latest.closed is True
    assert fake_capture.released is True
    return f"Non-increasing/negative timestamps resolved to monotonic frame-time values {observed}."


class FakeYolo:
    def __init__(self, path):
        path_text = str(path).lower()
        if "door_v5" in path_text:
            self.names = {0: "student", 1: "paper", 2: "phone", 3: "note", 4: "door"}
        elif "late_note" in path_text:
            self.names = {0: "student", 1: "phone", 2: "paper", 3: "note", 4: "door"}
        else:
            self.names = {}


def test_weight_class_order_rejection(_work_root: Path):
    original_builder = MONITOR.DualCoreAntiCheatingSystem._build_tracker_config
    MONITOR.DualCoreAntiCheatingSystem._build_tracker_config = lambda _self: "synthetic.yaml"
    try:
        with patch.object(MONITOR, "YOLO", FakeYolo):
            try:
                MONITOR.DualCoreAntiCheatingSystem(
                    phone_model_path=RELEASE_ROOT / "weights" / "door_v5_phone_authority.pt",
                    note_context_model_path=RELEASE_ROOT / "weights" / "late_note_context.pt",
                )
            except RuntimeError as error:
                message = str(error)
            else:
                raise AssertionError("Mismatched class order did not abort startup")
    finally:
        MONITOR.DualCoreAntiCheatingSystem._build_tracker_config = original_builder
    assert "expected=" in message and "actual=" in message
    return "A deliberately reordered phone model class map raised RuntimeError before monitoring began."


def test_manifest_failure_retention(work_root: Path):
    launcher = load_module("frozen_launcher_fault_test", LAUNCHER_PATH)
    case_root = work_root / "manifest_failure"
    if case_root.exists():
        shutil.rmtree(case_root)
    code_dir = case_root / "code"
    config_dir = case_root / "config"
    weights_dir = case_root / "weights"
    runtime_root = case_root / "runtime_outputs"
    code_dir.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    weights_dir.mkdir(parents=True)
    runtime_root.mkdir(parents=True)

    config = yaml.safe_load(
        (RELEASE_ROOT / "config" / "hybrid_config.yaml").read_text(encoding="utf-8")
    )
    config_path = config_dir / "hybrid_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    app_path = code_dir / "hybrid_monitor.py"
    app_path.write_text("# synthetic import target\n", encoding="utf-8")
    for key in ("phone_weight", "note_context_weight", "pose_weight", "coco_weight"):
        path = case_root / config["model"][key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {key}".encode("utf-8"))

    launcher.CODE_DIR = code_dir
    launcher.RELEASE_ROOT = case_root
    launcher.CONFIG_PATH = config_path
    launcher.APP_PATH = app_path
    launcher.RUNTIME_OUTPUT_ROOT = runtime_root
    launcher.MANIFEST_DIR = runtime_root / "manifests"
    launcher.LATEST_MANIFEST_PATH = runtime_root / "run_manifest.json"
    launcher.package_versions = lambda: {"synthetic": "test"}
    launcher.device_details = lambda: {"cuda_available": False, "synthetic": True}

    class FailingMonitor:
        def __init__(self, **_kwargs):
            pass

        def run_forever(self, **_kwargs):
            raise RuntimeError("synthetic runtime failure")

    launcher.load_monitor_class = lambda: FailingMonitor
    old_argv = sys.argv[:]
    old_cwd = Path.cwd()
    sys.argv = ["run_hybrid_v2.py", "--source", "0", "--no-display", "--max-frames", "1"]
    try:
        try:
            launcher.main()
        except RuntimeError as error:
            assert str(error) == "synthetic runtime failure"
        else:
            raise AssertionError("Synthetic monitor failure was not propagated")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    latest = json.loads(launcher.LATEST_MANIFEST_PATH.read_text(encoding="utf-8"))
    unique_manifests = sorted(launcher.MANIFEST_DIR.glob("run_*.json"))
    assert latest["status"] == "failed"
    assert "synthetic runtime failure" in latest["exception"]
    assert latest.get("finished_at_utc")
    assert len(unique_manifests) == 1
    assert json.loads(unique_manifests[0].read_text(encoding="utf-8"))["status"] == "failed"
    return "Both latest and timestamped manifests retained failed status, exception, and finish time."


def test_event_buffer_pre_post_length(work_root: Path):
    RecordingWriter.instances.clear()
    output_dir = work_root / "event_buffer"
    with patch.object(MONITOR.cv2, "VideoWriter", RecordingWriter), patch.object(
        MONITOR.cv2,
        "VideoWriter_fourcc",
        return_value=0,
    ):
        buffer = MONITOR.EventVideoBuffer(
            output_dir,
            nominal_fps=10.0,
            pre_seconds=1.0,
            post_seconds=2.0,
            min_free_mb=0.0,
        )
        for frame_id in range(21):
            frame = np.full((4, 4, 3), frame_id, dtype=np.uint8)
            buffer.add_frame(frame, frame_id / 10.0)
        final_path = buffer.trigger("phone", 2.0, (4, 4, 3))
        assert final_path is not None
        for frame_id in range(21, 41):
            frame = np.full((4, 4, 3), frame_id, dtype=np.uint8)
            buffer.add_frame(frame, frame_id / 10.0)
        buffer.close()

    assert len(RecordingWriter.instances) == 1
    writer = RecordingWriter.instances[0]
    assert writer.released is True
    assert writer.frame_ids == list(range(10, 41))
    assert len(writer.frame_ids) == 31
    assert final_path.is_file()
    assert not list(output_dir.glob("*_writing.mp4"))
    return "At 10 FPS the clip retained frames 10-40: 1.0 s pre-trigger plus 2.0 s post-trigger."


TESTS = [
    ("FT-01", "Low-disk-space degradation", test_low_disk_degradation),
    ("FT-02", "Camera disconnect and recovery", test_camera_disconnect_and_recovery),
    ("FT-03", "Video timestamp fallback", test_timestamp_fallback),
    ("FT-04", "Weight class-order startup rejection", test_weight_class_order_rejection),
    ("FT-05", "Manifest failure-state retention", test_manifest_failure_retention),
    ("FT-06", "Event-buffer pre/post duration", test_event_buffer_pre_post_length),
]


def run_tests(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    transcript_lines = []
    suite_started = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory(prefix="frozen_6fe329_fault_") as temporary:
        work_root = Path(temporary)
        for test_id, name, function in TESTS:
            started = time.perf_counter()
            captured = StringIO()
            try:
                with redirect_stdout(captured):
                    detail = function(work_root)
                status = "PASS"
                exception = None
            except Exception:
                status = "FAIL"
                detail = "Test assertion or injected-path execution failed."
                exception = traceback.format_exc()
            duration = time.perf_counter() - started
            result = TestResult(test_id, name, status, duration, detail, exception)
            results.append(result)
            line = f"{test_id} {name}: {status} ({duration:.3f}s) - {detail}"
            print(line)
            transcript_lines.append(line)
            internal_output = captured.getvalue().strip()
            if internal_output:
                transcript_lines.append(f"  captured runtime output: {internal_output}")
            if exception:
                transcript_lines.append(exception.rstrip())

    suite_finished = datetime.now(timezone.utc)
    summary = {
        "suite": "Frozen 6FE329 deterministic fault-injection tests",
        "scope": "Synthetic inputs only; no participant video and no detector inference",
        "started_at_utc": suite_started.isoformat(),
        "finished_at_utc": suite_finished.isoformat(),
        "python": sys.version,
        "release_root": str(RELEASE_ROOT),
        "monitor_sha256": sha256(MONITOR_PATH),
        "launcher_sha256": sha256(LAUNCHER_PATH),
        "tests": [result.__dict__ for result in results],
        "passed": sum(result.status == "PASS" for result in results),
        "failed": sum(result.status == "FAIL" for result in results),
    }
    (output_dir / "fault_test_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    command = " ".join([sys.executable, *sys.argv])
    header = [
        "Frozen 6FE329 deterministic fault-injection test evidence",
        f"Executed: {suite_started.isoformat()}",
        f"Working directory: {Path.cwd()}",
        f"Command: {command}",
        f"Monitor SHA-256: {summary['monitor_sha256']}",
        f"Launcher SHA-256: {summary['launcher_sha256']}",
        "",
    ]
    (output_dir / "fault_test_output.txt").write_text(
        "\n".join(header + transcript_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def sha256(path: Path):
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_tests(args.output_dir.resolve())
    print(f"SUMMARY: {summary['passed']} passed, {summary['failed']} failed")
    raise SystemExit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
