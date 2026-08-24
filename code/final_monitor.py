import cv2
import math
import argparse
import time
import traceback
import threading
import shutil
import numpy as np
from ultralytics import YOLO
import os
import json
from collections import deque
from datetime import datetime
import ultralytics
from pathlib import Path

FINAL_CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FINAL_CANDIDATE_ROOT.parents[1]
FINAL_MODEL_PATH = (
    FINAL_CANDIDATE_ROOT
    / "weights"
    / "yolov10n_5class_phone_hardcases_v1_candidate.pt"
)
EXPECTED_CUSTOM_CLASSES = {
    0: "student",
    1: "phone",
    2: "paper",
    3: "note",
    4: "door",
}


class RecentFrameCapture:
    """Continuously capture into a small queue containing only recent frames."""

    def __init__(
        self,
        source,
        width=1280,
        height=720,
        reopen_interval=100,
        queue_size=3,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.reopen_interval = max(1, int(reopen_interval))
        self.queue_size = max(1, int(queue_size))
        self.condition = threading.Condition()
        self.frames = deque(maxlen=self.queue_size)
        self.sequence = 0
        self.stopped = False
        self.capture = self._open()
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
        self.thread = threading.Thread(
            target=self._reader_loop,
            name="latest-camera-frame",
            daemon=True,
        )
        self.thread.start()

    def _open(self):
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _reader_loop(self):
        failures = 0
        while not self.stopped:
            ok, frame = self.capture.read()
            if ok:
                failures = 0
                captured_at = time.monotonic()
                with self.condition:
                    self.sequence += 1
                    self.frames.append(
                        (self.sequence, frame, captured_at)
                    )
                    self.condition.notify_all()
                continue

            failures += 1
            if failures == 1:
                print("[CAMERA] No frame was read; the capture thread remains active and will retry automatically.")
            if failures % self.reopen_interval == 0 and not self.stopped:
                print("[CAMERA] Capture has not recovered; reconnecting to the camera...")
                self.capture.release()
                time.sleep(0.2)
                self.capture = self._open()
            else:
                time.sleep(0.02)

    def read_recent(self, last_sequence, timeout=0.25):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stopped
                or any(
                    sequence > last_sequence
                    for sequence, _, _ in self.frames
                ),
                timeout=timeout,
            )
            newer_frames = [
                item for item in self.frames if item[0] > last_sequence
            ]
            if not newer_frames:
                return False, None, None, last_sequence, 0
            # Inference deliberately consumes the newest captured frame. Older
            # unprocessed frames are discarded so latency cannot grow silently.
            sequence, frame, captured_at = newer_frames[-1]
            self.frames.clear()
            dropped = max(0, sequence - last_sequence - 1)
            return (
                True,
                frame.copy(),
                captured_at,
                sequence,
                dropped,
            )

    def stop(self):
        self.stopped = True
        with self.condition:
            self.condition.notify_all()
        self.capture.release()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


class EventVideoBuffer:
    """Keep recent annotated frames and write short clips around alert events."""

    def __init__(
        self,
        output_dir,
        nominal_fps=20.0,
        pre_seconds=1.0,
        post_seconds=2.0,
        min_free_mb=200.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nominal_fps = max(1.0, float(nominal_fps))
        self.pre_seconds = max(0.0, float(pre_seconds))
        self.post_seconds = max(0.0, float(post_seconds))
        self.min_free_bytes = int(float(min_free_mb) * 1024 * 1024)
        self.prebuffer = deque()
        self.active = {}
        self.clip_sequence = 0
        self.disabled_for_space = False

    @staticmethod
    def _safe_name(value):
        return "".join(character if character.isalnum() else "_" for character in str(value))

    def _estimated_fps(self):
        if len(self.prebuffer) >= 2:
            elapsed = self.prebuffer[-1][0] - self.prebuffer[0][0]
            if elapsed > 0:
                measured = (len(self.prebuffer) - 1) / elapsed
                return min(self.nominal_fps, max(2.0, measured))
        return self.nominal_fps

    def _has_disk_space(self):
        try:
            free_bytes = shutil.disk_usage(self.output_dir).free
        except OSError:
            return False
        if free_bytes < self.min_free_bytes:
            if not self.disabled_for_space:
                print(
                    "[EVENT VIDEO] archived text,archived text;"
                    "archived text、archived text."
                )
                self.disabled_for_space = True
            return False
        self.disabled_for_space = False
        return True

    def trigger(self, alert_key, event_time, frame_shape):
        if not self._has_disk_space():
            return None

        height, width = frame_shape[:2]
        self.clip_sequence += 1
        wall_time = datetime.now()
        stamp = wall_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        safe_key = self._safe_name(alert_key)
        final_path = self.output_dir / f"Event_{safe_key}_{stamp}_{self.clip_sequence:04d}.mp4"
        writing_path = self.output_dir / f"Event_{safe_key}_{stamp}_{self.clip_sequence:04d}_writing.mp4"
        fps = self._estimated_fps()
        writer = cv2.VideoWriter(
            str(writing_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            print("[EVENT VIDEO] Could not create the event video; screenshots, logs, and monitoring remain active.")
            return None

        for _, buffered_frame in self.prebuffer:
            if buffered_frame.shape[:2] == (height, width):
                writer.write(buffered_frame)

        clip_id = self.clip_sequence
        self.active[clip_id] = {
            "writer": writer,
            "writing_path": writing_path,
            "final_path": final_path,
            "end_time": event_time + self.post_seconds,
            "shape": (height, width),
        }
        return final_path

    def add_frame(self, frame, event_time):
        snapshot = frame.copy()
        self.prebuffer.append((event_time, snapshot))
        oldest_allowed = event_time - self.pre_seconds
        while self.prebuffer and self.prebuffer[0][0] < oldest_allowed:
            self.prebuffer.popleft()

        for clip_id, clip in list(self.active.items()):
            if snapshot.shape[:2] == clip["shape"]:
                clip["writer"].write(snapshot)
            if event_time >= clip["end_time"]:
                self._finish_clip(clip_id)

    def _finish_clip(self, clip_id):
        clip = self.active.pop(clip_id, None)
        if clip is None:
            return
        clip["writer"].release()
        try:
            os.replace(clip["writing_path"], clip["final_path"])
            print(f"[EVENT VIDEO SAVED] {clip['final_path'].name}")
        except OSError as error:
            print(f"[EVENT VIDEO] The video was closed, but renaming failed: {error}")

    def close(self):
        for clip_id in list(self.active):
            self._finish_clip(clip_id)
        self.prebuffer.clear()


class DualCoreAntiCheatingSystem:
    def __init__(self, custom_model_path=None):
        print("Starting the dual-core monitoring system: YOLOv8-Pose for people plus a custom object model...")
        
        # ==============================================================
        # ==============================================================
        self.tracker_path = self._build_tracker_config()
        
        print("Loading core model 1: YOLOv8n-Pose for person tracking and full-frame keypoint extraction...")
        self.yolo_pose = YOLO("yolov8n-pose.pt")

        print("Loading auxiliary phone model: YOLOv8n COCO (cell phone only)...")
        self.yolo_coco = YOLO("yolov8n.pt")
        
        print("Loading core model 2: the custom prohibited-item detector...")
        if custom_model_path is None:
            custom_model_path = FINAL_MODEL_PATH
        self.yolo_custom = YOLO(str(custom_model_path))
        # Only a phone and a temporally confirmed small note are forbidden.
        # Ordinary exam paper is contextual counter-evidence, like a door.
        self.custom_class_names = {1: 'phone', 3: 'note'}
        model_names = self.yolo_custom.names
        if isinstance(model_names, dict):
            normalized_model_names = {
                int(class_id): str(name).strip().lower()
                for class_id, name in model_names.items()
            }
        else:
            normalized_model_names = {
                class_id: str(name).strip().lower()
                for class_id, name in enumerate(model_names)
            }
        if normalized_model_names != EXPECTED_CUSTOM_CLASSES:
            raise RuntimeError(
                "archived text class mismatch:"
                f"expected={EXPECTED_CUSTOM_CLASSES}, actual={normalized_model_names}"
            )
        model_name_items = normalized_model_names.items()
        self.DOOR_CLASS_ID = next(
            (int(class_id) for class_id, name in model_name_items if str(name).lower() == "door"),
            None,
        )
        # Door is contextual counter-evidence, never a forbidden object. It is
        # shown neutrally but never suppresses a real phone held in front of it.
        self.DOOR_NEUTRAL_CONFIDENCE = float(
            os.environ.get("DOOR_NEUTRAL_CONFIDENCE", "0.65")
        )
        self.CUSTOM_OBJECT_CONFIDENCE = float(
            os.environ.get("CUSTOM_OBJECT_CONFIDENCE", "0.50")
        )
        self.PAPER_CLASS_ID = 2
        self.NOTE_CLASS_ID = 3
        self.PAPER_NEUTRAL_CONFIDENCE = float(
            os.environ.get("PAPER_NEUTRAL_CONFIDENCE", "0.55")
        )
        self.PAPER_COUNTER_CONFIDENCE = float(
            os.environ.get("PAPER_COUNTER_CONFIDENCE", "0.20")
        )
        # Notes are often much smaller than phones. Run a dedicated high-
        # resolution note pass at a lower proposal threshold, then require
        # person/hand association and persistence before creating an alert.
        self.NOTE_DETECTION_CONFIDENCE = float(
            os.environ.get("NOTE_DETECTION_CONFIDENCE", "0.30")
        )
        self.NOTE_INFERENCE_SIZE = int(
            os.environ.get("NOTE_INFERENCE_SIZE", "960")
        )
        self.NOTE_CONFIRM_SECONDS = float(
            os.environ.get("NOTE_CONFIRM_SECONDS", "0.15")
        )
        self.NOTE_TRACK_GAP_SECONDS = float(
            os.environ.get("NOTE_TRACK_GAP_SECONDS", "0.35")
        )
        self.NOTE_TRANSFER_WINDOW_SECONDS = float(
            os.environ.get("NOTE_TRANSFER_WINDOW_SECONDS", "1.20")
        )
        self.NOTE_MIN_HAND_OBSERVATIONS = int(
            os.environ.get("NOTE_MIN_HAND_OBSERVATIONS", "2")
        )
        self.NOTE_HAND_DISTANCE_FACTOR = float(
            os.environ.get("NOTE_HAND_DISTANCE_FACTOR", "0.80")
        )
        self.NOTE_MAX_PERSON_AREA_RATIO = float(
            os.environ.get("NOTE_MAX_PERSON_AREA_RATIO", "0.12")
        )
        self.NOTE_MAX_SHOULDER_WIDTH_RATIO = float(
            os.environ.get("NOTE_MAX_SHOULDER_WIDTH_RATIO", "0.55")
        )
        self.NOTE_MAX_SHOULDER_HEIGHT_RATIO = float(
            os.environ.get("NOTE_MAX_SHOULDER_HEIGHT_RATIO", "0.65")
        )
        self.note_tracks = {}
        self.next_note_track_id = 1

        self.POSE_CONFIDENCE = float(
            os.environ.get("POSE_CONFIDENCE", "0.45")
        )
        self.POSE_TRACK_CONFIDENCE = float(
            os.environ.get("POSE_TRACK_CONFIDENCE", "0.40")
        )
        self.MAX_STUDENTS = int(os.environ.get("MAX_STUDENTS", "30"))
        self.COCO_PHONE_CLASS = 67
        self.COCO_PHONE_CONFIDENCE = float(
            os.environ.get("COCO_PHONE_CONFIDENCE", "0.15")
        )
        self.COCO_PHONE_INFERENCE_SIZE = int(
            os.environ.get("COCO_PHONE_INFERENCE_SIZE", "1280")
        )
        self.COCO_CONFIRM_SECONDS = float(
            os.environ.get("COCO_CONFIRM_SECONDS", "0.20")
        )
        self.COCO_MIN_OBSERVATIONS = int(
            os.environ.get("COCO_MIN_OBSERVATIONS", "3")
        )
        self.COCO_MAX_GAP_SECONDS = float(
            os.environ.get("COCO_MAX_GAP_SECONDS", "0.45")
        )
        self.COCO_DISTRACTOR_CONFIDENCE = float(
            os.environ.get("COCO_DISTRACTOR_CONFIDENCE", "0.55")
        )
        self.COCO_DISTRACTOR_IOU = float(
            os.environ.get("COCO_DISTRACTOR_IOU", "0.25")
        )
        # A phone held by an extended arm can lie just outside a tight pose
        # box.  Use a resolution-scaled allowance only for the temporally
        # confirmed COCO fallback; the authoritative custom path is unchanged.
        self.COCO_PERSON_MARGIN_RATIO = float(
            os.environ.get("COCO_PERSON_MARGIN_RATIO", "0.13")
        )
        self.COCO_PERSON_MARGIN_MIN = int(
            os.environ.get("COCO_PERSON_MARGIN_MIN", "80")
        )
        self.COCO_MIN_ASPECT_RATIO = float(
            os.environ.get("COCO_MIN_ASPECT_RATIO", "1.12")
        )
        self.COCO_STRONG_CONFIDENCE = float(
            os.environ.get("COCO_STRONG_CONFIDENCE", "0.20")
        )
        self.COCO_ROTATED_SUPPORT_ANGLE = float(
            os.environ.get("COCO_ROTATED_SUPPORT_ANGLE", "30")
        )
        self.COCO_ROTATED_SUPPORT_CONFIDENCE = float(
            os.environ.get("COCO_ROTATED_SUPPORT_CONFIDENCE", "0.20")
        )
        self.COCO_ROTATED_SUPPORT_IOU = float(
            os.environ.get("COCO_ROTATED_SUPPORT_IOU", "0.40")
        )
        self.COCO_ASSIST_MODE = os.environ.get(
            "COCO_ASSIST_MODE", "counter_evidence"
        ).strip().lower()
        # Phone boxes are expected to lie on the tracked student's body/arms.
        # A small allowance preserves hand-held phones without admitting nearby chairs.
        self.PHONE_PERSON_MARGIN = int(
            os.environ.get("PHONE_PERSON_MARGIN", "30")
        )
        self.PHONE_MAX_FRAME_AREA_RATIO = float(
            os.environ.get("PHONE_MAX_FRAME_AREA_RATIO", "0.15")
        )
        self.coco_phone_candidate_box = None
        self.coco_phone_candidate_first_seen = None
        self.coco_phone_candidate_last_seen = None
        self.coco_phone_candidate_observations = 0
        
        self.FPS_ESTIMATE = 30
        self.TIME_THRESHOLD_SECONDS = float(
            os.environ.get("CROSSING_TRIGGER_SECONDS", "0.50")
        )
        self.STATE_MAX_STEP_SECONDS = float(
            os.environ.get("STATE_MAX_STEP_SECONDS", "0.35")
        )
        self.crossing_states = {}
        self.track_states = {}
        self.frame_index = 0
        self.OCCLUSION_GRACE_SECONDS = float(
            os.environ.get("OCCLUSION_GRACE_SECONDS", "1.50")
        )
        # A live invigilation session must only stop on an explicit user action.
        # Camera read failures are retried indefinitely instead of terminating.
        self.CAMERA_REOPEN_INTERVAL = int(
            os.environ.get("CAMERA_REOPEN_INTERVAL", "100")
        )
        self.CAMERA_FRAME_QUEUE_SIZE = max(
            1,
            int(os.environ.get("CAMERA_FRAME_QUEUE_SIZE", "3")),
        )

        self.evidence_dir = PROJECT_ROOT / "Evidence_Vault"
        os.makedirs(self.evidence_dir, exist_ok=True)  
        self.last_capture_time = {}                    
        self.CAPTURE_COOLDOWN = float(
            os.environ.get("CAPTURE_COOLDOWN_SECONDS", "5.0")
        )
        self.manual_exit_requested = False
        self.EVENT_PRE_SECONDS = float(
            os.environ.get("EVENT_PRE_SECONDS", "1.0")
        )
        self.EVENT_POST_SECONDS = float(
            os.environ.get("EVENT_POST_SECONDS", "2.0")
        )
        self.EVENT_MIN_FREE_MB = float(
            os.environ.get("EVENT_MIN_FREE_MB", "200")
        )
        self.event_video_buffer = None
        self._active_capture = None
        self._active_writer = None
        self._active_raw_writer = None
        self.session_event_origin = None

    def _log_runtime_error(self, error):
        """Persist unexpected runtime errors so a closed window is diagnosable."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.evidence_dir / "runtime_errors.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n[{timestamp}] {type(error).__name__}: {error}\n")
                log_file.write(traceback.format_exc())
        except OSError:
            # Monitoring must stay alive even if the evidence drive is unavailable.
            pass

    def _build_tracker_config(self):
        """archived text,archived text"""
        default_yaml_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "botsort.yaml"
        custom_yaml_path = PROJECT_ROOT / "auto_exam_tracker.yaml"
        
        if default_yaml_path.exists():
            with open(default_yaml_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            has_fuse_score = False
            with open(custom_yaml_path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line)
                    has_fuse_score = has_fuse_score or line.strip().startswith("fuse_score:")
                if not has_fuse_score:
                    f.write("\nfuse_score: True\n")
        else:
            yaml_content = (
                "tracker_type: botsort\n"
                "track_high_thresh: 0.25\n"
                "track_low_thresh: 0.1\n"
                "new_track_thresh: 0.25\n"
                "track_buffer: 30\n"
                "match_thresh: 0.8\n"
                "gmc_method: sparseOptFlow\n"
                "proximity_thresh: 0.5\n"
                "appearance_thresh: 0.25\n"
                "with_reid: False\n"
                "fuse_score: True\n"
            )
            with open(custom_yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
                
        print(f"[*] Patched anti-frame-loss tracker configuration created and mounted: {custom_yaml_path}")
        return str(custom_yaml_path)

    @staticmethod
    def calculate_distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def calculate_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        intersection_w = max(0, min(ax2, bx2) - max(ax1, bx1))
        intersection_h = max(0, min(ay2, by2) - max(by1, ay1))
        intersection = intersection_w * intersection_h
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def calculate_containment(inner_box, outer_box):
        """Return the fraction of ``inner_box`` covered by ``outer_box``."""
        ix1, iy1, ix2, iy2 = inner_box
        ox1, oy1, ox2, oy2 = outer_box
        intersection_w = max(0, min(ix2, ox2) - max(ix1, ox1))
        intersection_h = max(0, min(iy2, oy2) - max(iy1, oy1))
        intersection = intersection_w * intersection_h
        inner_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        return intersection / inner_area if inner_area > 0 else 0.0

    def filter_false_positives(self, frame_width, frame_height, item_boxes, item_classes, person_boxes):
        """Initial object filter: reject only oversized phones and background objects."""
        valid_boxes = []
        valid_classes = []

        for box, cls in zip(item_boxes, item_classes):
            ix1, iy1, ix2, iy2 = map(int, box)
            cls_id = int(cls)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            icx, icy = (ix1 + ix2) / 2, (iy1 + iy2) / 2

            if (
                cls_id == 1
                and item_area
                > frame_width * frame_height * self.PHONE_MAX_FRAME_AREA_RATIO
            ):
                continue

            is_near_person = False
            for p_box in person_boxes:
                px1, py1, px2, py2 = p_box
                margin = self.PHONE_PERSON_MARGIN if cls_id == 1 else 80
                if (px1 - margin) < icx < (px2 + margin) and (py1 - margin) < icy < (py2 + margin):
                    is_near_person = True
                    break

            if not is_near_person:
                continue

            valid_boxes.append(box)
            valid_classes.append(cls)

        return valid_boxes, valid_classes

    def filter_coco_phone_candidates(
        self,
        frame_width,
        frame_height,
        item_boxes,
        item_confidences,
        person_boxes,
    ):
        """Associate weak COCO phones with a person using an arm-aware margin."""
        valid_boxes = []
        valid_confidences = []
        margin = max(
            self.COCO_PERSON_MARGIN_MIN,
            int(min(frame_width, frame_height) * self.COCO_PERSON_MARGIN_RATIO),
        )
        for box, confidence in zip(item_boxes, item_confidences):
            ix1, iy1, ix2, iy2 = map(int, box)
            item_width = max(0, ix2 - ix1)
            item_height = max(0, iy2 - iy1)
            item_area = item_width * item_height
            if item_area > frame_width * frame_height * 0.15:
                continue
            aspect_ratio = max(item_width, item_height) / max(
                1, min(item_width, item_height)
            )
            if aspect_ratio < self.COCO_MIN_ASPECT_RATIO:
                continue
            center_x = (ix1 + ix2) / 2
            center_y = (iy1 + iy2) / 2
            if any(
                (px1 - margin) < center_x < (px2 + margin)
                and (py1 - margin) < center_y < (py2 + margin)
                for px1, py1, px2, py2 in person_boxes
            ):
                valid_boxes.append(box)
                valid_confidences.append(float(confidence))
        return valid_boxes, valid_confidences

    @staticmethod
    def restore_rotated_box(box, inverse_matrix):
        """Map an axis-aligned box from a rotated frame back to the source."""
        x1, y1, x2, y2 = box
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
            dtype=np.float32,
        )
        restored = cv2.transform(corners, inverse_matrix)[0]
        return np.array(
            [
                restored[:, 0].min(),
                restored[:, 1].min(),
                restored[:, 0].max(),
                restored[:, 1].max(),
            ],
            dtype=np.float32,
        )

    def has_rotated_custom_phone_support(self, frame, coco_box):
        """Require custom-model agreement for a weak COCO proposal."""
        frame_height, frame_width = frame.shape[:2]
        center = (frame_width / 2.0, frame_height / 2.0)
        matrix = cv2.getRotationMatrix2D(
            center,
            self.COCO_ROTATED_SUPPORT_ANGLE,
            1.0,
        )
        inverse_matrix = cv2.invertAffineTransform(matrix)
        rotated = cv2.warpAffine(
            frame,
            matrix,
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        results = self.yolo_custom.predict(
            rotated,
            classes=[1],
            conf=self.COCO_ROTATED_SUPPORT_CONFIDENCE,
            imgsz=640,
            verbose=False,
        )
        if len(results[0].boxes) == 0:
            return False
        for rotated_box in results[0].boxes.xyxy.cpu().numpy():
            restored_box = self.restore_rotated_box(
                rotated_box,
                inverse_matrix,
            )
            if (
                self.calculate_iou(coco_box, restored_box)
                >= self.COCO_ROTATED_SUPPORT_IOU
            ):
                return True
        return False

    def update_coco_phone_confirmation(
        self,
        coco_boxes,
        event_time,
        force_reset=False,
    ):
        """Promote a COCO candidate by elapsed source time, not processed frames."""
        if force_reset:
            self.coco_phone_candidate_box = None
            self.coco_phone_candidate_first_seen = None
            self.coco_phone_candidate_last_seen = None
            self.coco_phone_candidate_observations = 0
            return None
        if not coco_boxes:
            if (
                self.coco_phone_candidate_last_seen is None
                or event_time - self.coco_phone_candidate_last_seen
                > self.COCO_MAX_GAP_SECONDS
            ):
                self.coco_phone_candidate_box = None
                self.coco_phone_candidate_first_seen = None
                self.coco_phone_candidate_last_seen = None
                self.coco_phone_candidate_observations = 0
            return None

        candidate = coco_boxes[0]
        if (
            self.coco_phone_candidate_box is not None
            and self.coco_phone_candidate_last_seen is not None
            and event_time - self.coco_phone_candidate_last_seen
            <= self.COCO_MAX_GAP_SECONDS
            and self.calculate_iou(candidate, self.coco_phone_candidate_box) >= 0.35
        ):
            self.coco_phone_candidate_observations += 1
        else:
            self.coco_phone_candidate_box = candidate
            self.coco_phone_candidate_first_seen = event_time
            self.coco_phone_candidate_observations = 1

        self.coco_phone_candidate_box = candidate
        self.coco_phone_candidate_last_seen = event_time
        elapsed = event_time - self.coco_phone_candidate_first_seen
        if (
            self.coco_phone_candidate_observations >= self.COCO_MIN_OBSERVATIONS
            and elapsed >= self.COCO_CONFIRM_SECONDS
        ):
            return candidate
        return None

    @staticmethod
    def draw_coco_assisted_phone(frame, box):
        """Draw a visually distinct alert so COCO-assisted detections are testable."""
        x1, y1, x2, y2 = map(int, box)
        color = (0, 215, 255)  # yellow: auxiliary result, not the custom model
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame, "COCO ASSIST: PHONE", (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
        )

    def draw_forbidden_objects(self, frame, boxes, classes):
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = int(cls)
            
            if cls_id in self.custom_class_names:
                obj_name = self.custom_class_names[cls_id]
                if cls_id == 1:   color = (0, 0, 255)   
                elif cls_id == 2: color = (255, 0, 0)   
                elif cls_id == 3: color = (0, 128, 255) 
                
                label = f"WARNING: {obj_name.upper()}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def draw_neutral_doors(self, frame, boxes, confidences):
        """Show recognized doors as context without creating an alert."""
        color = (150, 150, 150)
        for box, confidence in zip(boxes, confidences):
            if float(confidence) < self.DOOR_NEUTRAL_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"DOOR (context) {float(confidence):.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    def draw_neutral_papers(self, frame, boxes, confidences):
        """Show ordinary exam paper as neutral context, never as evidence."""
        color = (180, 180, 180)
        for box, confidence in zip(boxes, confidences):
            if float(confidence) < self.PAPER_NEUTRAL_CONFIDENCE:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"PAPER (context) {float(confidence):.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )

    @staticmethod
    def _box_center_and_size(box):
        x1, y1, x2, y2 = map(float, box)
        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            max(1.0, x2 - x1),
            max(1.0, y2 - y1),
        )

    def _associate_note_with_hand(self, box, students):
        """Return the most plausible holder using scale relative to each person."""
        center_x, center_y, note_w, note_h = self._box_center_and_size(box)
        best = None
        for student_id, data in students.items():
            px1, py1, px2, py2 = map(float, data["box"])
            person_w = max(1.0, px2 - px1)
            person_h = max(1.0, py2 - py1)
            person_area = person_w * person_h
            note_area_ratio = (note_w * note_h) / person_area
            if note_area_ratio > self.NOTE_MAX_PERSON_AREA_RATIO:
                continue

            shoulder_width = max(25.0, float(data["shoulder_width"]))
            if (
                note_w / shoulder_width > self.NOTE_MAX_SHOULDER_WIDTH_RATIO
                or note_h / shoulder_width > self.NOTE_MAX_SHOULDER_HEIGHT_RATIO
            ):
                continue
            margin = shoulder_width * 0.55
            if not (
                px1 - margin <= center_x <= px2 + margin
                and py1 - margin <= center_y <= py2 + margin
            ):
                continue

            wrists = data.get("wrists") or []
            if not wrists:
                continue
            hand_distance = min(
                self.calculate_distance((center_x, center_y), wrist)
                for wrist in wrists
            )
            normalized_distance = hand_distance / shoulder_width
            if normalized_distance > self.NOTE_HAND_DISTANCE_FACTOR:
                continue
            if best is None or normalized_distance < best[1]:
                best = (student_id, normalized_distance)
        return best

    def update_note_candidates(
        self,
        note_boxes,
        note_confidences,
        paper_boxes,
        paper_confidences,
        students,
        event_time,
    ):
        """Track note proposals using source-time windows instead of frame counts."""
        candidates = []
        for box, confidence in zip(note_boxes, note_confidences):
            confidence = float(confidence)
            _, _, note_w, note_h = self._box_center_and_size(box)
            if min(note_w, note_h) < 6:
                continue

            # When the detector describes essentially the same region as
            # ordinary paper with at least comparable confidence, keep it
            # neutral. A small note lying on a large exam sheet has low IoU and
            # is therefore not automatically suppressed.
            contradicted_by_paper = any(
                float(paper_confidence) >= self.PAPER_COUNTER_CONFIDENCE
                and (
                    (
                        float(paper_confidence) >= confidence - 0.15
                        and self.calculate_iou(box, paper_box) >= 0.35
                    )
                    or (
                        float(paper_confidence) >= confidence
                        and self.calculate_containment(box, paper_box) >= 0.75
                    )
                )
                for paper_box, paper_confidence in zip(
                    paper_boxes, paper_confidences
                )
            )
            if contradicted_by_paper:
                continue

            holder = self._associate_note_with_hand(box, students)
            candidates.append(
                {
                    "box": [float(value) for value in box],
                    "confidence": confidence,
                    "holder": None if holder is None else holder[0],
                }
            )

        for track_id in list(self.note_tracks):
            if (
                event_time - self.note_tracks[track_id]["last_seen"]
                > self.NOTE_TRACK_GAP_SECONDS
            ):
                del self.note_tracks[track_id]

        unmatched_tracks = set(self.note_tracks)
        for candidate in candidates:
            center_x, center_y, note_w, note_h = self._box_center_and_size(
                candidate["box"]
            )
            best_track_id = None
            best_score = -1.0
            for track_id in unmatched_tracks:
                track_box = self.note_tracks[track_id]["box"]
                old_x, old_y, old_w, old_h = self._box_center_and_size(track_box)
                iou = self.calculate_iou(candidate["box"], track_box)
                center_distance = self.calculate_distance(
                    (center_x, center_y), (old_x, old_y)
                )
                match_radius = max(
                    30.0,
                    2.5 * max(note_w, note_h, old_w, old_h),
                )
                if iou < 0.12 and center_distance > match_radius:
                    continue
                score = iou + max(0.0, 1.0 - center_distance / match_radius)
                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is None:
                best_track_id = self.next_note_track_id
                self.next_note_track_id += 1
                self.note_tracks[best_track_id] = {
                    "box": candidate["box"],
                    "first_seen": event_time,
                    "last_seen": event_time,
                    "observations": [],
                }
            else:
                unmatched_tracks.remove(best_track_id)

            track = self.note_tracks[best_track_id]
            track["box"] = candidate["box"]
            track["last_seen"] = event_time
            track["observations"].append((event_time, candidate["holder"]))
            window_start = event_time - self.NOTE_TRANSFER_WINDOW_SECONDS
            track["observations"] = [
                observation
                for observation in track["observations"]
                if observation[0] >= window_start
            ]
            candidate["track_id"] = best_track_id
            hand_observations = sum(
                holder is not None for _, holder in track["observations"]
            )
            candidate["confirmed"] = (
                event_time - track["first_seen"] >= self.NOTE_CONFIRM_SECONDS
                and hand_observations >= self.NOTE_MIN_HAND_OBSERVATIONS
                and candidate["holder"] is not None
            )
            holder_counts = {}
            for _, holder in track["observations"]:
                if holder is not None:
                    holder_counts[holder] = holder_counts.get(holder, 0) + 1
            candidate["transfer"] = sum(
                count >= 2 for count in holder_counts.values()
            ) >= 2

        for track_id in list(self.note_tracks):
            if (
                event_time - self.note_tracks[track_id]["last_seen"]
                > self.NOTE_TRACK_GAP_SECONDS
            ):
                del self.note_tracks[track_id]

        return candidates

    @staticmethod
    def draw_note_candidates(frame, candidates):
        """Distinguish proposals from confirmed, hand-associated notes."""
        confirmed_boxes = []
        for candidate in candidates:
            x1, y1, x2, y2 = map(int, candidate["box"])
            if candidate["confirmed"] and candidate["transfer"]:
                color = (0, 0, 255)
                label = "WARNING: NOTE TRANSFER"
                confirmed_boxes.append(candidate["box"])
                thickness = 3
            elif candidate["confirmed"]:
                color = (0, 128, 255)
                label = "NOTE HELD (review)"
                thickness = 2
            else:
                color = (0, 200, 255)
                label = "NOTE CANDIDATE"
                thickness = 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                f"{label} {candidate['confidence']:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return confirmed_boxes

    def _trigger_event_video(self, alert_key, event_time, frame):
        if self.event_video_buffer is None:
            return None
        return self.event_video_buffer.trigger(alert_key, event_time, frame.shape)

    def _source_time_seconds(self, event_time):
        if self.session_event_origin is None:
            return float(event_time)
        return float(event_time - self.session_event_origin)

    @staticmethod
    def _related_track_ids(box, students):
        """Return pose-track IDs whose expanded body box contains the object centre."""
        if box is None:
            return []
        x1, y1, x2, y2 = map(float, box)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        matches = []
        for track_id, data in students.items():
            px1, py1, px2, py2 = map(float, data["box"])
            margin = max(30.0, 0.12 * max(px2 - px1, py2 - py1))
            if (
                px1 - margin <= center_x <= px2 + margin
                and py1 - margin <= center_y <= py2 + margin
            ):
                matches.append(int(track_id))
        return sorted(matches)

    def check_and_log_objects(
        self,
        frame,
        classes,
        event_time,
        detections=None,
    ):
        current_time = datetime.now()
        detected_custom_classes = set(int(cls) for cls in classes if int(cls) in self.custom_class_names)
        detections = detections or []
        
        for cls_id in detected_custom_classes:
            cooldown_key = f"object_{cls_id}"
            last_time = self.last_capture_time.get(cooldown_key)
            
            if last_time is None or event_time - last_time > self.CAPTURE_COOLDOWN:
                obj_name = self.custom_class_names[cls_id]
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                img_filename = f"Violation_Object_{obj_name}_{timestamp_str}.jpg"
                img_path = os.path.join(self.evidence_dir, img_filename)
                
                cv2.imwrite(img_path, frame)
                clip_path = self._trigger_event_video(
                    f"object_{obj_name}",
                    event_time,
                    frame,
                )
                
                log_data = {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_time_seconds": round(
                        self._source_time_seconds(event_time),
                        3,
                    ),
                    "alert_type": f"Forbidden Object Detected: {obj_name.upper()}",
                    "image_path": img_path,
                    "event_clip_path": None if clip_path is None else str(clip_path),
                    "detections": [
                        detection
                        for detection in detections
                        if int(detection.get("class_id", -1)) == cls_id
                    ],
                }
                day_str = current_time.strftime("%Y%m%d")
                log_file = os.path.join(self.evidence_dir, f"log_{day_str}.jsonl")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    
                self.last_capture_time[cooldown_key] = event_time
                print(f"[EVIDENCE SAVED] Prohibited-item alert evidence saved: {img_filename}")

    def update_scene_context(self, students, event_time):
        """Infer broad depth layers and short-term occlusion state from current tracks."""
        if students:
            ordered = sorted(students.items(), key=lambda item: item[1]['anchor'][1])
            median_height = sorted(data['box_h'] for _, data in ordered)[len(ordered) // 2]
            layer_threshold = max(45, median_height * 0.35)
            layers = []
            for student_id, data in ordered:
                if not layers or abs(data['anchor'][1] - layers[-1]['center_y']) > layer_threshold:
                    layers.append({'center_y': data['anchor'][1], 'members': [student_id]})
                else:
                    layer = layers[-1]
                    layer['members'].append(student_id)
                    layer['center_y'] = sum(students[member]['anchor'][1] for member in layer['members']) / len(layer['members'])
            for layer_index, layer in enumerate(layers, start=1):
                for student_id in layer['members']:
                    students[student_id]['depth_layer'] = layer_index

            for student_id, data in students.items():
                overlap = max(
                    (self.calculate_iou(data['box'], other['box']) for other_id, other in students.items() if other_id != student_id),
                    default=0.0,
                )
                # Keypoint count varies greatly with framing (for example, a laptop camera often cannot see legs).
                # Treat a student as partially occluded only when their person box substantially overlaps another track.
                partial = overlap > 0.35
                data['visibility'] = 'partial' if partial else 'visible'
                self.track_states[student_id] = {
                    'last_seen_time': event_time,
                    'anchor': data['anchor'],
                    'depth_layer': data['depth_layer'],
                    'visibility': data['visibility'],
                }

        for student_id, state in list(self.track_states.items()):
            unseen_seconds = event_time - state['last_seen_time']
            if student_id not in students and unseen_seconds <= self.OCCLUSION_GRACE_SECONDS:
                state['visibility'] = 'occluded'
            elif unseen_seconds > self.OCCLUSION_GRACE_SECONDS:
                del self.track_states[student_id]

    def crossing_candidate(self, source, target):
        """Find a hand that crosses from its own body space toward another stable track."""
        if source['visibility'] != 'visible' or target['visibility'] != 'visible':
            return None
        own_anchor, target_anchor = source['anchor'], target['anchor']
        axis_x, axis_y = target_anchor[0] - own_anchor[0], target_anchor[1] - own_anchor[1]
        anchor_distance = math.hypot(axis_x, axis_y)
        if anchor_distance < max(source['shoulder_width'], target['shoulder_width']) * 1.2:
            return None

        best_wrist = None
        for wrist in source['wrists']:
            wrist_x, wrist_y = wrist[0] - own_anchor[0], wrist[1] - own_anchor[1]
            projection = (wrist_x * axis_x + wrist_y * axis_y) / anchor_distance
            perpendicular = abs(wrist_x * axis_y - wrist_y * axis_x) / anchor_distance
            if projection > anchor_distance * 0.55 and perpendicular < max(source['shoulder_width'], target['shoulder_width']) * 1.2:
                best_wrist = wrist
                break
        return best_wrist

    def analyze_interactions(self, frame, students, event_time):
        """Detect persistent reaching using elapsed source time."""
        candidate_pairs = {}
        student_ids = list(students)
        for source_id in student_ids:
            for target_id in student_ids:
                if source_id == target_id:
                    continue
                wrist = self.crossing_candidate(students[source_id], students[target_id])
                if wrist is None:
                    continue
                pair_key = tuple(sorted((source_id, target_id)))
                candidate_pairs[pair_key] = (wrist, students[target_id]['anchor'])

        all_pairs = set(self.crossing_states) | set(candidate_pairs)
        for pair_key in all_pairs:
            state = self.crossing_states.setdefault(
                pair_key,
                {
                    "active_seconds": 0.0,
                    "last_update": event_time,
                    "was_candidate": False,
                    "alerted": False,
                },
            )
            elapsed = max(0.0, event_time - state["last_update"])
            elapsed = min(elapsed, self.STATE_MAX_STEP_SECONDS)
            is_candidate = pair_key in candidate_pairs
            if is_candidate and state["was_candidate"]:
                state["active_seconds"] = min(
                    self.TIME_THRESHOLD_SECONDS,
                    state["active_seconds"] + elapsed,
                )
            elif not is_candidate:
                state["active_seconds"] = max(
                    0.0,
                    state["active_seconds"] - elapsed,
                )
            state["last_update"] = event_time
            state["was_candidate"] = is_candidate
            duration = state["active_seconds"]

            if is_candidate:
                wrist, target_anchor = candidate_pairs[pair_key]
                color = (
                    (0, 255, 255)
                    if duration < self.TIME_THRESHOLD_SECONDS
                    else (0, 0, 255)
                )
                thickness = 2 if duration < self.TIME_THRESHOLD_SECONDS else 4
                cv2.line(
                    frame,
                    wrist,
                    tuple(map(int, target_anchor)),
                    color,
                    thickness,
                )
                label = (
                    f"Cross-track reach: {pair_key[0]} & {pair_key[1]} "
                    f"({duration:.2f}/{self.TIME_THRESHOLD_SECONDS:.2f}s)"
                )
                cv2.putText(
                    frame,
                    label,
                    (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                )

            if (
                is_candidate
                and duration >= self.TIME_THRESHOLD_SECONDS
                and not state["alerted"]
            ):
                current_time = datetime.now()
                last_time = self.last_capture_time.get(pair_key)
                if last_time is None or event_time - last_time > self.CAPTURE_COOLDOWN:
                    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    img_filename = f"CrossTrack_{pair_key[0]}_{pair_key[1]}_{timestamp_str}.jpg"
                    img_path = os.path.join(self.evidence_dir, img_filename)
                    cv2.imwrite(img_path, frame)
                    clip_path = self._trigger_event_video(
                        f"cross_track_{pair_key[0]}_{pair_key[1]}",
                        event_time,
                        frame,
                    )
                    log_data = {
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "source_time_seconds": round(
                            self._source_time_seconds(event_time),
                            3,
                        ),
                        "students": list(pair_key),
                        "alert_type": "Persistent cross-track reach",
                        "image_path": str(img_path),
                        "event_clip_path": None if clip_path is None else str(clip_path),
                    }
                    log_file = os.path.join(self.evidence_dir, f"log_{current_time.strftime('%Y%m%d')}.jsonl")
                    with open(log_file, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    self.last_capture_time[pair_key] = event_time
                state["alerted"] = True
            elif duration <= self.TIME_THRESHOLD_SECONDS * 0.20:
                state["alerted"] = False

            if not is_candidate and duration <= 0.0:
                del self.crossing_states[pair_key]

    def _reset_runtime_state(self):
        self.note_tracks.clear()
        self.next_note_track_id = 1
        self.crossing_states.clear()
        self.track_states.clear()
        self.frame_index = 0
        self.last_capture_time.clear()
        self.coco_phone_candidate_box = None
        self.coco_phone_candidate_first_seen = None
        self.coco_phone_candidate_last_seen = None
        self.coco_phone_candidate_observations = 0
        self.session_event_origin = None

    def _release_runtime_resources(self):
        if self.event_video_buffer is not None:
            self.event_video_buffer.close()
            self.event_video_buffer = None
        if self._active_writer is not None:
            self._active_writer.release()
            self._active_writer = None
        if self._active_raw_writer is not None:
            self._active_raw_writer.release()
            self._active_raw_writer = None
        if self._active_capture is not None:
            if isinstance(self._active_capture, RecentFrameCapture):
                self._active_capture.stop()
            else:
                self._active_capture.release()
            self._active_capture = None
        cv2.destroyAllWindows()

    def run_live(
        self,
        source=0,
        save_output=None,
        save_raw=None,
        display=True,
        exit_on_eof=False,
        max_frames=None,
    ):
        self._reset_runtime_state()
        is_live_camera = isinstance(source, int)
        if is_live_camera:
            capture_source = RecentFrameCapture(
                source,
                reopen_interval=self.CAMERA_REOPEN_INTERVAL,
                queue_size=self.CAMERA_FRAME_QUEUE_SIZE,
            )
            cap = capture_source.capture
            source_fps = capture_source.fps
        else:
            capture_source = cv2.VideoCapture(source)
            cap = capture_source
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")
            source_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.FPS_ESTIMATE)
        self._active_capture = capture_source

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if frame_width == 0: frame_width = 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_height == 0: frame_height = 720

        writer = None
        raw_writer = None
        output_writer_failed = False
        raw_writer_failed = False
        if save_output:
            output_path = Path(save_output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        if save_raw:
            raw_path = Path(save_raw)
            raw_path.parent.mkdir(parents=True, exist_ok=True)

        self.event_video_buffer = EventVideoBuffer(
            self.evidence_dir / "Event_Clips_v14",
            nominal_fps=source_fps,
            pre_seconds=self.EVENT_PRE_SECONDS,
            post_seconds=self.EVENT_POST_SECONDS,
            min_free_mb=self.EVENT_MIN_FREE_MB,
        )
        print("System ready; live monitoring started (press 'q' to exit)...")
        if is_live_camera:
            print(
                f"[LIVE BUFFER] archived text {self.CAMERA_FRAME_QUEUE_SIZE} archived text:"
                "archived text,archived text."
            )
        print(
            f"[TIME STATE] crossing={self.TIME_THRESHOLD_SECONDS:.2f}s, "
            f"note_confirm={self.NOTE_CONFIRM_SECONDS:.2f}s, "
            f"note_transfer_window={self.NOTE_TRANSFER_WINDOW_SECONDS:.2f}s"
        )
        print(
            f"[EVENT VIDEO] archived text {self.EVENT_PRE_SECONDS:.1f}s + "
            f"archived text {self.EVENT_POST_SECONDS:.1f}s,archived text={self.event_video_buffer.output_dir}"
        )

        input_exhausted_notice_shown = False
        last_capture_sequence = 0
        total_dropped_frames = 0
        source_frame_index = 0
        previous_source_time = None
        processing_timestamps = deque(maxlen=30)
        while True:
            if is_live_camera:
                (
                    ret,
                    frame,
                    event_time,
                    capture_sequence,
                    dropped_frames,
                ) = capture_source.read_recent(last_capture_sequence)
                if ret:
                    last_capture_sequence = capture_sequence
                    total_dropped_frames += dropped_frames
            else:
                ret, frame = cap.read()
                if ret:
                    source_frame_index += 1
                    source_time = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    fallback_time = (source_frame_index - 1) / source_fps
                    if (
                        source_time < 0
                        or (
                            previous_source_time is not None
                            and source_time <= previous_source_time
                        )
                    ):
                        source_time = fallback_time
                    event_time = source_time
                    previous_source_time = event_time
            if not ret:
                if not is_live_camera:
                    if exit_on_eof or not display:
                        print("[VIDEO] End of video reached.")
                        break
                    if not input_exhausted_notice_shown:
                        print("[VIDEO] End of video reached; the last frame remains visible. Press 'q' to exit.")
                        input_exhausted_notice_shown = True

                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[SYSTEM] Manual exit requested.")
                    self.manual_exit_requested = True
                    break
                time.sleep(0.02)
                continue
            input_exhausted_notice_shown = False
            if self.session_event_origin is None:
                self.session_event_origin = event_time if is_live_camera else 0.0
            if is_live_camera:
                frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            self.frame_index += 1
            processing_timestamps.append(time.monotonic())
            if save_output and writer is None and not output_writer_failed:
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (frame_width, frame_height)
                )
                if not writer.isOpened():
                    print("[RECORDING] Annotated video could not be written; monitoring continues. Check the output path and available disk space.")
                    writer.release()
                    writer = None
                    output_writer_failed = True
                else:
                    self._active_writer = writer
            if save_raw and raw_writer is None and not raw_writer_failed:
                raw_writer = cv2.VideoWriter(
                    str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (frame_width, frame_height)
                )
                if not raw_writer.isOpened():
                    print("[RECORDING] Raw video could not be written; monitoring continues. Check the output path and available disk space.")
                    raw_writer.release()
                    raw_writer = None
                    raw_writer_failed = True
                else:
                    self._active_raw_writer = raw_writer
            if raw_writer is not None:
                raw_writer.write(frame)

            results_pose = self.yolo_pose.track(
                frame,
                classes=[0],
                conf=self.POSE_TRACK_CONFIDENCE,
                tracker=self.tracker_path,
                persist=True,
                verbose=False,
            )
            
            # Preserve the successful phone/door operating point. Notes use a
            # separate high-resolution, low-threshold proposal pass and are
            # never promoted without hand association and temporal support.
            prediction_classes = [1]
            if self.DOOR_CLASS_ID is not None:
                prediction_classes.append(self.DOOR_CLASS_ID)
            results_items = self.yolo_custom.predict(
                frame,
                classes=prediction_classes,
                conf=self.CUSTOM_OBJECT_CONFIDENCE,
                verbose=False,
            )
            results_paper_notes = self.yolo_custom.predict(
                frame,
                classes=[self.PAPER_CLASS_ID, self.NOTE_CLASS_ID],
                conf=min(
                    self.PAPER_COUNTER_CONFIDENCE,
                    self.NOTE_DETECTION_CONFIDENCE,
                ),
                imgsz=self.NOTE_INFERENCE_SIZE,
                verbose=False,
            )

            current_frame_wrists = {}
            person_count = 0 
            current_person_boxes = []

            if results_pose[0].boxes.id is not None and results_pose[0].keypoints is not None:
                boxes_p = results_pose[0].boxes.xyxy.cpu().numpy()
                track_ids = results_pose[0].boxes.id.cpu().numpy()
                keypoints = results_pose[0].keypoints.xy.cpu().numpy()
                keypoint_confidences = results_pose[0].keypoints.conf.cpu().numpy()

                for box, track_id, kpts, kpt_conf in zip(boxes_p, track_ids, keypoints, keypoint_confidences):
                    if person_count >= self.MAX_STUDENTS:
                        break
                    x1, y1, x2, y2 = map(int, box)
                    student_id = int(track_id)
                    current_person_boxes.append(box)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
                    
                    l_shoulder, r_shoulder = kpts[5], kpts[6]
                    l_wrist, r_wrist = kpts[9], kpts[10]
                    if kpt_conf[5] >= self.POSE_CONFIDENCE and kpt_conf[6] >= self.POSE_CONFIDENCE:
                        anchor = (
                            int((l_shoulder[0] + r_shoulder[0]) / 2),
                            int((l_shoulder[1] + r_shoulder[1]) / 2),
                        )
                    elif kpt_conf[0] >= self.POSE_CONFIDENCE:
                        anchor = (int(kpts[0][0]), int(kpts[0][1]))
                    else:
                        anchor = ((x1 + x2) // 2, y1 + max(20, (y2 - y1) // 4))

                    wrists = []
                    for wrist, wrist_confidence in ((l_wrist, kpt_conf[9]), (r_wrist, kpt_conf[10])):
                        if wrist_confidence >= self.POSE_CONFIDENCE and wrist[0] > 0 and wrist[1] > 0:
                            wrist_point = (int(wrist[0]), int(wrist[1]))
                            wrists.append(wrist_point)
                            cv2.circle(frame, wrist_point, 6, (0, 255, 0), -1)

                    if kpt_conf[5] >= self.POSE_CONFIDENCE and kpt_conf[6] >= self.POSE_CONFIDENCE:
                        shoulder_width = self.calculate_distance(l_shoulder, r_shoulder)
                    else:
                        shoulder_width = max(40, (x2 - x1) * 0.35)
                    current_frame_wrists[student_id] = {
                        'wrists': wrists,
                        'shoulder_width': shoulder_width,
                        'box_y1': y1,
                        'box_h': y2 - y1,
                        'box': box,
                        'anchor': anchor,
                        'visible_keypoints': int(sum(confidence >= self.POSE_CONFIDENCE for confidence in kpt_conf)),
                    }
                    
                    person_count += 1

            self.update_scene_context(current_frame_wrists, event_time)
            for student_id, data in current_frame_wrists.items():
                x1, y1, _, _ = map(int, data['box'])
                visibility = data['visibility']
                color = (255, 150, 0) if visibility == 'visible' else (0, 255, 255)
                label = f"Student ID: {student_id}"
                if visibility == 'partial':
                    label += " | partial"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            for student_id, state in self.track_states.items():
                if student_id not in current_frame_wrists and state['visibility'] == 'occluded':
                    anchor = tuple(map(int, state['anchor']))
                    cv2.putText(frame, f"Student ID: {student_id} | occluded (no inference)", anchor,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            valid_boxes, valid_classes = [], []
            custom_phone_records = []
            paper_boxes = np.empty((0, 4), dtype=np.float32)
            paper_confidences = np.empty((0,), dtype=np.float32)
            if len(results_items[0].boxes) > 0:
                raw_boxes = results_items[0].boxes.xyxy.cpu().numpy()
                raw_classes = results_items[0].boxes.cls.cpu().numpy()
                raw_confidences = results_items[0].boxes.conf.cpu().numpy()
                if self.DOOR_CLASS_ID is not None:
                    door_mask = raw_classes.astype(np.int32) == self.DOOR_CLASS_ID
                    self.draw_neutral_doors(
                        frame,
                        raw_boxes[door_mask],
                        raw_confidences[door_mask],
                    )
                    forbidden_mask = ~door_mask
                    raw_boxes = raw_boxes[forbidden_mask]
                    raw_classes = raw_classes[forbidden_mask]
                    raw_confidences = raw_confidences[forbidden_mask]

                phone_mask = raw_classes.astype(np.int32) == 1
                raw_phone_boxes = raw_boxes[phone_mask]
                raw_phone_confidences = raw_confidences[phone_mask]
                valid_boxes, valid_classes = self.filter_false_positives(
                    frame_width,
                    frame_height,
                    raw_phone_boxes,
                    raw_classes[phone_mask],
                    current_person_boxes,
                )
                for valid_box in valid_boxes:
                    best_confidence = max(
                        (
                            float(confidence)
                            for candidate_box, confidence in zip(
                                raw_phone_boxes,
                                raw_phone_confidences,
                            )
                            if self.calculate_iou(valid_box, candidate_box) >= 0.50
                        ),
                        default=None,
                    )
                    custom_phone_records.append(
                        {
                            "source": "custom_exam_model",
                            "class_id": 1,
                            "class_name": "phone",
                            "confidence": best_confidence,
                            "bbox_xyxy": [float(value) for value in valid_box],
                            "track_ids": self._related_track_ids(
                                valid_box,
                                current_frame_wrists,
                            ),
                        }
                    )

            note_boxes = np.empty((0, 4), dtype=np.float32)
            note_confidences = np.empty((0,), dtype=np.float32)
            if len(results_paper_notes[0].boxes) > 0:
                auxiliary_boxes = (
                    results_paper_notes[0].boxes.xyxy.cpu().numpy()
                )
                auxiliary_classes = (
                    results_paper_notes[0].boxes.cls.cpu().numpy().astype(np.int32)
                )
                auxiliary_confidences = (
                    results_paper_notes[0].boxes.conf.cpu().numpy()
                )
                paper_mask = auxiliary_classes == self.PAPER_CLASS_ID
                note_mask = auxiliary_classes == self.NOTE_CLASS_ID
                paper_boxes = auxiliary_boxes[paper_mask]
                paper_confidences = auxiliary_confidences[paper_mask]
                note_boxes = auxiliary_boxes[note_mask]
                note_confidences = auxiliary_confidences[note_mask]
                self.draw_neutral_papers(
                    frame,
                    paper_boxes,
                    paper_confidences,
                )
            note_candidates = self.update_note_candidates(
                note_boxes,
                note_confidences,
                paper_boxes,
                paper_confidences,
                current_frame_wrists,
                event_time,
            )
            confirmed_note_boxes = self.draw_note_candidates(
                frame,
                note_candidates,
            )
            confirmed_note_records = [
                {
                    "source": "custom_note_temporal",
                    "class_id": self.NOTE_CLASS_ID,
                    "class_name": "note",
                    "confidence": float(candidate["confidence"]),
                    "bbox_xyxy": [float(value) for value in candidate["box"]],
                    "track_ids": []
                    if candidate.get("holder") is None
                    else [int(candidate["holder"])],
                    "note_track_id": int(candidate["track_id"]),
                    "transfer": bool(candidate["transfer"]),
                }
                for candidate in note_candidates
                if candidate.get("confirmed")
            ]

            # The custom exam model remains authoritative.  COCO is queried only
            # when that model has no surviving phone result in this frame.
            has_custom_phone = any(int(cls) == 1 for cls in valid_classes)
            if has_custom_phone:
                self.update_coco_phone_confirmation(
                    [],
                    event_time,
                    force_reset=True,
                )
            else:
                coco_results = self.yolo_coco.predict(
                    frame,
                    classes=[self.COCO_PHONE_CLASS],
                    conf=self.COCO_PHONE_CONFIDENCE,
                    imgsz=self.COCO_PHONE_INFERENCE_SIZE,
                    verbose=False,
                )
                coco_boxes = []
                coco_confidences = []
                if len(coco_results[0].boxes) > 0:
                    raw_coco_boxes = coco_results[0].boxes.xyxy.cpu().numpy()
                    raw_coco_confidences = (
                        coco_results[0].boxes.conf.cpu().numpy()
                    )
                    coco_boxes, coco_confidences = self.filter_coco_phone_candidates(
                        frame_width,
                        frame_height,
                        raw_coco_boxes,
                        raw_coco_confidences,
                        current_person_boxes,
                    )
                    coco_candidates = [
                        (phone_box, float(confidence))
                        for phone_box, confidence in zip(
                            coco_boxes,
                            coco_confidences,
                        )
                        if (
                            confidence >= self.COCO_STRONG_CONFIDENCE
                            or self.has_rotated_custom_phone_support(
                                frame,
                                phone_box,
                            )
                        )
                    ]
                    coco_boxes = [candidate[0] for candidate in coco_candidates]
                    coco_confidences = [candidate[1] for candidate in coco_candidates]
                if self.COCO_ASSIST_MODE == "counter_evidence" and coco_boxes:
                    # Counter-evidence mode asks COCO about cup/chair as well.
                    # A COCO phone box overlapping a high-confidence distractor
                    # is discarded; COCO never replaces a custom-model result.
                    distractor_results = self.yolo_coco.predict(
                        frame,
                        classes=[41, 56],
                        conf=self.COCO_DISTRACTOR_CONFIDENCE,
                        imgsz=self.COCO_PHONE_INFERENCE_SIZE,
                        verbose=False,
                    )
                    distractor_boxes = []
                    if len(distractor_results[0].boxes) > 0:
                        distractor_boxes = distractor_results[0].boxes.xyxy.cpu().numpy().tolist()
                    retained_candidates = [
                        (phone_box, confidence)
                        for phone_box, confidence in zip(
                            coco_boxes,
                            coco_confidences,
                        )
                        if not any(
                            self.calculate_iou(phone_box, distractor_box)
                            >= self.COCO_DISTRACTOR_IOU
                            for distractor_box in distractor_boxes
                        )
                    ]
                    coco_boxes = [candidate[0] for candidate in retained_candidates]
                    coco_confidences = [candidate[1] for candidate in retained_candidates]
                confirmed_coco_phone = self.update_coco_phone_confirmation(
                    coco_boxes,
                    event_time,
                )
                if confirmed_coco_phone is not None:
                    self.draw_coco_assisted_phone(frame, confirmed_coco_phone)
                    confirmed_confidence = max(
                        (
                            float(confidence)
                            for candidate_box, confidence in zip(
                                coco_boxes,
                                coco_confidences,
                            )
                            if self.calculate_iou(
                                confirmed_coco_phone,
                                candidate_box,
                            ) >= 0.30
                        ),
                        default=None,
                    )
                    self.check_and_log_objects(
                        frame,
                        [1],
                        event_time,
                        detections=[
                            {
                                "source": "coco_auxiliary",
                                "class_id": 1,
                                "class_name": "phone",
                                "confidence": confirmed_confidence,
                                "bbox_xyxy": [
                                    float(value)
                                    for value in confirmed_coco_phone
                                ],
                                "track_ids": self._related_track_ids(
                                    confirmed_coco_phone,
                                    current_frame_wrists,
                                ),
                            }
                        ],
                    )

            if valid_boxes:
                self.draw_forbidden_objects(frame, valid_boxes, valid_classes)
                self.check_and_log_objects(
                    frame,
                    valid_classes,
                    event_time,
                    detections=custom_phone_records,
                )
            if confirmed_note_boxes:
                self.check_and_log_objects(
                    frame,
                    [self.NOTE_CLASS_ID],
                    event_time,
                    detections=confirmed_note_records,
                )

            if len(current_frame_wrists) >= 2:
                self.analyze_interactions(
                    frame,
                    current_frame_wrists,
                    event_time,
                )

            processing_fps = 0.0
            if len(processing_timestamps) >= 2:
                processing_elapsed = (
                    processing_timestamps[-1] - processing_timestamps[0]
                )
                if processing_elapsed > 0:
                    processing_fps = (
                        len(processing_timestamps) - 1
                    ) / processing_elapsed
            status_text = f"Process: {processing_fps:.1f} FPS"
            if is_live_camera:
                status_text += f" | stale frames skipped: {total_dropped_frames}"
            cv2.putText(
                frame,
                status_text,
                (12, frame_height - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 0),
                1,
            )

            if self.event_video_buffer is not None:
                self.event_video_buffer.add_frame(frame, event_time)

            if writer is not None:
                writer.write(frame)

            if display:
                cv2.imshow("Dual-Core Anti-Cheating Monitor", frame)
            
            if display and cv2.waitKey(1) & 0xFF == ord('q'):
                print("[SYSTEM] Manual exit requested.")
                self.manual_exit_requested = True
                break
            if max_frames is not None and self.frame_index >= max_frames:
                print(f"[TEST] archived text max_frames={max_frames}.")
                break

        self._release_runtime_resources()

    def run_forever(
        self,
        source=0,
        save_output=None,
        save_raw=None,
        display=True,
        exit_on_eof=False,
        max_frames=None,
    ):
        """Recover from unexpected errors and keep monitoring until manual exit."""
        self.manual_exit_requested = False
        while not self.manual_exit_requested:
            try:
                self.run_live(
                    source=source,
                    save_output=save_output,
                    save_raw=save_raw,
                    display=display,
                    exit_on_eof=exit_on_eof,
                    max_frames=max_frames,
                )
                if exit_on_eof or max_frames is not None:
                    self.manual_exit_requested = True
            except KeyboardInterrupt:
                print("[SYSTEM] Manual interruption requested.")
                self.manual_exit_requested = True
                self._release_runtime_resources()
            except Exception as error:
                self._release_runtime_resources()
                self._log_runtime_error(error)
                print(
                    f"[SYSTEM] archived text {self.evidence_dir / 'runtime_errors.log'};"
                    "1 archived text."
                )
                cv2.destroyAllWindows()
                time.sleep(1.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-Core Anti-Cheating Monitor")
    parser.add_argument("--source", default="0", help="camera index such as 0, or a path to a recorded video")
    parser.add_argument("--save-output", help="optional path for the annotated replay video")
    parser.add_argument("--save-raw", help="optional path for the unannotated camera recording used for future regression tests")
    parser.add_argument("--no-display", action="store_true", help="disable the preview window")
    parser.add_argument("--exit-on-eof", action="store_true", help="exit after a video file ends")
    parser.add_argument("--max-frames", type=int, help="optional smoke-test frame limit")
    arguments = parser.parse_args()
    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    system = DualCoreAntiCheatingSystem()
    system.run_forever(
        source=source,
        save_output=arguments.save_output,
        save_raw=arguments.save_raw,
        display=not arguments.no_display,
        exit_on_eof=arguments.exit_on_eof,
        max_frames=arguments.max_frames,
    )
