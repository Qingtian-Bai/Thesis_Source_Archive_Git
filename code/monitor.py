# import cv2
# import math
# from ultralytics import YOLO
# import os
# import json
# from datetime import datetime

# class DualCoreAntiCheatingSystem:
#     def __init__(self, custom_model_path="Model_Vault/yolov10n_4class_v1_baseline_epoch78.pt"):
#         print("Starting the dual-core monitoring system: YOLOv8-Pose for people plus a custom object model...")
        
#         print("Loading core model 1: YOLOv8n-Pose for person tracking and full-frame keypoint extraction...")
#         self.yolo_pose = YOLO("yolov8n-pose.pt")
        
#         print("Loading core model 2: the custom prohibited-item detector...")
#         self.yolo_custom = YOLO(custom_model_path)
#         self.custom_class_names = {1: 'phone', 2: 'paper', 3: 'note'}
        
#         self.FPS_ESTIMATE = 30
#         self.TIME_THRESHOLD_SECONDS = 0.5    
#         self.FRAMES_TO_TRIGGER = int(self.FPS_ESTIMATE * self.TIME_THRESHOLD_SECONDS) 
#         self.contact_timers = {}

#         self.evidence_dir = "Evidence_Vault"
#         os.makedirs(self.evidence_dir, exist_ok=True)  
#         self.last_capture_time = {}                    
#         self.CAPTURE_COOLDOWN = 5.0                    

#     @staticmethod
#     def calculate_distance(p1, p2):
#         return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

#     # ==============================================================
#     # ==============================================================
#     def filter_false_positives(self, frame_width, frame_height, item_boxes, item_classes, person_boxes):
#         valid_boxes = []
#         valid_classes = []
        
#         for box, cls in zip(item_boxes, item_classes):
#             ix1, iy1, ix2, iy2 = map(int, box)
#             cls_id = int(cls)
            
#             item_w = ix2 - ix1
#             item_h = iy2 - iy1
#             item_area = item_w * item_h

#             if cls_id == 1 and item_area > (frame_width * frame_height * 0.15): 

#             is_near_person = False
#             for p_box in person_boxes:
#                 px1, py1, px2, py2 = p_box
#                 margin = 80
#                 if (px1 - margin) < icx < (px2 + margin) and (py1 - margin) < icy < (py2 + margin):
#                     is_near_person = True
#                     break
            
#             if not is_near_person:
#                 # cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (100, 100, 100), 1)
                
#             valid_boxes.append(box)
#             valid_classes.append(cls)
            
#         return valid_boxes, valid_classes

#     def draw_forbidden_objects(self, frame, boxes, classes):
#         for box, cls in zip(boxes, classes):
#             x1, y1, x2, y2 = map(int, box)
#             cls_id = int(cls)
            
#             if cls_id in self.custom_class_names:
#                 obj_name = self.custom_class_names[cls_id]
                
#                 label = f"WARNING: {obj_name.upper()}"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 cv2.putText(frame, label, (x1, max(20, y1 - 10)), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#     def check_and_log_objects(self, frame, classes):
#         current_time = datetime.now()
#         detected_custom_classes = set(int(cls) for cls in classes if int(cls) in self.custom_class_names)
        
#         for cls_id in detected_custom_classes:
#             cooldown_key = f"object_{cls_id}"
#             last_time = self.last_capture_time.get(cooldown_key)
            
#             if last_time is None or (current_time - last_time).total_seconds() > self.CAPTURE_COOLDOWN:
#                 obj_name = self.custom_class_names[cls_id]
#                 timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
#                 img_filename = f"Violation_Object_{obj_name}_{timestamp_str}.jpg"
#                 img_path = os.path.join(self.evidence_dir, img_filename)
                
#                 cv2.imwrite(img_path, frame)
                
#                 log_data = {
#                     "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
#                     "alert_type": f"Forbidden Object Detected: {obj_name.upper()}",
#                     "image_path": img_path
#                 }
#                 day_str = current_time.strftime("%Y%m%d")
#                 log_file = os.path.join(self.evidence_dir, f"log_{day_str}.jsonl")
#                 with open(log_file, "a", encoding="utf-8") as f:
#                     f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    
#                 self.last_capture_time[cooldown_key] = current_time
#                 print(f"[EVIDENCE SAVED] Prohibited-item alert evidence saved: {img_filename}")

#     def analyze_interactions(self, frame, current_frame_wrists):
#         student_ids = list(current_frame_wrists.keys())
#         current_frame_pairs = []

#         if len(student_ids) >= 2:
#             for i in range(len(student_ids)):
#                 for j in range(i + 1, len(student_ids)):
#                     id_A, id_B = student_ids[i], student_ids[j]
#                     pair_key = tuple(sorted([id_A, id_B]))
#                     current_frame_pairs.append(pair_key)

#                     data_A = current_frame_wrists[id_A]
#                     data_B = current_frame_wrists[id_B]

#                     sw_A, sw_B = data_A['shoulder_width'], data_B['shoulder_width']
#                     if max(sw_A, sw_B) == 0: 
#                         continue
                        
#                     sw_ratio = min(sw_A, sw_B) / max(sw_A, sw_B)
#                     if sw_ratio < 0.70: 
#                         continue

#                     y1_A, y1_B = data_A['box_y1'], data_B['box_y1']
#                     y_diff = abs(y1_A - y1_B)
#                     avg_box_h = (data_A['box_h'] + data_B['box_h']) / 2.0
                    
#                     if y_diff > avg_box_h * 0.25:
#                         continue
                        
#                     wrists_A = [data_A['left'], data_A['right']]
#                     wrists_B = [data_B['left'], data_B['right']]

#                     min_dist = float('inf')
#                     closest_pair = None
#                     for w_a in wrists_A:
#                         for w_b in wrists_B:
#                             dist = self.calculate_distance(w_a, w_b)
#                             if dist < min_dist:
#                                 min_dist, closest_pair = dist, (w_a, w_b)

#                     dynamic_threshold = max(80, ((sw_A + sw_B) / 2.0) * 1.8)

#                     if closest_pair and min_dist < dynamic_threshold:
#                         self.contact_timers[pair_key] = min(self.contact_timers.get(pair_key, 0) + 2, self.FRAMES_TO_TRIGGER)
#                     else:
#                         if self.contact_timers.get(pair_key, 0) > 0:
#                             self.contact_timers[pair_key] -= 1

#                     current_duration = self.contact_timers.get(pair_key, 0)

#                     if current_duration > 0:
#                         if current_duration < self.FRAMES_TO_TRIGGER:
#                             cv2.line(frame, closest_pair[0], closest_pair[1], (0, 255, 255), 2)
#                             cv2.putText(frame, f"Suspicious: {id_A} & {id_B} ({current_duration}/{self.FRAMES_TO_TRIGGER})", 
#                                         (50, 100 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                         else:
#                             cv2.line(frame, closest_pair[0], closest_pair[1], (0, 0, 255), 5)
#                             cv2.putText(frame, f"ALERT: Passing Notes! {id_A} & {id_B}", 
#                                         (50, 100 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

#                             current_time = datetime.now()
#                             last_time = self.last_capture_time.get(pair_key)

#                             if last_time is None or (current_time - last_time).total_seconds() > self.CAPTURE_COOLDOWN:
#                                 timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
#                                 img_filename = f"Violation_{id_A}_and_{id_B}_{timestamp_str}.jpg"
#                                 img_path = os.path.join(self.evidence_dir, img_filename)

#                                 cv2.imwrite(img_path, frame)

#                                 log_data = {
#                                     "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
#                                     "student_A": id_A,
#                                     "student_B": id_B,
#                                     "alert_type": "Spatial Invasion / Potential Passing Notes",
#                                     "image_path": img_path
#                                 }

#                                 day_str = current_time.strftime("%Y%m%d")
#                                 log_file = os.path.join(self.evidence_dir, f"log_{day_str}.jsonl")
#                                 with open(log_file, "a", encoding="utf-8") as f:
#                                     f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

#                                 self.last_capture_time[pair_key] = current_time
#                                 print(f"[EVIDENCE SAVED] Interaction-risk evidence saved: {img_filename}")

#         for key in list(self.contact_timers.keys()):
#             if key not in current_frame_pairs and self.contact_timers[key] > 0:
#                 self.contact_timers[key] -= 1

#     def run_live(self):
#         cap = cv2.VideoCapture(0)
        
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         if frame_width == 0: frame_width = 1280
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         if frame_height == 0: frame_height = 720
        
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#         print("System ready; live monitoring started (press 'q' to exit)...")

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.flip(frame, 1)

#             results_pose = self.yolo_pose.track(frame, classes=[0], conf=0.4, tracker="botsort.yaml", persist=True, verbose=False)
            
#             results_items = self.yolo_custom.predict(frame, classes=[1, 2, 3], conf=0.45, verbose=False)

#             current_frame_wrists = {}
#             person_count = 0 
            
#             current_person_boxes = []

#             if results_pose[0].boxes.id is not None and results_pose[0].keypoints is not None:
#                 boxes_p = results_pose[0].boxes.xyxy.cpu().numpy()
#                 track_ids = results_pose[0].boxes.id.cpu().numpy()
#                 keypoints = results_pose[0].keypoints.xy.cpu().numpy()

#                 for box, track_id, kpts in zip(boxes_p, track_ids, keypoints):
#                     if person_count >= 5: break 
                    
#                     current_person_boxes.append(box)
                    
#                     x1, y1, x2, y2 = map(int, box)
#                     student_id = int(track_id)

#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
#                     cv2.putText(frame, f"Student ID: {student_id}", (x1, max(20, y1 - 10)), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
                    
#                     l_shoulder, r_shoulder = kpts[5], kpts[6]
#                     l_wrist, r_wrist = kpts[9], kpts[10]

#                     if l_wrist[0] > 0 and r_wrist[0] > 0 and l_shoulder[0] > 0 and r_shoulder[0] > 0:
#                         glx, gly = int(l_wrist[0]), int(l_wrist[1])
#                         grx, gry = int(r_wrist[0]), int(r_wrist[1])
                        
#                         shoulder_width = self.calculate_distance(
#                             (int(l_shoulder[0]), int(l_shoulder[1])), 
#                             (int(r_shoulder[0]), int(r_shoulder[1]))
#                         )

#                         cv2.circle(frame, (glx, gly), 6, (0, 255, 0), -1)
#                         cv2.circle(frame, (grx, gry), 6, (0, 255, 0), -1)

#                         current_frame_wrists[student_id] = {
#                             'left': (glx, gly), 
#                             'right': (grx, gry), 
#                             'shoulder_width': shoulder_width,
#                             'box_y1': y1,             
#                             'box_h': y2 - y1          
#                         }
                    
#                     person_count += 1

#             if len(results_items[0].boxes) > 0:
#                 raw_boxes = results_items[0].boxes.xyxy.cpu().numpy()
#                 raw_classes = results_items[0].boxes.cls.cpu().numpy()
                
#                 valid_boxes, valid_classes = self.filter_false_positives(
#                     frame_width, frame_height, raw_boxes, raw_classes, current_person_boxes
#                 )
                
#                 if len(valid_boxes) > 0:
#                     self.draw_forbidden_objects(frame, valid_boxes, valid_classes)
#                     self.check_and_log_objects(frame, valid_classes)

#             if len(current_frame_wrists) >= 2:
#                 self.analyze_interactions(frame, current_frame_wrists)

#             cv2.imshow("Dual-Core Anti-Cheating Monitor", frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     MODEL_PATH = r"D:\project zero\YOLO_Project\Model_Vault\yolov10n_4class_v1_baseline_epoch78.pt" 
    
#     system = DualCoreAntiCheatingSystem(custom_model_path=MODEL_PATH)
#     system.run_live()
import cv2
import math
import argparse
import time
import traceback
import numpy as np
from ultralytics import YOLO
import os
import json
from datetime import datetime
import ultralytics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

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
            custom_model_path = PROJECT_ROOT / "Model_Vault" / "yolov10n_4class_v1_baseline_epoch78.pt"
        self.yolo_custom = YOLO(str(custom_model_path))
        # Only a phone and a temporally confirmed small note are forbidden.
        # Ordinary exam paper is contextual counter-evidence, like a door.
        self.custom_class_names = {1: 'phone', 3: 'note'}
        model_names = self.yolo_custom.names
        if isinstance(model_names, dict):
            model_name_items = model_names.items()
        else:
            model_name_items = enumerate(model_names)
        self.DOOR_CLASS_ID = next(
            (int(class_id) for class_id, name in model_name_items if str(name).lower() == "door"),
            None,
        )
        # Door is contextual counter-evidence, never a forbidden object. It is
        # shown neutrally but never suppresses a real phone held in front of it.
        self.DOOR_NEUTRAL_CONFIDENCE = 0.65
        # Keep the frozen baseline at its original 0.45 threshold. Test
        # launchers may override this without changing the formal baseline.
        self.CUSTOM_OBJECT_CONFIDENCE = float(
            os.environ.get("CUSTOM_OBJECT_CONFIDENCE", "0.45")
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
        self.NOTE_CONFIRM_FRAMES = int(
            os.environ.get("NOTE_CONFIRM_FRAMES", "4")
        )
        self.NOTE_MAX_MISSES = 4
        self.NOTE_HAND_DISTANCE_FACTOR = 0.80
        self.NOTE_MAX_PERSON_AREA_RATIO = 0.12
        self.NOTE_MAX_SHOULDER_WIDTH_RATIO = 0.55
        self.NOTE_MAX_SHOULDER_HEIGHT_RATIO = 0.65
        self.note_tracks = {}
        self.next_note_track_id = 1

        self.POSE_CONFIDENCE = 0.45
        self.MAX_STUDENTS = 30
        self.COCO_PHONE_CLASS = 67
        self.COCO_PHONE_CONFIDENCE = 0.80
        self.COCO_CONFIRM_FRAMES = 3
        # ``fallback`` preserves the current behavior.  The v4 test launcher
        # uses ``counter_evidence`` so COCO never creates or replaces a phone
        # alert; the custom exam model remains authoritative.
        self.COCO_ASSIST_MODE = os.environ.get("COCO_ASSIST_MODE", "fallback").strip().lower()
        # Phone boxes are expected to lie on the tracked student's body/arms.
        # A small allowance preserves hand-held phones without admitting nearby chairs.
        self.PHONE_PERSON_MARGIN = 30
        self.coco_phone_candidate_box = None
        self.coco_phone_candidate_frames = 0
        
        self.FPS_ESTIMATE = 30
        self.TIME_THRESHOLD_SECONDS = 0.5    
        self.FRAMES_TO_TRIGGER = int(self.FPS_ESTIMATE * self.TIME_THRESHOLD_SECONDS) 
        self.crossing_timers = {}
        self.track_states = {}
        self.frame_index = 0
        self.OCCLUSION_GRACE_FRAMES = 45
        # A live invigilation session must only stop on an explicit user action.
        # Camera read failures are retried indefinitely instead of terminating.
        self.CAMERA_REOPEN_INTERVAL = 100

        self.evidence_dir = PROJECT_ROOT / "Evidence_Vault"
        os.makedirs(self.evidence_dir, exist_ok=True)  
        self.last_capture_time = {}                    
        self.CAPTURE_COOLDOWN = 5.0                    
        self.manual_exit_requested = False

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

    @staticmethod
    def calculate_skin_ratio(frame, box):
        """Return the fraction of the box that resembles exposed human skin in HSV color space."""
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, x1), min(frame_width, x2)
        y1, y2 = max(0, y1), min(frame_height, y2)
        if x2 <= x1 or y2 <= y1:
            return 1.0
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        skin_low_hue = cv2.inRange(hsv, (0, 20, 50), (25, 255, 255))
        skin_high_hue = cv2.inRange(hsv, (160, 20, 50), (179, 255, 255))
        skin_pixels = cv2.countNonZero(cv2.bitwise_or(skin_low_hue, skin_high_hue))
        return skin_pixels / (hsv.shape[0] * hsv.shape[1])

    def _filter_false_positives_extended(
        self, frame, frame_width, frame_height, item_boxes, item_classes,
        item_confidences, item_sources, person_boxes,
    ):
        """archived text、archived text."""
        valid_boxes = []
        valid_classes = []
        
        custom_phone_candidates = [
            (candidate_box, candidate_confidence)
            for candidate_box, candidate_cls, candidate_confidence, candidate_source in zip(
                item_boxes, item_classes, item_confidences, item_sources
            )
            if int(candidate_cls) == 1 and candidate_source == "custom"
        ]

        for box, cls, confidence, source in zip(item_boxes, item_classes, item_confidences, item_sources):
            ix1, iy1, ix2, iy2 = map(int, box)
            cls_id = int(cls)
            
            item_w = ix2 - ix1
            item_h = iy2 - iy1
            item_area = item_w * item_h
            icx, icy = (ix1 + ix2) / 2, (iy1 + iy2) / 2

            if cls_id == 1:
                if source == "coco" and confidence >= self.COCO_UNSUPPORTED_HIGH_CONFIDENCE:
                    has_custom_support = any(
                        custom_confidence >= self.COCO_CUSTOM_SUPPORT_CONFIDENCE
                        and self.calculate_iou(box, custom_box) >= self.COCO_CUSTOM_SUPPORT_IOU
                        for custom_box, custom_confidence in custom_phone_candidates
                    )
                    if not has_custom_support:
                        continue
                area_ratio = item_area / (frame_width * frame_height)
                aspect_ratio = max(item_w, item_h) / max(1, min(item_w, item_h))
                min_confidence = self.PHONE_MIN_CONFIDENCE
                if aspect_ratio < self.PHONE_MIN_ASPECT_RATIO:
                    min_confidence = self.PHONE_NEAR_SQUARE_MIN_CONFIDENCE
                skin_ratio = self.calculate_skin_ratio(frame, box)
                if skin_ratio >= self.SKIN_PHONE_REJECT_RATIO:
                    continue
                if skin_ratio >= self.SKIN_PHONE_HIGH_CONF_RATIO:
                    min_confidence = max(min_confidence, self.SKIN_PHONE_HIGH_CONFIDENCE)
                if (
                    confidence < min_confidence
                    or area_ratio > self.PHONE_MAX_AREA_RATIO
                ):
                    continue
            elif confidence < self.OTHER_ITEM_MIN_CONFIDENCE:
                continue

            is_near_person = False
            for p_box in person_boxes:
                px1, py1, px2, py2 = p_box
                margin = 80
                if (px1 - margin) < icx < (px2 + margin) and (py1 - margin) < icy < (py2 + margin):
                    is_near_person = True
                    break
            
            if not is_near_person:
                continue 
                
            valid_boxes.append(box)
            valid_classes.append(cls)
            
        return valid_boxes, valid_classes

    def filter_false_positives(self, frame_width, frame_height, item_boxes, item_classes, person_boxes):
        """Initial object filter: reject only oversized phones and background objects."""
        valid_boxes = []
        valid_classes = []

        for box, cls in zip(item_boxes, item_classes):
            ix1, iy1, ix2, iy2 = map(int, box)
            cls_id = int(cls)
            item_area = (ix2 - ix1) * (iy2 - iy1)
            icx, icy = (ix1 + ix2) / 2, (iy1 + iy2) / 2

            if cls_id == 1 and item_area > frame_width * frame_height * 0.15:
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

    def update_coco_phone_confirmation(self, coco_boxes):
        """Only promote a COCO candidate after it remains at the same location briefly."""
        if not coco_boxes:
            self.coco_phone_candidate_box = None
            self.coco_phone_candidate_frames = 0
            return None

        candidate = coco_boxes[0]
        if (
            self.coco_phone_candidate_box is not None
            and self.calculate_iou(candidate, self.coco_phone_candidate_box) >= 0.35
        ):
            self.coco_phone_candidate_frames += 1
        else:
            self.coco_phone_candidate_box = candidate
            self.coco_phone_candidate_frames = 1

        self.coco_phone_candidate_box = candidate
        if self.coco_phone_candidate_frames >= self.COCO_CONFIRM_FRAMES:
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

    def has_valid_person_pose(self, box, keypoints, confidences):
        """archived text,archived text、archived text person."""
        if confidences is None:
            return False

        visible_count = sum(confidence >= self.POSE_CONFIDENCE for confidence in confidences)
        body_count = sum(confidences[index] >= self.POSE_CONFIDENCE for index in (5, 6, 9, 10, 11, 12))
        if visible_count < 4 or body_count < 2:
            return False

        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            return False

        anchor_indices = [index for index in (0, 5, 6) if confidences[index] >= self.POSE_CONFIDENCE]
        if not anchor_indices:
            return False
        if not any(x1 <= keypoints[index][0] <= x2 and y1 <= keypoints[index][1] <= y2 for index in anchor_indices):
            return False

        if confidences[5] >= self.POSE_CONFIDENCE and confidences[6] >= self.POSE_CONFIDENCE:
            shoulder_width = self.calculate_distance(keypoints[5], keypoints[6])
            if not box_w * 0.12 <= shoulder_width <= box_w * 1.1:
                return False

        return True

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
    ):
        """Track low-threshold note proposals and promote only persistent hand-held ones."""
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

        unmatched_tracks = set(self.note_tracks)
        current_track_ids = []
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
                    "hits": 0,
                    "misses": 0,
                    "observations": [],
                }
            else:
                unmatched_tracks.remove(best_track_id)

            track = self.note_tracks[best_track_id]
            track["box"] = candidate["box"]
            track["hits"] += 1
            track["misses"] = 0
            track["observations"].append(candidate["holder"])
            track["observations"] = track["observations"][-30:]
            candidate["track_id"] = best_track_id
            hand_observations = sum(
                holder is not None for holder in track["observations"]
            )
            candidate["confirmed"] = (
                track["hits"] >= self.NOTE_CONFIRM_FRAMES
                and hand_observations >= 2
                and candidate["holder"] is not None
            )
            holder_counts = {}
            for holder in track["observations"]:
                if holder is not None:
                    holder_counts[holder] = holder_counts.get(holder, 0) + 1
            candidate["transfer"] = sum(
                count >= 2 for count in holder_counts.values()
            ) >= 2
            current_track_ids.append(best_track_id)

        for track_id in list(self.note_tracks):
            if track_id in current_track_ids:
                continue
            track = self.note_tracks[track_id]
            track["misses"] += 1
            track["hits"] = max(0, track["hits"] - 1)
            if track["misses"] > self.NOTE_MAX_MISSES:
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

    def check_and_log_objects(self, frame, classes):
        current_time = datetime.now()
        detected_custom_classes = set(int(cls) for cls in classes if int(cls) in self.custom_class_names)
        
        for cls_id in detected_custom_classes:
            cooldown_key = f"object_{cls_id}"
            last_time = self.last_capture_time.get(cooldown_key)
            
            if last_time is None or (current_time - last_time).total_seconds() > self.CAPTURE_COOLDOWN:
                obj_name = self.custom_class_names[cls_id]
                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                img_filename = f"Violation_Object_{obj_name}_{timestamp_str}.jpg"
                img_path = os.path.join(self.evidence_dir, img_filename)
                
                cv2.imwrite(img_path, frame)
                
                log_data = {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "alert_type": f"Forbidden Object Detected: {obj_name.upper()}",
                    "image_path": img_path
                }
                day_str = current_time.strftime("%Y%m%d")
                log_file = os.path.join(self.evidence_dir, f"log_{day_str}.jsonl")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    
                self.last_capture_time[cooldown_key] = current_time
                print(f"[EVIDENCE SAVED] Prohibited-item alert evidence saved: {img_filename}")

    def update_scene_context(self, students):
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
                    'last_seen_frame': self.frame_index,
                    'anchor': data['anchor'],
                    'depth_layer': data['depth_layer'],
                    'visibility': data['visibility'],
                }

        for student_id, state in list(self.track_states.items()):
            unseen_frames = self.frame_index - state['last_seen_frame']
            if student_id not in students and unseen_frames <= self.OCCLUSION_GRACE_FRAMES:
                state['visibility'] = 'occluded'
            elif unseen_frames > self.OCCLUSION_GRACE_FRAMES:
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

    def analyze_interactions(self, frame, students):
        """Detect persistent cross-track reaching, not instantaneous wrist-to-wrist distance."""
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

        for pair_key, (wrist, target_anchor) in candidate_pairs.items():
            self.crossing_timers[pair_key] = min(
                self.crossing_timers.get(pair_key, 0) + 1,
                self.FRAMES_TO_TRIGGER,
            )
            duration = self.crossing_timers[pair_key]
            color = (0, 255, 255) if duration < self.FRAMES_TO_TRIGGER else (0, 0, 255)
            cv2.line(frame, wrist, tuple(map(int, target_anchor)), color, 2 if duration < self.FRAMES_TO_TRIGGER else 4)
            label = f"Cross-track reach: {pair_key[0]} & {pair_key[1]} ({duration}/{self.FRAMES_TO_TRIGGER})"
            cv2.putText(frame, label, (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            if duration == self.FRAMES_TO_TRIGGER:
                current_time = datetime.now()
                last_time = self.last_capture_time.get(pair_key)
                if last_time is None or (current_time - last_time).total_seconds() > self.CAPTURE_COOLDOWN:
                    timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                    img_filename = f"CrossTrack_{pair_key[0]}_{pair_key[1]}_{timestamp_str}.jpg"
                    img_path = os.path.join(self.evidence_dir, img_filename)
                    cv2.imwrite(img_path, frame)
                    log_data = {
                        "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "students": list(pair_key),
                        "alert_type": "Persistent cross-track reach",
                        "image_path": str(img_path),
                    }
                    log_file = os.path.join(self.evidence_dir, f"log_{current_time.strftime('%Y%m%d')}.jsonl")
                    with open(log_file, "a", encoding="utf-8") as handle:
                        handle.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    self.last_capture_time[pair_key] = current_time

        for pair_key in list(self.crossing_timers):
            if pair_key not in candidate_pairs:
                self.crossing_timers[pair_key] = max(0, self.crossing_timers[pair_key] - 1)

    def run_live(self, source=0, save_output=None, save_raw=None):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if frame_width == 0: frame_width = 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_height == 0: frame_height = 720
        
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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

        print("System ready; live monitoring started (press 'q' to exit)...")

        consecutive_read_failures = 0
        input_exhausted_notice_shown = False
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(source, int):
                    consecutive_read_failures += 1
                    if consecutive_read_failures == 1:
                        print("[CAMERA] No frame was read; monitoring remains active and will retry automatically. Press 'q' to exit.")
                    if consecutive_read_failures % self.CAMERA_REOPEN_INTERVAL == 0:
                        print("[CAMERA] Capture has not recovered; reconnecting to the camera...")
                        cap.release()
                        time.sleep(0.2)
                        cap = cv2.VideoCapture(source)
                else:
                    if not input_exhausted_notice_shown:
                        print("[VIDEO] End of video reached; the last frame remains visible. Press 'q' to exit.")
                        input_exhausted_notice_shown = True

                # Keep the preview window responsive while retrying/waiting so
                # the user can always terminate explicitly with the q key.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[SYSTEM] Manual exit requested.")
                    self.manual_exit_requested = True
                    break
                time.sleep(0.05)
                continue
            consecutive_read_failures = 0
            input_exhausted_notice_shown = False
            if isinstance(source, int):
                frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            self.frame_index += 1
            if save_output and writer is None and not output_writer_failed:
                fps = cap.get(cv2.CAP_PROP_FPS) or self.FPS_ESTIMATE
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
                )
                if not writer.isOpened():
                    print("[RECORDING] Annotated video could not be written; monitoring continues. Check the output path and available disk space.")
                    writer.release()
                    writer = None
                    output_writer_failed = True
            if save_raw and raw_writer is None and not raw_writer_failed:
                fps = cap.get(cv2.CAP_PROP_FPS) or self.FPS_ESTIMATE
                raw_writer = cv2.VideoWriter(
                    str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
                )
                if not raw_writer.isOpened():
                    print("[RECORDING] Raw video could not be written; monitoring continues. Check the output path and available disk space.")
                    raw_writer.release()
                    raw_writer = None
                    raw_writer_failed = True
            if raw_writer is not None:
                raw_writer.write(frame)

            results_pose = self.yolo_pose.track(frame, classes=[0], conf=0.4, tracker=self.tracker_path, persist=True, verbose=False)
            
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

            self.update_scene_context(current_frame_wrists)
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
                valid_boxes, valid_classes = self.filter_false_positives(
                    frame_width,
                    frame_height,
                    raw_boxes[phone_mask],
                    raw_classes[phone_mask],
                    current_person_boxes,
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
            )
            confirmed_note_boxes = self.draw_note_candidates(
                frame,
                note_candidates,
            )

            # The custom exam model remains authoritative.  COCO is queried only
            # when that model has no surviving phone result in this frame.
            has_custom_phone = any(int(cls) == 1 for cls in valid_classes)
            if has_custom_phone:
                self.update_coco_phone_confirmation([])
            else:
                coco_results = self.yolo_coco.predict(
                    frame,
                    classes=[self.COCO_PHONE_CLASS],
                    conf=self.COCO_PHONE_CONFIDENCE,
                    verbose=False,
                )
                coco_boxes = []
                if len(coco_results[0].boxes) > 0:
                    raw_coco_boxes = coco_results[0].boxes.xyxy.cpu().numpy()
                    # Convert COCO's class 67 to this program's phone class 1,
                    # then reuse the exact same area/person-association filter.
                    coco_phone_classes = np.ones(len(raw_coco_boxes), dtype=np.int32)
                    coco_boxes, _ = self.filter_false_positives(
                        frame_width, frame_height, raw_coco_boxes, coco_phone_classes, current_person_boxes
                    )
                if self.COCO_ASSIST_MODE == "counter_evidence" and coco_boxes:
                    # Counter-evidence mode asks COCO about cup/chair as well.
                    # A COCO phone box overlapping a high-confidence distractor
                    # is discarded; COCO never replaces a custom-model result.
                    distractor_results = self.yolo_coco.predict(
                        frame, classes=[41, 56], conf=0.70, verbose=False
                    )
                    distractor_boxes = []
                    if len(distractor_results[0].boxes) > 0:
                        distractor_boxes = distractor_results[0].boxes.xyxy.cpu().numpy().tolist()
                    coco_boxes = [
                        phone_box for phone_box in coco_boxes
                        if not any(self.calculate_iou(phone_box, distractor_box) >= 0.25 for distractor_box in distractor_boxes)
                    ]
                confirmed_coco_phone = self.update_coco_phone_confirmation(coco_boxes)
                if confirmed_coco_phone is not None:
                    self.draw_coco_assisted_phone(frame, confirmed_coco_phone)
                    # Keep evidence semantics consistent with a normal phone alert;
                    # the on-screen yellow label makes the source explicit for testing.
                    self.check_and_log_objects(frame, [1])

            if valid_boxes:
                self.draw_forbidden_objects(frame, valid_boxes, valid_classes)
                self.check_and_log_objects(frame, valid_classes)
            if confirmed_note_boxes:
                self.check_and_log_objects(frame, [self.NOTE_CLASS_ID])

            if len(current_frame_wrists) >= 2:
                self.analyze_interactions(frame, current_frame_wrists)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Dual-Core Anti-Cheating Monitor", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[SYSTEM] Manual exit requested.")
                self.manual_exit_requested = True
                break

        cap.release()
        if writer is not None:
            writer.release()
        if raw_writer is not None:
            raw_writer.release()
        cv2.destroyAllWindows()

    def run_forever(self, source=0, save_output=None, save_raw=None):
        """Recover from unexpected errors and keep monitoring until manual exit."""
        self.manual_exit_requested = False
        while not self.manual_exit_requested:
            try:
                self.run_live(source=source, save_output=save_output, save_raw=save_raw)
            except KeyboardInterrupt:
                print("[SYSTEM] Manual interruption requested.")
                self.manual_exit_requested = True
            except Exception as error:
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
    arguments = parser.parse_args()
    source = int(arguments.source) if arguments.source.isdigit() else arguments.source
    system = DualCoreAntiCheatingSystem()
    system.run_forever(source=source, save_output=arguments.save_output, save_raw=arguments.save_raw)
