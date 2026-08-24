import cv2
import math
from ultralytics import YOLO
import os
import json
from datetime import datetime

class DualCoreAntiCheatingSystem:
    def __init__(self, custom_model_path="Model_Vault/yolov10n_4class_v1_baseline_epoch78.pt"):
        print("Starting the dual-core monitoring system: YOLOv8-Pose for people plus a custom object model...")
        
        print("Loading core model 1: YOLOv8n-Pose for person tracking and full-frame keypoint extraction...")
        self.yolo_pose = YOLO("yolov8n-pose.pt")
        
        print("Loading core model 2: the custom prohibited-item detector...")
        self.yolo_custom = YOLO(custom_model_path)
        self.custom_class_names = {1: 'phone', 2: 'paper', 3: 'note'}
        
        self.FPS_ESTIMATE = 30
        self.TIME_THRESHOLD_SECONDS = 0.5    
        self.FRAMES_TO_TRIGGER = int(self.FPS_ESTIMATE * self.TIME_THRESHOLD_SECONDS) 
        self.contact_timers = {}

        self.evidence_dir = "Evidence_Vault"
        os.makedirs(self.evidence_dir, exist_ok=True)  
        self.last_capture_time = {}                    
        self.CAPTURE_COOLDOWN = 5.0                    

    @staticmethod
    def calculate_distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # ==============================================================
    # ==============================================================
    def filter_false_positives(self, frame_width, frame_height, item_boxes, item_classes, person_boxes):
        """archived text:archived text、archived text"""
        valid_boxes = []
        valid_classes = []
        
        for box, cls in zip(item_boxes, item_classes):
            ix1, iy1, ix2, iy2 = map(int, box)
            cls_id = int(cls)
            
            item_w = ix2 - ix1
            item_h = iy2 - iy1
            item_area = item_w * item_h
            icx, icy = (ix1 + ix2) / 2, (iy1 + iy2) / 2

            if cls_id == 1 and item_area > (frame_width * frame_height * 0.15): 
                continue

            is_near_person = False
            for p_box in person_boxes:
                px1, py1, px2, py2 = p_box
                margin = 80
                if (px1 - margin) < icx < (px2 + margin) and (py1 - margin) < icy < (py2 + margin):
                    is_near_person = True
                    break
            
            if not is_near_person:
                # cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (100, 100, 100), 1)
                continue
                
            valid_boxes.append(box)
            valid_classes.append(cls)
            
        return valid_boxes, valid_classes

    def draw_forbidden_objects(self, frame, boxes, classes):
        """archived text"""
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

    def check_and_log_objects(self, frame, classes):
        """archived text"""
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

    def analyze_interactions(self, frame, current_frame_wrists):
        """archived text,archived text,archived text"""
        student_ids = list(current_frame_wrists.keys())
        current_frame_pairs = []

        if len(student_ids) >= 2:
            for i in range(len(student_ids)):
                for j in range(i + 1, len(student_ids)):
                    id_A, id_B = student_ids[i], student_ids[j]
                    pair_key = tuple(sorted([id_A, id_B]))
                    current_frame_pairs.append(pair_key)

                    data_A = current_frame_wrists[id_A]
                    data_B = current_frame_wrists[id_B]

                    sw_A, sw_B = data_A['shoulder_width'], data_B['shoulder_width']
                    if max(sw_A, sw_B) == 0: 
                        continue
                        
                    sw_ratio = min(sw_A, sw_B) / max(sw_A, sw_B)
                    if sw_ratio < 0.70: 
                        continue

                    y1_A, y1_B = data_A['box_y1'], data_B['box_y1']
                    y_diff = abs(y1_A - y1_B)
                    avg_box_h = (data_A['box_h'] + data_B['box_h']) / 2.0
                    
                    if y_diff > avg_box_h * 0.25:
                        continue
                        
                    wrists_A = [data_A['left'], data_A['right']]
                    wrists_B = [data_B['left'], data_B['right']]

                    min_dist = float('inf')
                    closest_pair = None
                    for w_a in wrists_A:
                        for w_b in wrists_B:
                            dist = self.calculate_distance(w_a, w_b)
                            if dist < min_dist:
                                min_dist, closest_pair = dist, (w_a, w_b)

                    dynamic_threshold = max(80, ((sw_A + sw_B) / 2.0) * 1.8)

                    if closest_pair and min_dist < dynamic_threshold:
                        self.contact_timers[pair_key] = min(self.contact_timers.get(pair_key, 0) + 2, self.FRAMES_TO_TRIGGER)
                    else:
                        if self.contact_timers.get(pair_key, 0) > 0:
                            self.contact_timers[pair_key] -= 1

                    current_duration = self.contact_timers.get(pair_key, 0)

                    if current_duration > 0:
                        if current_duration < self.FRAMES_TO_TRIGGER:
                            cv2.line(frame, closest_pair[0], closest_pair[1], (0, 255, 255), 2)
                            cv2.putText(frame, f"Suspicious: {id_A} & {id_B} ({current_duration}/{self.FRAMES_TO_TRIGGER})", 
                                        (50, 100 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        else:
                            cv2.line(frame, closest_pair[0], closest_pair[1], (0, 0, 255), 5)
                            cv2.putText(frame, f"ALERT: Passing Notes! {id_A} & {id_B}", 
                                        (50, 100 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

                            current_time = datetime.now()
                            last_time = self.last_capture_time.get(pair_key)

                            if last_time is None or (current_time - last_time).total_seconds() > self.CAPTURE_COOLDOWN:
                                timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
                                img_filename = f"Violation_{id_A}_and_{id_B}_{timestamp_str}.jpg"
                                img_path = os.path.join(self.evidence_dir, img_filename)

                                cv2.imwrite(img_path, frame)

                                log_data = {
                                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "student_A": id_A,
                                    "student_B": id_B,
                                    "alert_type": "Spatial Invasion / Potential Passing Notes",
                                    "image_path": img_path
                                }

                                day_str = current_time.strftime("%Y%m%d")
                                log_file = os.path.join(self.evidence_dir, f"log_{day_str}.jsonl")
                                with open(log_file, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

                                self.last_capture_time[pair_key] = current_time
                                print(f"[EVIDENCE SAVED] Interaction-risk evidence saved: {img_filename}")

        for key in list(self.contact_timers.keys()):
            if key not in current_frame_pairs and self.contact_timers[key] > 0:
                self.contact_timers[key] -= 1

    def run_live(self):
        cap = cv2.VideoCapture(0)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if frame_width == 0: frame_width = 1280
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_height == 0: frame_height = 720
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        print("System ready; live monitoring started (press 'q' to exit)...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            results_pose = self.yolo_pose.track(frame, classes=[0], conf=0.4, tracker="botsort.yaml", persist=True, verbose=False)
            
            results_items = self.yolo_custom.predict(frame, classes=[1, 2, 3], conf=0.45, verbose=False)

            current_frame_wrists = {}
            person_count = 0 
            
            current_person_boxes = []

            if results_pose[0].boxes.id is not None and results_pose[0].keypoints is not None:
                boxes_p = results_pose[0].boxes.xyxy.cpu().numpy()
                track_ids = results_pose[0].boxes.id.cpu().numpy()
                keypoints = results_pose[0].keypoints.xy.cpu().numpy()

                for box, track_id, kpts in zip(boxes_p, track_ids, keypoints):
                    if person_count >= 5: break 
                    
                    current_person_boxes.append(box)
                    
                    x1, y1, x2, y2 = map(int, box)
                    student_id = int(track_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
                    cv2.putText(frame, f"Student ID: {student_id}", (x1, max(20, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)
                    
                    l_shoulder, r_shoulder = kpts[5], kpts[6]
                    l_wrist, r_wrist = kpts[9], kpts[10]

                    if l_wrist[0] > 0 and r_wrist[0] > 0 and l_shoulder[0] > 0 and r_shoulder[0] > 0:
                        glx, gly = int(l_wrist[0]), int(l_wrist[1])
                        grx, gry = int(r_wrist[0]), int(r_wrist[1])
                        
                        shoulder_width = self.calculate_distance(
                            (int(l_shoulder[0]), int(l_shoulder[1])), 
                            (int(r_shoulder[0]), int(r_shoulder[1]))
                        )

                        cv2.circle(frame, (glx, gly), 6, (0, 255, 0), -1)
                        cv2.circle(frame, (grx, gry), 6, (0, 255, 0), -1)

                        current_frame_wrists[student_id] = {
                            'left': (glx, gly), 
                            'right': (grx, gry), 
                            'shoulder_width': shoulder_width,
                            'box_y1': y1,             
                            'box_h': y2 - y1          
                        }
                    
                    person_count += 1

            if len(results_items[0].boxes) > 0:
                raw_boxes = results_items[0].boxes.xyxy.cpu().numpy()
                raw_classes = results_items[0].boxes.cls.cpu().numpy()
                
                valid_boxes, valid_classes = self.filter_false_positives(
                    frame_width, frame_height, raw_boxes, raw_classes, current_person_boxes
                )
                
                if len(valid_boxes) > 0:
                    self.draw_forbidden_objects(frame, valid_boxes, valid_classes)
                    self.check_and_log_objects(frame, valid_classes)

            if len(current_frame_wrists) >= 2:
                self.analyze_interactions(frame, current_frame_wrists)

            cv2.imshow("Dual-Core Anti-Cheating Monitor", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = r"D:\project zero\YOLO_Project\Model_Vault\yolov10n_4class_v1_baseline_epoch78.pt" 
    
    system = DualCoreAntiCheatingSystem(custom_model_path=MODEL_PATH)
    system.run_live()
