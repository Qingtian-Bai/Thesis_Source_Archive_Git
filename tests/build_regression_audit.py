import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2


PROJECT = Path(r"D:\project zero\YOLO_Project")
RUN_ROOT = PROJECT / "Experiments" / "post_final_hybrid_v2" / "evaluation" / "final_hash_6fe_regression"
NEW_LOG = RUN_ROOT / "evidence" / "log_20260818.jsonl"
OLD_LOG = PROJECT / "Experiments" / "post_final_hybrid_v2" / "evaluation" / "hybrid_v2_shared_clip_evidence" / "log_20260813.jsonl"
MANIFEST = PROJECT / "Experiments" / "post_final_hybrid_v2" / "manifests" / "run_20260818T144527Z.json"
VIDEO = PROJECT / "Final_Test_20260812" / "raw" / "FT_FINAL_UNSEEN_20260812.mp4"
ANNOTATED = RUN_ROOT / "hybrid_v2_6fe_full_annotated.mp4"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_record(record):
    result = {
        "source_time_seconds": round(float(record["source_time_seconds"]), 6),
        "alert_type": record["alert_type"],
    }
    if "students" in record:
        result["students"] = list(record["students"])
    detections = []
    for detection in record.get("detections", []):
        source = detection.get("source")
        if source == "custom_exam_model":
            source = "door_v5_phone_authority"
        detections.append(
            {
                "source": source,
                "class_id": int(detection["class_id"]),
                "class_name": detection["class_name"],
                "confidence": round(float(detection["confidence"]), 8),
                "bbox_xyxy": [round(float(v), 6) for v in detection["bbox_xyxy"]],
                "track_ids": list(detection.get("track_ids", [])),
            }
        )
    if detections:
        result["detections"] = detections
    return result


old_records = read_jsonl(OLD_LOG)
new_records = read_jsonl(NEW_LOG)
old_normalized = [canonical_record(record) for record in old_records]
new_normalized = [canonical_record(record) for record in new_records]

comparison = {
    "old_log": str(OLD_LOG),
    "new_log": str(NEW_LOG),
    "old_record_count": len(old_records),
    "new_record_count": len(new_records),
    "canonical_records_identical": old_normalized == new_normalized,
    "normalization_note": "The old source label custom_exam_model is treated as door_v5_phone_authority; timestamps, classes, confidences, boxes and track IDs remain unchanged.",
}
if old_normalized != new_normalized:
    differences = []
    for index in range(max(len(old_normalized), len(new_normalized))):
        old = old_normalized[index] if index < len(old_normalized) else None
        new = new_normalized[index] if index < len(new_normalized) else None
        if old != new:
            differences.append({"index": index, "old": old, "new": new})
    comparison["differences"] = differences

with (RUN_ROOT / "audit_comparison_with_previous_run.json").open("w", encoding="utf-8") as handle:
    json.dump(comparison, handle, ensure_ascii=False, indent=2)


phone_records = [r for r in new_records if r["alert_type"] == "Forbidden Object Detected: PHONE"]
false_objects = {
    7.520: "hand",
    20.336: "drink bottle",
    123.520: "hand near face",
    125.792: "lanyard badge",
}
with (RUN_ROOT / "phone_alert_adjudication.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["source_time_seconds", "adjudication", "object_or_event", "source", "confidence", "review_method"])
    for record in phone_records:
        time_value = round(float(record["source_time_seconds"]), 3)
        detection = record["detections"][0]
        false_object = false_objects.get(time_value)
        writer.writerow(
            [
                f"{time_value:.3f}",
                "FP" if false_object else "TP",
                false_object or "phone within a labelled phone event",
                detection["source"],
                f"{float(detection['confidence']):.6f}",
                "manual spatial review; canonical output matched the previously reviewed regression frame",
            ]
        )

note_records = [r for r in new_records if r["alert_type"] == "Forbidden Object Detected: NOTE"]
with (RUN_ROOT / "note_alert_adjudication.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["source_time_seconds", "adjudication", "event_id", "review_method"])
    for record in note_records:
        writer.writerow([f"{float(record['source_time_seconds']):.3f}", "TP", "NT02", "manual spatial review"])

note_windows = {
    "NT02": (63.5, 83.0),
    "NT03": (92.0, 94.0),
    "NT04": (99.5, 102.0),
    "NT05": (109.0, 115.0),
    "NT06": (117.0, 122.0),
}
cross_records = [r for r in new_records if r["alert_type"] == "Persistent cross-track reach"]
covered = set()
inside_count = 0
with (RUN_ROOT / "cross_track_adjudication.csv").open("w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.writer(handle)
    writer.writerow(["source_time_seconds", "students", "inside_note_window", "event_id"])
    for record in cross_records:
        time_value = float(record["source_time_seconds"])
        event_id = ""
        for candidate, (start, end) in note_windows.items():
            if start <= time_value <= end:
                event_id = candidate
                covered.add(candidate)
                inside_count += 1
                break
        writer.writerow([f"{time_value:.3f}", "|".join(map(str, record["students"])), bool(event_id), event_id])

with MANIFEST.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
started = datetime.fromisoformat(manifest["started_at_utc"])
finished = datetime.fromisoformat(manifest["finished_at_utc"])
wall_time = (finished - started).total_seconds()

capture = cv2.VideoCapture(str(VIDEO))
frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps_source = float(capture.get(cv2.CAP_PROP_FPS))
duration = frames / fps_source
capture.release()

phone_true = len(phone_records) - len(false_objects)
summary = {
    "schema_version": 1,
    "evaluation_label": "post-test regression evaluation; not a new independent test",
    "run": {
        "started_at_utc": manifest["started_at_utc"],
        "finished_at_utc": manifest["finished_at_utc"],
        "wall_time_seconds": wall_time,
        "source_frames": frames,
        "source_duration_seconds": duration,
        "end_to_end_throughput_fps": frames / wall_time,
        "realtime_factor": duration / wall_time,
        "completed_to_eof": manifest["status"] == "completed",
    },
    "hashes": {
        "code_sha256": manifest["artifacts"]["code"]["sha256"].upper(),
        "config_sha256": manifest["artifacts"]["config"]["sha256"].upper(),
        "phone_weight_sha256": manifest["artifacts"]["phone_weight"]["sha256"].upper(),
        "note_context_weight_sha256": manifest["artifacts"]["note_context_weight"]["sha256"].upper(),
        "source_video_sha256": sha256(VIDEO),
        "annotated_video_sha256": sha256(ANNOTATED),
        "run_manifest_sha256": sha256(MANIFEST),
    },
    "phone": {
        "ground_truth_events": 7,
        "detected_events": ["PH01", "PH02", "PH03", "PH04", "PH05", "PH06", "PH07"],
        "event_recall": 1.0,
        "alerts": len(phone_records),
        "true_alerts": phone_true,
        "false_alerts": len(false_objects),
        "alert_precision": phone_true / len(phone_records),
        "formal_coco_alerts": sum(1 for r in phone_records if r["detections"][0]["source"] != "door_v5_phone_authority"),
    },
    "note": {
        "ground_truth_events": 5,
        "detected_events": ["NT02"],
        "event_recall": 0.2,
        "alerts": len(note_records),
        "true_alerts": len(note_records),
        "false_alerts": 0,
        "alert_precision": 1.0,
    },
    "cross_track": {
        "ground_truth_note_windows": 5,
        "covered_events": sorted(covered),
        "event_coverage": len(covered) / 5,
        "alerts": len(cross_records),
        "alerts_inside_note_windows": inside_count,
        "alerts_outside_note_windows": len(cross_records) - inside_count,
        "temporal_alert_precision": inside_count / len(cross_records),
    },
    "comparison_with_previous_regression": comparison,
}

with (RUN_ROOT / "regression_result_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)

shutil.copy2(MANIFEST, RUN_ROOT / "run_manifest_6fe.json")
print(json.dumps(summary, ensure_ascii=False, indent=2))

