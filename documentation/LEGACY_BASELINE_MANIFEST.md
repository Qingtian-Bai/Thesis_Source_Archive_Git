# Legacy baseline definition

`legacy_baseline.py` is the verbatim executable extraction of lines 1–324 from the commented legacy block at the beginning of `monitor.py`, frozen on 2026-07-26.

## Scope

- It intentionally preserves the original algorithm and runtime assumptions.
- It uses `yolov8n-pose.pt`, `Model_Vault/yolov10n_4class_v1_baseline_epoch78.pt`, and the original `botsort.yaml` tracker reference.
- It processes camera `0` only, mirrors the camera frame, accepts at most five tracked students, applies an 80-pixel person-association margin to all objects, and uses wrist-distance interaction detection.
- It has no COCO auxiliary detector, no occlusion layer logic, no replay/save CLI, and no automatic runtime recovery.

## Integrity

SHA-256 at extraction:

```text
42E277BFBA92B87F9CDF97942027CD58150C99E7BE2E5CD3D25B6605787A9C5F  legacy_baseline.py
```

Treat this file as read-only. Future experiment code must be created separately; do not edit this baseline to improve it.

## Comparison rule

For a formal baseline-versus-improved comparison, both versions must process the same frozen input video. The baseline source remains untouched; a separate replay wrapper can be created later solely to provide the same input video to both versions without changing their detection logic.
