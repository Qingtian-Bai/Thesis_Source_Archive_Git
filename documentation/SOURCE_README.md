# Frozen five-class success baseline — 2026-07-29

This directory freezes the most successful version reached before beginning
the paper/note redesign.

## Identity

- Model: `weights/yolov10n_5class_door_v1_frozen.pt`
- Classes: `student`, `phone`, `paper`, `note`, `door`
- Runtime custom-object confidence: `0.50`
- Door display confidence: `0.65`
- Door is neutral context: it never creates an alert and never forcibly
  suppresses a phone.
- Starting checkpoint: `yolov10n_4class_v1_baseline_epoch78.pt`
- Training device: NVIDIA GeForce RTX 5060 Laptop GPU

## Contents

- `weights/`: frozen model weight
- `code/monitor.py`: exact application code at freeze time
- `code/run_frozen_v5_door.py`: self-contained frozen launcher
- `config/`: dataset descriptor, tracker configuration and runtime parameters
- `metrics/model_metrics.txt`: validation, test and threshold-calibration results
- `training/`: exact two-stage training script
- `checksums.sha256`: integrity hashes for every frozen file except itself

The image dataset is not duplicated into this snapshot. Its descriptor is
frozen in `config/dataset.yaml`; the source dataset remains
`ultimate_exam_dataset_camera_v5_door`.

## Run

Double-click `launch_frozen_v5_door.cmd`, or run:

```powershell
& "D:\python 3.13.7\python.exe" `
  "Baselines\baseline_20260729_v5_door_success\code\run_frozen_v5_door.py" `
  --source 0
```

Do not edit files in this directory. Create a new experiment directory for
all paper/note changes.
