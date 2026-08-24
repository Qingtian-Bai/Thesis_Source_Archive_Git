# 6FE329 frozen technical release

This package contains the code, configuration, model weights, regression input and retained evaluation evidence for the 6FE329 post-test engineering regression described in the dissertation. The regression result is not an independent generalisation test.

## Environment

The recorded environment used Python 3.13.7 on Windows 11, PyTorch 2.8.0 with CUDA 12.8, torchvision 0.23.0, Ultralytics 8.4.105, OpenCV 4.13.0.92, NumPy 2.4.4 and PyYAML 6.0.3. See `requirements.txt`.

## Run the packaged video

From the release root:

```powershell
python code/run_hybrid_v2.py --source input/FT_FINAL_UNSEEN_20260812.mp4 --no-display --exit-on-eof --save-output runtime_outputs/full_annotated.mp4 --evidence-dir runtime_outputs/evidence
```

All packaged resources are resolved relative to the release root. Runtime manifests, temporary tracker configuration and application outputs are written under `runtime_outputs/`.

## Validation evidence

The unit-test script, commands and retained outputs are under `tests/`. `FROZEN_MANIFEST.json` and `SHA256SUMS.txt` provide the frozen inventory and integrity hashes. Ethics and participant-consent status is stated in `documentation/ETHICS_AND_CONSENT_STATUS.md`.
