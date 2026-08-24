# Frozen-release validation commands

Run these commands from the release root with the recorded Python environment.

## UT-01: spatial evidence cooldown logic

```powershell
python tests/test_hybrid_logic.py
```

The retained output is in `tests/evidence/unit_test_output.txt`.

## Full-video clean-copy smoke test

Copy the entire release folder to a new directory, enter that copied directory, and run:

```powershell
python code/run_hybrid_v2.py --source input/FT_FINAL_UNSEEN_20260812.mp4 --no-display --exit-on-eof --save-output runtime_outputs/smoke_full_annotated.mp4 --evidence-dir runtime_outputs/evidence
```

The retained command log and summary are in `tests/evidence/`. The smoke test checks that the packaged paths, weights, configuration and end-to-end code path run from a clean copied location. It reuses the regression video and is not an independent generalisation test.
