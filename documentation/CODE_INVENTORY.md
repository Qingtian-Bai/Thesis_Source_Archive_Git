# Maintained code inventory

This release intentionally keeps the runnable source small and explicit.

## Production runtime

- `code/run_hybrid_v2.py` is the supported launcher. It resolves packaged paths, validates required files, applies the frozen configuration and records the run manifest.
- `code/hybrid_monitor.py` contains the maintained monitoring implementation, including camera and video input, phone and note processing, pose/tracking interaction logic, evidence capture and failure handling.
- `config/hybrid_config.yaml` is the authoritative runtime configuration.

## Validation code

- `tests/test_hybrid_logic.py` covers deterministic core logic.
- `tests/test_fault_tolerance.py` covers six simulated failure and boundary paths.
- `tests/run_fault_tests.ps1` is the supported fault-suite command wrapper.

The fault suite now creates transient working files in an operating-system temporary directory and removes them automatically. Only the text and JSON result summaries are retained in the release.

## Removed obsolete material

The 26 August 2026 cleanup removed two non-runtime artefacts:

- `evaluation/build_regression_audit.py`, a one-off script with historical absolute paths that generated the superseded automatic-window summary. Its retained outputs remain under `evaluation/automatic_window_matching/` for provenance, while final reporting continues to use the manual-review records.
- `tests/evidence/fault_tolerance/synthetic_work/`, generated test scratch data containing temporary videos, copied weights and a copied monitor module. The fault suite recreates equivalent scratch data when needed and now removes it after each run.

No production method, model path, threshold, configuration value, weight or evaluation decision was removed or changed during this cleanup.
