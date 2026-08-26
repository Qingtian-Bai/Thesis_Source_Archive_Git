# Thesis source version-control archive

This is the Git-controlled source archive for the examination-monitoring dissertation project. It preserves six genuine development and release snapshots as a linear, inspectable history.

## Version-control status

During development, versions were controlled through named and dated snapshot directories, including the Legacy v4, Door v5, Note v6, Final candidate v1 and 6FE329 snapshots. On 24 August 2026, those existing snapshots were migrated into Git so that their differences, provenance and final release state could be inspected with standard version-control tools.

The Git commits record the migration operation. Their 24 August commit dates are not substitutes for the original file timestamps. Original paths, timestamps and SHA-256 values are retained in `documentation/SNAPSHOT_PROVENANCE.json` at every version tag. See `RECONSTRUCTION_NOTICE.md` for the interpretation boundary and `documentation/VERSION_CONTROL_LEDGER.md` for the tag-to-source mapping.

On 26 August 2026, an assessor-facing English-only history was generated from those preserved states. This language normalization translated or omitted comments, console messages, explanatory prose, and archived filename references without changing detector logic, thresholds, configuration values, model weights, or evaluation decisions. The resulting commit mapping is recorded in `documentation/ENGLISH_NORMALIZATION_LEDGER.md`; hashes changed because textual bytes changed.

## Version tags

| Tag | Preserved version |
|---|---|
| `legacy-v4` | Legacy v4 baseline |
| `door-v5` | Door v5 phone-specialist version |
| `note-v6` | Note v6 paper/note-context version |
| `final-candidate-v1` | Final candidate v1 |
| `6fe329-original` | Original 6FE329 engineering version |
| `thesis-submission-20260823` | Portable frozen release used for submission |
| `thesis-git-archive-20260824` | Verified Git delivery state |
| `thesis-english-delivery-20260826` | English-only assessor delivery |

## Inspect and verify

```powershell
git status
git log --graph --decorate --oneline --all
git tag -n99
git diff door-v5..note-v6
git show thesis-submission-20260823:documentation/SNAPSHOT_PROVENANCE.json
git fsck --full
git bundle verify ..\Thesis_Source_Archive_Git_Delivery_20260824\Thesis_Source_Archive_Git.bundle
```

To restore the complete repository from the submitted bundle:

```powershell
git clone Thesis_Source_Archive_Git.bundle Thesis_Source_Archive_Git_restored
cd Thesis_Source_Archive_Git_restored
git log --graph --decorate --oneline --all
```

## Repository scope

The repository contains code, configuration, tests, technical documentation, dependency information and integrity manifests. It intentionally excludes participant recordings, personal consent records, model-weight binaries and large derived videos. Those exclusions are documented rather than silently omitted.

The runnable frozen technical package is distributed separately because it contains model weights and other non-Git artefacts. Its `FROZEN_MANIFEST.json` and `SHA256SUMS.txt` remain in this repository to connect the source history to the frozen release inventory.

## Final runtime environment

The recorded environment used Python 3.13.7 on Windows 11, PyTorch 2.8.0 with CUDA 12.8, torchvision 0.23.0, Ultralytics 8.4.105, OpenCV 4.13.0.92, NumPy 2.4.4 and PyYAML 6.0.3. See `requirements.txt` and `tests/README.md`.
