# Reconstruction notice

## Status of this repository

This Git repository was created on **24 August 2026** as a submission-time archival reconstruction. The commits were assembled from already existing snapshot directories. They do **not** represent commits made during the original development process, and their Git author dates must not be interpreted as development dates.

Every reconstructed snapshot commit uses this statement:

> Reconstructed on 24 August 2026 from an archived source snapshot. This commit does not represent contemporaneous development-time version control.

No commit date has been backdated. The original filesystem modification times and SHA-256 values of the selected source artefacts are recorded in `documentation/SNAPSHOT_PROVENANCE.json` at each tag.

## Source mapping

| Reconstructed version | Existing source snapshot | Git tag |
|---|---|---|
| Legacy v4 | `Baselines/legacy_baseline.py` and `Baselines/LEGACY_BASELINE_MANIFEST.md` | `legacy-v4` |
| Door v5 | `Baselines/baseline_20260729_v5_door_success/` | `door-v5` |
| Note v6 | `Baselines/baseline_20260730_v6_paper_note_success/` | `note-v6` |
| Final candidate v1 | `Experiments/final_candidate_v1/` | `final-candidate-v1` |
| 6FE329 original | `Experiments/post_final_hybrid_v2/` | `6fe329-original` |
| 6FE329 portability reissue | `Thesis_Frozen_Release_6FE329_20260818/` | `thesis-submission-20260823` |

The final tag name refers to the submission artefact dated 23 August 2026; the tag itself was created during this reconstruction on 24 August 2026.

## Repository scope

The repository contains only source code, configuration, tests, technical documentation, dependency information and integrity manifests. Snapshot files may be placed under a normalised path; `documentation/SNAPSHOT_PROVENANCE.json` records both the original path and the path used in Git.

The following materials are intentionally excluded:

- participant or examination recordings;
- consent forms and records containing signatures or personal information;
- model-weight binaries;
- large annotated or raw output videos;
- evidence screenshots and runtime evidence logs;
- Python caches, editor files and temporary outputs.

The final frozen package's `FROZEN_MANIFEST.json` and `SHA256SUMS.txt` are retained as documentation even though some binaries listed by those files are intentionally not copied into this Git repository.

## Commit identity

The repository uses the local archival identity `Thesis Archive Reconstruction <thesis-archive@local.invalid>`. The `.invalid` address is deliberately non-routable. This prevents an unrelated global Git identity configured on the workstation from being attributed to the reconstruction.

## Interpretation rule

This repository improves traceability of existing snapshots. It does not repair the absence of development-time Git history, and it must not be presented as evidence that version control was used throughout development.
