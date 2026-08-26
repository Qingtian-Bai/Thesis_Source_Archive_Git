# Reconstruction notice

## Status of this repository

This Git repository was created on **24 August 2026** by migrating the project's existing version-controlled snapshots into Git. During development, version changes had been managed through named and dated snapshot directories rather than Git commits. The migrated commits therefore preserve real development states, but their Git author dates record the migration and must not be interpreted as the dates on which the underlying development work occurred.

Every reconstructed snapshot commit uses this statement:

> Reconstructed on 24 August 2026 from an archived source snapshot. This commit does not represent contemporaneous development-time version control.

No commit date has been backdated. The original filesystem modification times and SHA-256 values of the selected source artefacts are recorded in `documentation/SNAPSHOT_PROVENANCE.json` at each tag.

## English-only delivery normalization

On 26 August 2026, each preserved tree was normalized for an English-only assessor delivery. Chinese-language comments, console messages, explanatory prose, and archived filename references were translated or omitted. The normalization did not change detector logic, thresholds, configuration values, model weights, or evaluation decisions. Because source bytes changed, the normalized commits and tags have new Git object IDs. `documentation/ENGLISH_NORMALIZATION_LEDGER.md` maps the earlier migration commits to the English-only commits.

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

This repository provides standard Git inspection, comparison, tagging and recovery for the existing snapshot-based version history. It is evidence of the preserved versions and their differences, while `documentation/SNAPSHOT_PROVENANCE.json` is the evidence for their original paths, timestamps and hashes. It must not be presented as evidence that Git itself was used throughout development.
