# Version-control ledger

## Purpose

This ledger connects the project's original snapshot-based version control to the Git migration completed on 24 August 2026. Each row names an already existing source snapshot, the Git commit created from it and the annotated tag used for assessment and recovery.

The Git commit dates record the migration. Original artefact paths, last-write timestamps, byte sizes and SHA-256 hashes are stored in `documentation/SNAPSHOT_PROVENANCE.json` at the corresponding tag.

## Preserved history

| Order | Original controlled snapshot | Original source location | Git commit | Annotated tag |
|---:|---|---|---|---|
| 1 | Legacy v4 | `Baselines/legacy_baseline.py` and `Baselines/LEGACY_BASELINE_MANIFEST.md` | `7f37f09fda7f9e268c66c3dc74a5916605200caf` | `legacy-v4` |
| 2 | Door v5 | `Baselines/baseline_20260729_v5_door_success/` | `a195004436024bddfb8440e43e3672f05d1c42cf` | `door-v5` |
| 3 | Note v6 | `Baselines/baseline_20260730_v6_paper_note_success/` | `1b798b1180d78880e2f762b31daa3ef9e42ac9ff` | `note-v6` |
| 4 | Final candidate v1 | `Experiments/final_candidate_v1/` | `079efc77b0543f791a7e90ee7221d7ca52626222` | `final-candidate-v1` |
| 5 | 6FE329 original | `Experiments/post_final_hybrid_v2/` | `6e833ef4d535fd40ae0120ec92a884ef25e7ef5b` | `6fe329-original` |
| 6 | 6FE329 portability reissue | `Thesis_Frozen_Release_6FE329_20260818/` | `7e1688a6ca9e7aa54f21573b53a6598bed40c436` | `thesis-submission-20260823` |

## Verification procedure

1. Run `git fsck --full` to check the object database.
2. Run `git log --graph --decorate --oneline --all` to inspect the migrated history.
3. Run `git diff <older-tag>..<newer-tag>` to inspect version changes.
4. At any tag, inspect `documentation/SNAPSHOT_PROVENANCE.json` and compare its hashes with the original archived source artefacts.
5. Run `git bundle verify` against the submitted bundle and clone it into a clean directory to test recovery.

## Interpretation

The project used versioned snapshots during development and Git from the archive-migration point onward. The repository supports reproducible comparison and recovery of the preserved versions. It does not claim that the earlier development actions were originally recorded as Git commits, branches, pull requests or issues.
