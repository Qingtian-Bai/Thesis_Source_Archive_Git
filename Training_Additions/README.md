# Training and dataset reconstruction materials

This directory contains the non-personal scripts used to build and train the
Door-v5, Paper-note v6 and Phone-hardcases v7 dataset/model variants described
in the dissertation.

The image datasets, participant recordings, reviewed hard-case frames and model
weight binaries are intentionally not included in this public repository. Some
source material contains images of people and remains under the project's
private data-handling controls. The scripts are published to document the
processing and training procedure, not to redistribute restricted inputs.

The scripts preserve the original project-relative layout:

- dataset directories are expected at the repository root;
- generated YOLO runs are written under `runs/detect/`;
- reviewed hard-case inputs are expected under `Training_Additions/`;
- starting model weights must be supplied separately.

Before running a script, review its required paths and provide the relevant
lawfully held inputs. The public dataset descriptors under
`dataset_descriptors/` use placeholders instead of the original workstation
paths.

`build_camera_v5_door_dataset.py` also identifies its external DoorDetect source
in the module documentation. Users remain responsible for complying with the
licence and terms of every external dataset.

