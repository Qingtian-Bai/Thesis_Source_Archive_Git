# Paper/note experiment v1.3

This experiment branches from the frozen
`baseline_20260729_v5_door_success`. It does not modify the frozen baseline.

## Data

- Base: `ultimate_exam_dataset_camera_v5_door`
- Experiment: `ultimate_exam_dataset_camera_v6_paper_note`
- Every existing v5 image and label is retained.
- Existing note training boxes: 97
- Note boxes after train-only targeted augmentation: 291
- Existing paper training boxes: 100
- Paper boxes after train-only targeted augmentation: 300
- Validation and test were not augmented.

## Runtime policy

- `paper`: neutral grey context; never evidence.
- `note`: proposal threshold 0.30 at input size 960.
- A proposal becomes `NOTE HELD (review)` only after hand association and at
  least four temporally matched detections. This orange state is not evidence.
- Red `WARNING: NOTE TRANSFER` evidence additionally requires the same paper
  track to associate with two different students for at least two frames each.
- Candidate size is limited relative to tracked shoulder width, rather than by
  an absolute pixel threshold.
- A strong overlapping `paper` result can keep an ambiguous region neutral.
- The successful phone/door pipeline remains at custom confidence 0.50.

## Test limitation

The untouched test split contains only six note instances. Improvements are
promising but statistically weak. The fixed long video is also represented in
some historical training frames, so it is suitable for regression
visualization, not final thesis accuracy.

## Run

Double-click `launch_frozen_note_v6.cmd`.

## Fixed-video v1.3 regression

- Full frames processed: 5052 / 5052
- Normal large-paper smoke range (first 700 frames): 0 note evidence images
- Front/back note range (602 frames): 2 note-transfer evidence images
- Full fixed video: 3 note-transfer evidence images
- Processing time: about 250.6 seconds for 221.3 seconds of video

The final comparison video is
`evaluation/frozen_v5_vs_paper_note_v13_full.mp4`.
