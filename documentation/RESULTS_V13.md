# Paper/note v1.3 results

## Model-only test comparison at 960 input

The test images were not augmented. However, there are only six note
instances, so note metrics are highly uncertain.

| Metric | Frozen v5 | Paper/note candidate |
|---|---:|---:|
| Overall mAP50 | 0.812 | 0.858 |
| Overall mAP50-95 | 0.531 | 0.628 |
| Phone mAP50 | 0.912 | 0.955 |
| Paper mAP50 | 0.982 | 0.988 |
| Note mAP50 | 0.290 | 0.405 |
| Door mAP50 | 0.883 | 0.949 |

At the note proposal threshold 0.30, note recall rose from 0.167 to 0.500.
Raw proposal precision was 0.500, which is why proposals are not treated as
violations without temporal hand-transfer evidence.

## Runtime regression

- Frozen v5 phone/door success version remains unchanged.
- Normal large-paper smoke range, 700 frames: zero note evidence.
- Front/back transfer range, 602 frames: two note-transfer evidence images.
- Full fixed video, 5052 frames: three note-transfer evidence images.
- Every final note evidence box was visually reviewed and lay on the small
  hand-held paper involved in the transfer, rather than the full exam sheet.

## Evidence boundary

The fixed long video has historical frames represented in the training data.
It is a regression and visualization source, not an independent thesis test.
An unseen video with interval-level ground truth is still required for final
precision, recall and false-alarm reporting.
