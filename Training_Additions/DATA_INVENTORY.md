# Dataset version inventory

This inventory records dataset structure and annotation counts without
redistributing images, participant recordings or personal data.

Class order for every version:

`student`, `phone`, `paper`, `note`, `door`

Instance counts are shown in that class order.

| Version | Split | Images | Label files | Instances: student / phone / paper / note / door |
|---|---:|---:|---:|---:|
| Door-v5 | train | 1,888 | 1,888 | 234 / 1,667 / 100 / 97 / 480 |
| Door-v5 | validation | 320 | 320 | 56 / 316 / 24 / 3 / 40 |
| Door-v5 | test | 322 | 322 | 66 / 318 / 26 / 6 / 40 |
| Paper-note v6 | train | 2,126 | 2,126 | 234 / 1,887 / 300 / 291 / 480 |
| Paper-note v6 | validation | 320 | 320 | 56 / 316 / 24 / 3 / 40 |
| Paper-note v6 | test | 322 | 322 | 66 / 318 / 26 / 6 / 40 |
| Phone-hardcases v7 | train | 2,226 | 2,226 | 234 / 1,980 / 300 / 291 / 480 |
| Phone-hardcases v7 | validation | 320 | 320 | 56 / 316 / 24 / 3 / 40 |
| Phone-hardcases v7 | test | 322 | 322 | 66 / 318 / 26 / 6 / 40 |

## Version relationship

- **Door-v5** adds an explicit door class and filters door examples to reduce
  dark-background false positives.
- **Paper-note v6** starts from the frozen Door-v5 dataset and adds only
  training-set paper/note augmentation. Validation and test splits remain
  unchanged.
- **Phone-hardcases v7** starts from the frozen Paper-note v6 dataset and adds
  100 manually reviewed phone hard-case training frames. Validation and test
  splits remain unchanged.

The original image data is excluded from GitHub. The counts above and the
public build scripts provide the non-personal inventory and reconstruction
record promised in the dissertation.

