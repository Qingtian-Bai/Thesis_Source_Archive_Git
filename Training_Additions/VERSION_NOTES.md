# Training version notes

| Version | Starting point | Main change | Dissertation role |
|---|---|---|---|
| Door-v5 | Frozen four-class baseline | Adds `door` as the fifth class | Final authoritative phone-evidence branch |
| Paper-note v6 | Frozen Door-v5 state | Adds paper/note training material and mild augmentation | Final paper/note contextual branch |
| Phone-hardcases v7 | Frozen Paper-note v6 state | Adds 100 reviewed hard-case phone frames | Development comparison; not retained as the final phone branch |

Only training inputs were added in the v6 and v7 steps. The retained validation
and test partitions were not augmented. Full image data and model binaries are
not distributed in this public repository.

