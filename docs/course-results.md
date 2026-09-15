# Archived course results

The original VUB group report recorded the following MiniImageNet results. They are preserved
for academic traceability but are **not reproduced by the public software demo**, because the
course dataset snapshot and trained checkpoints are not redistributed.

| Model | Sampling | 1-shot | 4-shot | 8-shot | 16-shot |
|---|---|---:|---:|---:|---:|
| Contrastive Siamese | Random | .10 (.20) | .30 (.20) | .30 (.20) | .10 (.20) |
| Contrastive Siamese | Hard | .25 (.50) | .25 (.44) | .35 (.34) | .50 (.39) |
| Contrastive Siamese | Tuned | .55 (.72) | .60 (.61) | .70 (.51) | .85 (.57) |
| Triplet Siamese | Random | .55 (.71) | .65 (.55) | .80 (.60) | .70 (.60) |
| Triplet Siamese | Semi-hard | .70 (.82) | .75 (.68) | .80 (.65) | .70 (.61) |
| Triplet Siamese | Tuned | .60 (.77) | .85 (.73) | .90 (.76) | .70 (.62) |
| ResNet18 Places365 | Contrastive | .75 (.85) | .70 (.67) | .70 (.71) | .60 (.58) |
| ResNet18 Places365 | Triplet | .85 (.92) | .75 (.76) | .85 (.83) | .95 (.77) |

Each entry is Top-1 accuracy with the report's mAP value in parentheses. The evaluation used a
5-way setup with 20 query images. This small evaluation produces coarse accuracy increments,
and the original report did not provide uncertainty intervals or repeated-run statistics.
Therefore, comparisons should be treated as descriptive course results rather than definitive
benchmarks.

Top-5 accuracy is intentionally omitted: in a 5-way task, Top-5 is always 100% and provides no
discriminative information.
