# Final robustness report

The latest final E2-on-SID checkpoint was calibrated using validation predictions and then
evaluated once on untouched test partitions. The validation-selected decision threshold was
`0.905192`. The primary held-out test partition contains 1,006 images: 953 authentic and 53
AI-generated.

## Untouched test result

| Measure | Value |
| --- | ---: |
| Clean ROC AUC | 0.9974 |
| Macro robust ROC AUC | 0.9971 |
| Composite score | 0.9973 |
| Worst-condition ROC AUC | 0.9966 |
| Unseen-generator ROC AUC | 0.9638 |
| Accuracy at selected threshold | 0.9841 |
| Precision at selected threshold | 0.7846 |
| Recall at selected threshold | 0.9623 |
| F1 at selected threshold | 0.8644 |
| False positives | 14 |
| False negatives | 2 |

## Canonical robustness conditions

| Condition | Family | ROC AUC | Errors |
| --- | --- | ---: | ---: |
| JPEG quality 30 | JPEG | 0.9966 | 21 / 1,006 |
| Center crop 80% | Crop | 0.9970 | 18 / 1,006 |
| JPEG quality 50 | JPEG | 0.9968 | 18 / 1,006 |
| Resize 0.25x | Resize | 0.9970 | 18 / 1,006 |
| Resize 0.5x | Resize | 0.9970 | 18 / 1,006 |
| Blur sigma 0.5 | Blur | 0.9973 | 17 / 1,006 |
| JPEG quality 90 | JPEG | 0.9974 | 17 / 1,006 |
| Blur sigma 1.0 | Blur | 0.9971 | 16 / 1,006 |
| Blur sigma 2.0 | Blur | 0.9971 | 16 / 1,006 |
| Color jitter 20% | Color | 0.9973 | 16 / 1,006 |
| JPEG quality 70 | JPEG | 0.9972 | 15 / 1,006 |
| Noise sigma 0.02 | Noise | 0.9978 | 14 / 1,006 |
| Noise sigma 0.10 | Noise | 0.9970 | 14 / 1,006 |
| Noise sigma 0.05 | Noise | 0.9973 | 11 / 1,006 |

## Supplemental redistribution stress

| Condition | ROC AUC | Mean absolute probability shift |
| --- | ---: | ---: |
| WebP quality 80 | 0.9971 | 0.0108 |
| WebP quality 50 | 0.9967 | 0.0175 |
| Screenshot at 1440 px | 0.9965 | 0.0220 |
| Screenshot at 1080 px | 0.9965 | 0.0221 |

The parity-gated ONNX export compared 32 validation or test images. It passed the `1e-4`
tolerance with maximum absolute logit difference `0.00002551` and maximum absolute probability
difference `0.00000131`.

Machine-readable evidence for this final report is generated at
`artifacts/reports/sid-comparison-final/metrics.json`,
`artifacts/reports/sid-comparison-stress/stress-metrics.json`, and
`artifacts/export/export_report.json`.
The public RC1 release currently contains an earlier model bundle, so these latest metrics must
not be attributed to the downloadable RC1 model until the matching E2-on-SID artifacts are
published.
Those generated files and the 346,632,491-byte ONNX model are intentionally ignored by Git and
must be distributed through separately reviewed release storage.
