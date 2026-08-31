# Final robustness report

The selected E2 checkpoint was calibrated using validation predictions and then evaluated once
on the untouched test partitions. Temperature was `1.89234459400177`; the validation-selected
decision threshold was `0.37237685918807983`.

## Untouched test result

| Measure | Value |
| --- | ---: |
| Clean ROC AUC | 0.9788 |
| Macro robust ROC AUC | 0.9783 |
| Composite score | 0.9785 |
| Worst-condition ROC AUC | 0.9728 |
| Worst-family ROC AUC | 0.9745 |
| Unseen-generator ROC AUC | 0.9657 |
| Accuracy at selected threshold | 0.9100 |
| Precision at selected threshold | 0.9020 |
| Recall at selected threshold | 0.9200 |
| F1 at selected threshold | 0.9109 |
| False positives | 5 |
| False negatives | 4 |

## Supplemental redistribution stress

| Condition | ROC AUC | Mean absolute probability shift |
| --- | ---: | ---: |
| WebP quality 80 | 0.9740 | 0.005910 |
| WebP quality 50 | 0.9744 | 0.007005 |
| Screenshot at 1440 px | 0.9724 | 0.010530 |
| Screenshot at 1080 px | 0.9732 | 0.009782 |

The parity-gated ONNX export compared 32 validation or test images. It passed the `1e-4`
tolerance with maximum absolute logit difference `0.0000705718994140625` and maximum absolute
probability difference `0.0000025704503059387207`.

Machine-readable local evidence is generated at `artifacts/reports/final/metrics.json`,
`artifacts/reports/stress/stress-metrics.json`, and `artifacts/export/export_report.json`.
Those generated files and the 346,632,491-byte ONNX model are intentionally ignored by Git and
must be distributed through separately reviewed release storage.
