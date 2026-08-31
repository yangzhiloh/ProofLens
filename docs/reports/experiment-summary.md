# Experiment summary

ProofLens ran E0 through E4 on the same frozen primary manifest and split. Candidate selection
used validation data only and scored each run as `0.50 * clean AUC + 0.50 * macro robust AUC`.

| Run | Clean AUC | Macro robust AUC | Composite | Worst family | Worst condition | Unseen generator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E0 | 0.8340 | 0.8247 | 0.8293 | 0.8182 | 0.8100 | 0.9313 |
| E1 | 0.9780 | 0.9764 | 0.9772 | 0.9708 | 0.9708 | 0.9948 |
| **E2** | **0.9844** | **0.9821** | **0.9832** | **0.9768** | **0.9736** | **1.0000** |
| E3 | 0.9732 | 0.9702 | 0.9717 | 0.9648 | 0.9632 | 0.9979 |
| E4 | 0.9676 | 0.9663 | 0.9669 | 0.9628 | 0.9628 | 0.9958 |

E2 was selected before calibration. It fine-tunes the final two DINOv2 blocks and trains on
clean and randomly transformed views. These are validation-selection results, not final-test
results. The calibrated untouched-test result is reported in
[`final-robustness.md`](final-robustness.md).

Generated run directories, checkpoints, and predictions remain below ignored `artifacts/` paths.
They are not committed to Git.
