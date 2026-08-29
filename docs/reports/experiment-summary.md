# Day 3 experiment summary

This execution record covers the network-free miniature workflow used to verify
the E3 and E4 implementation paths. It is a software/integration smoke result,
not a claim about primary-dataset detector quality.

| Experiment | Configuration | Checkpoint | Clean AUC | Macro robust AUC | Composite | Worst family | Worst condition | Unseen-generator AUC | Train s | Median CPU ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E3 | consistency losses | `artifacts/day3-e3-release/run/checkpoints/best.pt` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2.063 | 0.588 |
| E4 | consistency + hard mining enabled | `artifacts/day3-e4-release/run/checkpoints/best.pt` | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 2.225 | 0.648 |

Both runs used seed 17, fixed fixture data, CPU float32, one epoch, all six
canonical transformation families, and the same grouped split. The E4 run
records `hard_mining: true` and `candidate_count: 3` in its generated
configuration and records selected candidate conditions in `history.jsonl`.

Commands exercised:

```text
python scripts/reproduce_small.py --output artifacts/day3-e3-final --experiment e3
python scripts/reproduce_small.py --output artifacts/day3-e4-final --experiment e4
```

Generated evidence includes `history.jsonl`, `run_metadata.json`, checkpoint
files, prediction Parquet files, metrics JSON, robustness Markdown, and the
AUC plot under each output directory. Primary E3/E4 training remains dependent
on a user-provided approved manifest and the committed DINOv2 weights.
