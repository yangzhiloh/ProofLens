# Error analysis

At the validation-selected threshold, the untouched test set produced five false positives and
four false negatives. The aggregate condition error rates below describe the canonical
evaluation rows; they do not expose or redistribute source images.

- The highest observed transformed-condition error rate was 10% for Gaussian blur sigma 2.0,
  color jitter, JPEG quality 30, JPEG quality 70, Gaussian noise sigma 0.02, and JPEG quality 50.
- Clean rows produced 16 errors among 162 rows (9.88%). The larger clean count includes the
  generator-test partition.
- Resize 0.5, blur sigma 1.0, blur sigma 0.5, JPEG quality 90, and resize 0.25 each produced a 9%
  error rate.
- Noise sigma 0.10 and 0.05 each produced an 8% error rate; center crop 80% produced 7%.

Among the 162 clean rows, authentic images accounted for 9 errors among 81 examples. The largest
synthetic family with errors was fast-sdxl with 3 errors among 31 examples. Several other
generator families contained only two to four clean test examples, so their individual rates are
too uncertain for comparative claims.

The analysis indicates that failures are not isolated to a single post-processing family.
Content domain, generator coverage, and compound edits remain important unmeasured sources of
shift. ProofLens scores therefore require contextual human review and must not be treated as
forensic proof.
