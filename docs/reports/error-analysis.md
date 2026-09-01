# Error analysis

At the validation-selected threshold `0.905192`, the untouched primary test set produced 14
false positives and 2 false negatives. The aggregate condition error rates below describe the
canonical evaluation rows; they do not expose or redistribute source images.

- The highest observed transformed-condition error count was 21 of 1,006 images for JPEG quality
  30, an error rate of 2.09%.
- The clean primary test partition produced 16 errors: 14 authentic images were classified as
  AI-generated and 2 AI-generated images were classified as authentic.
- The false negatives came from `flux-pro-v1.1` (1 of 3 examples) and
  `stable-diffusion-v35-medium` (1 of 1 example). These generator-specific subsets are too
  small for comparative claims.
- The lowest observed condition error count was 11 of 1,006 images for Gaussian noise sigma 0.05.

On the unseen-generator partition, ranking remained strong at 0.9638 ROC AUC, but the fixed
threshold produced 3,656 false negatives and 19 false positives. This is a calibration-transfer
failure rather than a collapse in ranking. The model still tends to rank generated images above
authentic images, but the probability distribution shifts downward on the unseen domain.

The analysis indicates that failures are not isolated to a single post-processing family.
Content domain, generator coverage, and compound edits remain important unmeasured sources of
shift. ProofLens scores therefore require contextual human review and must not be treated as
forensic proof.
