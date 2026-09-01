# Publication checklist

Date: 2026-08-31

## Automated and repository-owned checks

- [x] Public repository destination: `https://github.com/yangzhiloh/ProofLens`
- [x] E2 checkpoint and CPU ONNX model stored outside Git history
- [x] Calibration, artifact manifest, selection record, parity report, and SHA-256 checksums
      included with the model release
- [x] Public release destination:
      `https://github.com/yangzhiloh/ProofLens/releases/tag/prooflens-e2-rc1`
- [ ] Matching E2-on-SID checkpoint, ONNX export, calibration, and provenance bundle published
      for the latest final evaluation report
- [x] ONNX asset digest matches local SHA-256
      `5fa8e6dd804d8f9c9a2908262048d7ba0d7924e37dafa90906b9c3c7db60b84c`
- [x] Checkpoint asset digest matches local SHA-256
      `1dbfac985e20263ccf13db02762d1b6cc075ba5945721a72fc7cdbaabc3df56c`
- [x] CPU inference and the programmatic clean/transformed comparison utility passed
- [x] Clean-checkout locked installation, miniature reproduction, and release scan passed
- [x] Full local suite passed: 454 tests, one optional OpenVINO skip, Ruff, and release check
- [x] GitHub Windows/Linux Python 3.11/3.12 CI passed for Tasks 8 and 9
- [x] README, model card, results, error analysis, acceptance report, Devpost copy, and video
      script contain consistent measured values and limitations
- [x] Dataset and pretrained-model terms are recorded separately from the MIT project licence

## Human-only publication actions

- [x] Record the 2-to-4-minute demonstration using `docs/video-script.md`
- [x] Upload the video to an approved public destination
- [x] Add and open the public video URL in `docs/devpost-draft.md`:
      `https://youtu.be/r-eNOps1qo0`
- [ ] Review the final text, screenshots, attribution, and platform-required fields
- [ ] Submit or publish through the owner's authenticated account

The unchecked items require a person's voice/screen recording, account access, and publication
approval. They must not be represented as complete until the owner performs them.
