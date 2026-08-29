# Third-party notices

ProofLens project code is distributed under the MIT License in `LICENSE`. Dependencies installed
from `pyproject.toml`, pretrained model material, and datasets retain their own terms. This file
records the declarations used by the repository. It does not replace the terms distributed by
the corresponding provider.

## DINOv2

- Component: DINOv2 base backbone, configured as `facebook/dinov2-base`
- Licence declaration: Apache-2.0
- Use in ProofLens: pretrained backbone loaded through Transformers

The installed Transformers package metadata also declares the Apache 2.0 License. Before
redistributing any downloaded model files, retain their accompanying licence and model-card
notices.

## SID-Set

- Dataset identifier: `saberzl/SID_Set`
- Pinned revision: `c1674903d858c78e04809c1c6f2703627ac1a621`
- Licence declaration: CC-BY-4.0
- Attribution requirement recorded by the project: source-material attribution is required

The acquisition command stores the pinned revision and licence identifier in local metadata.
Dataset contents are not distributed by this repository.

## CIFAKE

- Use in ProofLens: separate low-resolution stress test only
- Licence declaration: MIT
- Attribution requirement recorded by the project: retain the required dataset citations

CIFAKE is excluded from primary training. Dataset contents are not distributed by this
repository.

## WildFake

- Official repository recorded by the project:
  `https://github.com/hy-zpg/AIGC-Image-Detection-Dataset`
- Acquisition source recorded by the project:
  `https://modelscope.cn/datasets/hy2628982280/WildFake/summary`
- Licence declaration: REQUIRES-VERIFICATION
- Attribution requirement recorded by the project: retain the official paper and acquisition
  source attribution

WildFake acquisition is manual. Verify the terms attached to the specific obtained copy before
training, publication, or redistribution. ProofLens does not grant permission to redistribute
WildFake.
