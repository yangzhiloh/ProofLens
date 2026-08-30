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
- [Official CIFAKE dataset page](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Dataset paper: Jordan J. Bird and Ahmad Lotfi, “CIFAKE: Image Classification and Explainable
  Identification of AI-Generated Synthetic Images,” IEEE Access 12 (2024), 15642-15650,
  [arXiv 2303.14126](https://arxiv.org/abs/2303.14126)
- Source-dataset citation required by the official dataset page: Alex Krizhevsky and Geoffrey
  Hinton (2009), “Learning Multiple Layers of Features from Tiny Images”

CIFAKE is excluded from primary training. Dataset contents are not distributed by this
repository.

## AIGenImages2026

- [Official dataset card](https://huggingface.co/datasets/pthan12/AIGenImages2026)
- Pinned revision: `073e1924d9d0d85ac97a53b07947b6ac95ce241c`
- Licence declaration: CC-BY-4.0
- Dataset paper: “Automated In-the-Wild Data Collection for Continual AI Generated Image
  Detection,” MAD '26, DOI 10.1145/3810988.3812662

ProofLens uses the paired validation subset and preserves generator identities and pair groups.
Retain the dataset citation and CC BY 4.0 attribution. Dataset contents are not distributed by
this repository.

## WildFake

- [Official repository recorded by the project](https://github.com/hy-zpg/AIGC-Image-Detection-Dataset)
- [Acquisition source recorded by the project](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
- Licence declaration: REQUIRES-VERIFICATION
- Dataset paper: Yan Hong, Jianming Feng, Haoxing Chen, Jun Lan, Huijia Zhu, Weiqiang Wang, and
  Jianfu Zhang, “WildFake: A Large-Scale and Hierarchical Dataset for AI-Generated Images
  Detection,” AAAI 39(4), 3500-3508 (2025),
  [DOI 10.1609/aaai.v39i4.32363](https://doi.org/10.1609/aaai.v39i4.32363)
- Attribution requirement recorded by the project: retain the dataset paper and acquisition
  source attribution

WildFake acquisition is manual. Verify the terms attached to the specific obtained copy before
training, publication, or redistribution. ProofLens does not grant permission to redistribute
WildFake.
