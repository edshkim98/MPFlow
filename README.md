<div align="center">

# MPFlow: Multi-Modal Posterior-Guided Flow Matching for Zero-Shot MRI Reconstruction

**Accepted to MICCAI 2026 — to appear**

[![Paper](https://img.shields.io/badge/arXiv-2603.03710-b31b1b.svg)](https://arxiv.org/abs/2603.03710)
[![Conference](https://img.shields.io/badge/MICCAI-2026-4b44ce.svg)](https://www.miccai.org/)
[![Python](https://img.shields.io/badge/Python-3.9-3776ab.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg)](https://pytorch.org/)

Seunghoi Kim · Chen Jin · Henry F. J. Tregidgo · Matteo Figini · Daniel C. Alexander

</div>

---

> **Note** — This paper has been accepted to **MICCAI 2026** but the conference proceedings are not yet published. This repository is a research preview; the citation below points to the arXiv preprint and will be updated once the official proceedings are available.

**MPFlow** is a **zero-shot** MRI reconstruction framework built on **rectified flow** (flow matching). Zero-shot reconstruction relies on generative priors, but a single-modality unconditional prior tends to *hallucinate* under severe ill-posedness. In real clinical workflows, complementary acquisitions (e.g. a high-quality structural scan) are routinely available — MPFlow brings that auxiliary modality in **at inference time, without retraining the generative prior**, to anchor anatomical fidelity and suppress hallucination.

## Table of Contents

- [Overview](#overview)
- [Why It Works](#why-it-works)
- [Method](#method)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Overview

Reconstructing a high-quality MR image from an undersampled or degraded measurement is severely ill-posed: many images are consistent with the same measurement, so an unconditional generative prior can invent structures that are plausible but absent from the true anatomy. MPFlow resolves this ambiguity using a second, complementary MRI contrast that is already available in the scan session — steering the reconstruction toward the anatomy the auxiliary modality actually supports, entirely at inference time.

## Why It Works

A generative prior over a *single* modality knows what MR images of that contrast look like in general, but nothing about *this specific patient's* anatomy beyond the measurement. When the measurement is uninformative (high acceleration, low field), the prior fills the gap with population-typical structure — an **intrinsic hallucination** from the prior, or an **extrinsic hallucination** inherited from an upstream conditional model.

MPFlow closes this gap by guiding sampling with **two complementary signals** at every step:

- **Data consistency** anchors the reconstruction to the observed measurement, so the solution cannot drift away from the acquired signal.
- **Cross-modal feature alignment** anchors it to a *second* MRI contrast of the same patient — via the pre-trained **PAMRI** encoder — so anatomy that the auxiliary scan does not support is actively suppressed.

Because the auxiliary modality carries patient-specific structure the measurement lacks, it removes exactly the ambiguity that produces hallucination — without any task-specific or paired training. Rectified flow makes this efficient: its near-straight trajectories from noise to image reach diffusion-level quality in a **fraction of the sampling steps**.

## Method

MPFlow has two stages — a one-time self-supervised pretraining of the cross-modal prior (**PAMRI**), and the zero-shot flow-matching reconstruction that uses it.

| Component | What it does | Where |
| :-------- | :----------- | :---- |
| **PAMRI** — Patch-level Multi-modal MR Image Pretraining | A self-supervised strategy that learns a **shared representation across modalities** (e.g. T1 & T2) from unpaired patches, using global + dense contrastive (InfoNCE) alignment with an auxiliary reconstruction head. This provides the cross-modal guidance signal — no reconstruction labels needed. | `ssl_mri/` |
| **Rectified-flow prior** | An unconditional flow-matching generative model of MR images, sampled with a fast ODE solver (Euler). | `flowmatching/` |
| **Posterior guidance** | At each flow step, sampling is jointly steered by **data consistency** (measurement) and **cross-modal feature alignment** (pre-trained PAMRI), systematically suppressing intrinsic and extrinsic hallucinations. | `flowmatching/guided_diffusion/condition_methods.py` |

The reconstruction is **zero-shot**: neither the flow prior nor PAMRI is retrained per task or per acceleration factor.

## Results

On **HCP** and **BraTS**, MPFlow matches diffusion baselines on image quality while using only **~20% of the sampling steps**, and reduces **tumor hallucinations by more than 15%** (segmentation Dice score) — demonstrating that cross-modal guidance enables more reliable *and* more efficient zero-shot MRI reconstruction. See the [paper](https://arxiv.org/abs/2603.03710) for quantitative tables and qualitative comparisons.

## Installation

Python ≥ 3.9 with a CUDA-capable GPU is recommended. The generative backbone builds on OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion); the flow-matching and posterior-guidance components live under `flowmatching/`.

```bash
# core dependencies: torch, numpy, nibabel, pyyaml, tqdm, matplotlib
pip install torch numpy nibabel pyyaml tqdm matplotlib

# evaluation extras (FID, etc.)
pip install -r flowmatching/evaluations/requirements.txt
```

## Usage

### Stage 1 — Pretrain the cross-modal prior (PAMRI)

Learns the shared T1/T2 representation used for cross-modal guidance. Set `MODE = 'dense'` (recommended) or `'global'` at the top of the script:

```bash
cd ssl_mri
python ssl_train.py
```

The best encoder is saved to `best_ssl_model_*.pth`; point the reconstruction stage at this checkpoint.

### Stage 2 — Zero-shot reconstruction with MPFlow

Runs rectified-flow sampling with posterior + cross-modal guidance:

```bash
cd flowmatching
python image_sample.py --flow_matching True --steps 100 --model_path /path/to/flow_model.pt
```

> **Before running** — update the data paths, config path, and checkpoint locations at the top of `ssl_train.py` / `image_sample.py` (and the referenced `configs_*.yaml`) to match your environment. The scripts currently point at absolute cluster paths.

## Configuration

Inference is driven by a YAML config (e.g. `configs_kspace.yaml`) loaded by `image_sample.py`. The main knobs:

| Key / Flag | Meaning |
| :--------- | :------ |
| `--flow_matching` | use the rectified-flow sampler (vs. the original diffusion sampler) |
| `--steps` | number of ODE integration steps (flow matching needs far fewer than diffusion) |
| `multimodal` | enable cross-modal guidance from the auxiliary modality (PAMRI) |
| `skip_timestep` | warm-start the trajectory from the measurement rather than pure noise |
| `conditioning.method` | measurement-guidance method (posterior sampling) |
| `measurement.operator` | forward measurement model (e.g. k-space undersampling / blur) |
| `measurement.noise` | measurement noise model (e.g. Gaussian) |
| `norm` | intensity normalisation applied to the MR volumes |

## Repository Structure

```
├── ssl_mri/
│   └── ssl_train.py              # PAMRI: patch-level multi-modal self-supervised pretraining
│                                 #   (T1/T2 InfoNCE global + dense contrastive + reconstruction head)
└── flowmatching/                 # MPFlow: rectified-flow zero-shot reconstruction
    ├── image_sample.py           # inference entry point (flow-matching sampler + posterior guidance)
    ├── guided_diffusion/         # backbone: U-Net, flow/gaussian sampling, measurements, conditioning
    ├── util/                     # fastMRI / k-space helpers, metrics, resizer
    └── evaluations/              # FID / evaluation utilities
```

## Citation

This work is accepted to MICCAI 2026 (proceedings forthcoming). For now, please cite the arXiv preprint:

```bibtex
@article{kim2026mpflow,
  title   = {MPFlow: Multi-modal Posterior-Guided Flow Matching for Zero-Shot MRI Reconstruction},
  author  = {Kim, Seunghoi and Jin, Chen and Tregidgo, Henry F. J. and Figini, Matteo and Alexander, Daniel C.},
  journal = {arXiv preprint arXiv:2603.03710},
  year    = {2026},
  note    = {To appear in MICCAI 2026}
}
```

## Acknowledgements

This code builds upon OpenAI's [guided-diffusion](https://github.com/openai/guided-diffusion) and the [DPS](https://github.com/DPS2022/diffusion-posterior-sampling) (Diffusion Posterior Sampling) framework. We thank the authors for releasing their code.

For questions, contact **Seunghoi Kim** — [seunghoi.kim.17@ucl.ac.uk](mailto:seunghoi.kim.17@ucl.ac.uk).
