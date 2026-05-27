# IbnSinaNet — MS Lesion Segmentation

An attention-based Fully Convolutional DenseNet for Multiple Sclerosis (MS) lesion segmentation from FLAIR MRI, built on a modified FC-DenseNet architecture with Cross-Stage Partial (CSP) connections, Grouped Convolutions (GCNN), and Squeeze Attention (SA) blocks.

This repository extends and accompanies the paper:

> **IbnSinaNet: A Hybrid CSPDenseNet and Squeeze Attention under the Supervision of Group CNNs for Automatic Segmentation of MS Lesions in MRI**
> Dehghanpour et al., *Scientific Reports*, 2026.
<!-- > [https://www.sciencedirect.com/science/article/pii/S0010482523004869](https://www.sciencedirect.com/science/article/pii/S0010482523004869) -->

The model backbone is based on the PyTorch implementation of [The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326), which we retained and extended.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Key Hyperparameters](#key-hyperparameters)
- [Evaluation Metrics](#evaluation-metrics)
- [Ablation Study](#ablation-study)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Overview

IbnSinaNet addresses the MS lesion segmentation problem — a highly class-imbalanced task where lesion pixels constitute less than 1% of the total brain volume. The model combines:

- **FC-DenseNet backbone** for dense feature reuse across encoder and decoder
- **CSP (Cross-Stage Partial) connections** to reduce redundant gradient information
- **GCNN (Grouped Convolutions)** for grouped feature processing across input channels
- **Squeeze Attention (SA) blocks** for spatially-aware recalibration of feature maps

Training is evaluated using **Repeated 5-Fold Cross-Validation** to produce statistically reliable performance estimates, and results are validated with a **Wilcoxon signed-rank test** against each ablation variant.

---

## Architecture

```
Input (3 × 160 × 160)
       │
  First Conv (GCNN, groups=3)
       │
  ┌────▼─────────────────────────────────┐
  │  Encoder (5 stages)                  │
  │  FCDenseStage × 5                    │
  │  └─ FCDenseBlock (CSP + DenseLayer)  │
  │  └─ SqueezeAttentionBlock (SA)       │
  │  └─ TransitionDown (stride-2 conv)   │
  └──────────────────────────────────────┘
       │  skip connections
  ┌────▼─────────────────────────────────┐
  │  Decoder (4 stages)                  │
  │  TransitionUp (bilinear + concat)    │
  │  FCDenseStage × 4                    │
  │  └─ FCDenseBlock                     │
  │  └─ SqueezeAttentionBlock            │
  └──────────────────────────────────────┘
       │
  Final Conv (1×1) → 2-class output
```

### Architecture Diagram

The proposed architecture, IbnSinaNet, comprises three fundamental components: Transition Down, Dense, and Squeeze Attention blocks. These components are illustrated with corresponding color coding of yellow, blue, and green, respectively. The blue block represents a dense block. A portion of this block is directed to the CSP block, while another portion is not forwarded to the CSP block. Ultimately, the output of the dense block is transmitted to the Squeeze Attention block. In addition, the use of group convolution lies in the proposed architecture.

<p align="center">
       <img width="1299" height="591" alt="Ostovane for IbnSinaNet" src="https://github.com/user-attachments/assets/bd0ff29a-55a5-491d-84cc-68e093fd6ceb" />

  <br>
  <em>The detailed schematic illustration of the IbnSinaNet architecture. G indicates the group of subjects.</em>
</p>


<p align="center">
      <img width="3508" height="2480" alt="DPR (e)" src="https://github.com/user-attachments/assets/c46c462b-60dd-458c-9de0-d5b07c006458" />

  <br>
  <em>The hierarchical illustration of IbnSinaNet architecture. IbnSinaNet is a hybrid deep learning model combining Transition Down
Block, Dense Block, Squeeze-Attention Block. The Dense Blocks are categorized into two groups, including DenseNet Blocks
and CSPDenseNet Blocks.</em>
</p>
  <br>

**Ablation variants** (controlled via `opt.use_sa`, `opt.ablation_csp`, `opt.ablation_gcnn`):

| Config | Dense | CSP | SA | GCNN |
|---|---|---|---|---|
| Baseline U-Net | ✗ | ✗ | ✗ | ✗ |
| +DenseNet | ✓ | ✗ | ✗ | ✗ |
| +DenseNet +CSP | ✓ | ✓ | ✗ | ✗ |
| +DenseNet +CSP +SA | ✓ | ✓ | ✓ | ✗ |
| **Full IbnSinaNet** | ✓ | ✓ | ✓ | ✓ |

---

## Project Structure

```
.
├── run_repeated_cv.py          # Main entry point: repeated CV + ablation study
│
├── datasets/
│   ├── MSDataset.py            # Dataset class — loads FLAIR slices + masks
│   ├── datautils.py            # get_transforms() for train/val pipelines
│   └── joint_transforms.py     # Joint augmentations (scale, crop, flip, affine…)
│
├── models/
│   ├── tiramisu.py             # FCDenseNet67 (full model)
│   ├── tiramisu_ablation.py    # FCDenseNet67 with ablation flags (CSP, GCNN)
│   ├── blocks.py               # FCDenseStage, FCDenseBlock, CSPDenseLayer
│   ├── blocks_ablation.py      # Same blocks with opt.use_sa toggle
│   ├── layers.py               # TransitionDown/Up, SqueezeAttentionBlock
│   └── BiConvLSTM.py           # Bidirectional ConvLSTM (optional module)
│
├── training/
│   ├── argprocess.py           # All hyperparameter definitions and defaults
│   ├── trainer.py              # Multi-GPU DDP training loop (nccl backend)
│   └── training.py             # Train/test utilities, metrics, weight I/O
│
├── utils/
│   ├── general_utils.py        # Tensor-to-image helpers, batch saving
│   ├── lesion_metrics.py       # Lesion-level LDR and FPPS (connected components)
│   ├── losses.py               # DiceLoss, FocalLoss, CE+Dice combined
│   └── imgs.py                 # Normalization utilities
│
├── test.py                     # ⚠️ Legacy — not used by run_repeated_cv.py
├── train.py                    # ⚠️ Legacy — not used by run_repeated_cv.py
├── save_results.py             # Saves 5-panel visualization images per slice
├── plot_curves.py              # Training curve plots (per-fold + overlaid)
├── statistical_analysis.py     # Wilcoxon test vs. Full IbnSinaNet
│
└── all_results.json            # Output: all CV and ablation results
```

---

## System Requirements

> **This code runs on Linux only, via command line. Windows is not supported.**

The training backend uses **NCCL** (NVIDIA Collective Communications Library) for multi-GPU communication, which is Linux-only. The code has no GUI dependency and is designed to run entirely from the terminal on a GPU server.

**GPU support** is fully automatic — the code detects all available GPUs at runtime:
```python
world_size = torch.cuda.device_count()  # automatically uses all GPUs
```
- 1 GPU → effective batch = `batch_size × 1`
- 4 GPU → effective batch = `batch_size × 4`
- 6 GPU → effective batch = `batch_size × 6`

There is no need to set the number of GPUs manually.

---

## Installation

Create a conda environment and install dependencies:

```bash
conda create -n ibnsina-ms python=3.9
conda activate ibnsina-ms
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install matplotlib scipy tqdm ptflops tensorboard pillow
```

- **OS:** Linux only
- **Python:** ≥ 3.9
- **PyTorch:** ≥ 1.12 with CUDA
- **GPUs:** 1 or more NVIDIA GPUs (NCCL backend)

---

## Dataset

The code supports the **ISBI 2015 MS Lesion Segmentation Challenge** dataset.

Download the original NIfTI data here:
[https://smart-stats-tools.org/lesion-challenge-2015](https://smart-stats-tools.org/lesion-challenge-2015)

Or use the pre-processed version (PNG slices, already organized) available here:
[ISBI dataset ready for training](https://iplab.dmi.unict.it/mfs/dataset/alessiarondinella/ISBI_2015.tar)

### Required folder structure

After converting NIfTI files to PNG slices, arrange the dataset as follows:

```
dataset/
└── ISBI_2015/
    ├── train1/
    │   ├── P1_T1/      ← FLAIR images
    │   └── P5_T4/
    ├── train1annot/
    │   ├── P1_T1/      ← binary lesion masks
    │   └── P5_T4/
    ├── ...
    ├── train5/
    ├── train5annot/
    ├── val1/
    ├── val1annot/
    ├── ...
    ├── val5/
    ├── val5annot/
    ├── test1/
    ├── test1annot/
    ├── ...
    ├── test5/
    └── test5annot/
```

`MSDataset` automatically pairs each mask in `*annot/` with its corresponding FLAIR image in the matching `*/` folder. Only slices with at least one lesion pixel are included in training.

**Normalization statistics:** `mean=0.1026`, `std=0.0971` (per-slice min-max normalization applied first).

### Data Augmentation (training only)

Implemented in `datasets/joint_transforms.py` and applied jointly to image and mask:

| Transform | Details |
|---|---|
| `JointScale` | Resize to `input_dim` (default 160) |
| `JointCenterCrop` | Center crop with optional random translation (±25px during training) |
| `JointRandomHorizontalFlip` | p=0.5 |
| `JointRandomRotation` | 90°/180°/270° (available, off by default) |
| `JointRandomAffine` | Affine transform (available, off by default) |

---

## Usage

> All commands must be run on **Linux** from the terminal. The pipeline entry point is always `run_repeated_cv.py` — `train.py` and `test.py` are legacy files and are not part of the main pipeline.

### Execution Order

Run these commands in order:

**Step 1 — Main pipeline** (train + test for all folds and repetitions):

```bash
conda activate ibnsina-ms

# Full pipeline: Repeated CV + Ablation Study
python run_repeated_cv.py -dp /path/to/dataset/ISBI_2015

# Or: Repeated CV only
python run_repeated_cv.py -dp /path/to/dataset/ISBI_2015 --cv-only

# Or: Ablation Study only
python run_repeated_cv.py -dp /path/to/dataset/ISBI_2015 --ablation-only
```

Outputs `all_results.json` and all model checkpoints (`model-rep{R}-fold{F}.pt`).

---

**Step 2 — Statistical analysis** (requires `all_results.json`):

```bash
python statistical_analysis.py
```

Runs Wilcoxon signed-rank test for each ablation configuration against Full IbnSinaNet.

---

**Step 3 — Training curves** (requires TensorBoard logs from Step 1):

```bash
python plot_curves.py
```

Produces:
- `figure13_original.png` — Loss and Dice curves per fold
- `figure13_overlay.png` — All folds overlaid + mean ± std band

---

**Step 4 — Visualization images** (requires trained checkpoints from Step 1):

```bash
python save_results.py -dp /path/to/dataset/ISBI_2015
```

Saves 5-panel composite images per test slice:

```
[Input MRI] | [Ground Truth] | [Prediction] | [FP/FN Map] | [Overlay]
```

---

### What `run_repeated_cv.py` does internally

`run_repeated_cv.py` handles the full training and testing pipeline by itself. It uses `training/trainer.py` for multi-GPU training (not `train.py`) and contains its own test loop (not `test.py`):

```
run_repeated_cv.py
    ├── train_fold()  →  training/trainer.py  (multi-GPU, NCCL)
    └── test_fold()   →  built-in test loop   (single GPU)
```

`train.py` and `test.py` are **not called** by `run_repeated_cv.py` and can be ignored.

---

## Key Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `--batch-size` / `-b` | 2 per GPU (effective: **12** with 6 GPUs) | Batch size per GPU |
| `--dropout` / `-drop` | 0.1 | Dropout rate in SA conv block (attention branch always 0) |
| `--learning-rate` / `-lr` | 3e-4 | Initial learning rate (Adam) |
| `--weight-decay` / `-wd` | 3e-4 | Adam weight decay |
| `--learning-rate-decay-by` / `-lrdb` | 0.99 | Exponential LR decay factor (per epoch) |
| `--learning-rate-decay-every` / `-lrde` | 1 | Decay every N epochs |
| `--num-epochs` / `-ne` | 150 | Training epochs per fold |
| `--input-dim` / `-dim` | 160 | Input image resolution (160×160) |
| `--seq-size` / `-ss` | 3 | Number of consecutive slices per sample |
| `--loss-type` | CE+Dice | Loss function (CE / Dice / Focal / CE+Dice) |
| `--use-sa` / `-usesa` | True | Enable Squeeze Attention blocks |
| `--dataset-path` / `-dp` | — | Path to ISBI_2015 root folder |
| Optimizer | Adam | Fixed in code |
| CV repetitions | 10 | Fixed in `run_repeated_cv.py` |
| Ablation repetitions | 10 | Fixed in `run_repeated_cv.py` |

All parameters can be overridden via command-line flags. See `training/argprocess.py` for the complete list.

---

## Evaluation Metrics

### Pixel-Level

| Metric | Description |
|---|---|
| Dice (DSC) | Primary segmentation metric |
| Sensitivity | True positive rate |
| Specificity | True negative rate |
| PPV | Precision / Positive Predictive Value |
| NPV | Negative Predictive Value |
| F1 | Harmonic mean of PPV and Sensitivity |
| Accuracy | Overall pixel accuracy |

### Lesion-Level

Computed via connected-component analysis (8-connectivity, IoU threshold = 0.1), consistent with MICCAI MSSEG challenge evaluation:

| Metric | Description |
|---|---|
| **LDR** | Lesion Detection Rate — fraction of GT lesions detected |
| **FPPS** | False Positives Per Scan — average spurious lesions per image |

> **Note:** Pixel-level Specificity above 99.9% is a well-known artifact of extreme class imbalance in MS lesion data. LDR and FPPS are the clinically meaningful alternatives and are reported alongside pixel-level metrics.

### Statistical Validation

Repeated CV reports **95% confidence intervals** and **coefficient of variation (CV%)** for every metric. CV < 10% indicates stable, reproducible results.

---

## Ablation Study

Each architectural component is evaluated independently by enabling/disabling flags in `argprocess.py`:

| Flag | Controls |
|---|---|
| `opt.use_sa` | Squeeze Attention blocks |
| `opt.ablation_csp` | Cross-Stage Partial connections |
| `opt.ablation_gcnn` | Grouped convolutions (GCNN) |

Results are stored in `all_results.json` under the `ablation` key. Each configuration is identified by its tag: `baseline`, `dense`, `dense_csp`, `dense_csp_sa`, `full`.

---

## Outputs

| File / Folder | Description |
|---|---|
| `all_results.json` | All CV and ablation metrics (mean, std, 95% CI, per-fold breakdown) |
| `model-rep{R}-fold{F}.pt` | Trained model checkpoints |
| `results_ms/rep{R}/fold{F}/tb/` | TensorBoard event logs |
| `results_ms/test_images/fold{F}/combined/` | 5-panel composite images |
| `results_ms/test_images/fold{F}/input/` | Raw FLAIR MRI slices |
| `results_ms/test_images/fold{F}/target/` | Ground truth masks |
| `results_ms/test_images/fold{F}/predict/` | Predicted masks |
| `results_ms/test_images/fold{F}/errors/` | FP/FN error maps |
| `results_ms/test_images/fold{F}/overlay/` | Prediction overlaid on MRI |
| `figure13_original.png` | Per-fold training curves |
| `figure13_overlay.png` | All folds overlaid + mean ± std |

### Visualization Color Coding

| Color | Meaning |
|---|---|
| 🔵 Blue (cornflower) | True Positive (TP) |
| 🔴 Red | False Positive (FP) — predicted lesion with no ground truth |
| 🟢 Green | False Negative (FN) — ground truth lesion not detected |

---

## Qualitative Results

The segmentation results of the proposed approach on the test set are visually represented in the following figure, which displays the outcomes on slices from the same patient, extracted from three distinct regions of the brain. An axial slice of a subject is shown, illustrating the ground truth segmentations alongside the segmentations produced by the proposed method. False positive and false negative pixels are distinctly marked in red and green, respectively. The predicted lesions mask closely resembles the ground truth mask, indicating that the proposed approach accurately segments the majority of lesions. The mask highlighting false positive and false negative pixels further corroborates that the model effectively detected most of the Flair MS lesions. Also, our proposed method demonstrates a remarkable ability to accurately segment MS plaques, including those that are very small in size. This level of precision in segmentation is evident and noteworthy.

<p align="center">
 <img width="732" height="572" alt="False Positive and False Negative" src="https://github.com/user-attachments/assets/bbd56ae4-8649-4556-93fd-d03151843e89" />
  <br>
  <em> (a, e, i) Axial slice of a subject, (b, f, j) The ground truth segmentations, (c, g, k) The segmentations obtained by the proposed approach, (d, h, l) The false positive and false negative pixels distinguished in red and green pixels respectively.</em>
</p>

<p align="center">
      <img width="647" height="623" alt="hitmaps" src="https://github.com/user-attachments/assets/4a46011c-8a0f-4e16-865c-7860e3605847" />

  <br>
  <em> (a, d, g) Axial slice of a subject with the lesion highlighted in yellow; (b, e, h) Segmented lesion utilizing the proposed method, indicated in blue; (c, f, i) False positive and false negative pixels distinguished by red and green, respectively.</em>
</p>

## Reproducibility

Fixed seeds are used for each CV repetition:

```python
SEEDS = [42, 123, 256, 512, 1024, 2048, 3141, 9999, 7777, 1111]
```

Each seed is applied to Python `random`, NumPy, and PyTorch (including CUDA) before data loading and training. `torch.backends.cudnn.deterministic = True` is set globally.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{dehghanpour2024ibnsinanet,
  title={IbnSinaNet: A Hybrid CSPDenseNet and Squeeze Attention under the Supervision of Group CNNs for Automatic Segmentation of MS Lesions in MRI},
  author={Dehghanpour, Mahshid and Fateh, Mansoor and Mohammadpoory, Zeynab and Ferdowsi, Saideh},
  journal={Scientific Reports},
  volume={XX},
  number={X},
  pages={XX-XX},
  year={2026},
  publisher={Nature Publishing Group},
  doi={XX.XXXX/s41598-XXX-XXXXX-X}
}
```

---

## License

This repository is licensed under the **MIT License**.
