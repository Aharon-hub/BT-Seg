# BT‑Seg: 3D‑TransUNet for Brain Tumor Segmentation (with HD95)

**Authors:** Ari Aharon Shemesh, Itay Asael

> A reproducible Colab‑first pipeline for training and evaluating a 3D‑TransUNet‑style model on BraTS‑2019, including HD95 plots.

---

## Table of Contents
- [Overview](#overview)
- [Dataset (BraTS‑2019)](#dataset-brats2019)
- [Colab Quick Start](#colab-quick-start)
- [Inputs & File Types](#inputs--file-types)
- [One‑Time Environment Setup](#one-time-environment-setup)
- [Step‑by‑Step Pipeline](#step-by-step-pipeline)
- [Hardware & Runtime Hints](#hardware--runtime-hints)
- [Regenerating the Plots](#regenerating-the-plots)
- [Notes & Scope](#notes--scope)
- [Citations & Acknowledgements](#citations--acknowledgements)

---

## Overview
This repository explains exactly how to reproduce the training pipeline and figures shown in the accompanying presentation. All commands and paths assume **Google Colab Pro** with **Google Drive** mounted.

- **Frameworks:** PyTorch, nnU‑Net (as the training/runtime backbone), 3D‑TransUNet‑style network.
- **Metrics:** Dice score and **HD95 (95th percentile Hausdorff distance)**.
- **Outputs:** Trained checkpoints, loss curves, Dice plots, and HD95 (mm) plots.

---

## Dataset (BraTS‑2019)
- **Name:** BraTS‑2019 brain‑tumor segmentation dataset
- **Source:** Kaggle (download and unzip locally _or_ directly to your Google Drive)
- **Archive Structure:**
  - `BraTS-Training/` — **234** cases
  - `BraTS-Validation/` — **31** cases
- **Licence / Embargo:** Research‑only
- **Per‑case modalities:**
  - `*-t1n.nii`
  - `*-t1c.nii`
  - `*-t2w.nii`
  - `*-t2f.nii`
  - `*-seg.nii` (ground‑truth mask)

> **Important:** Only this dataset is used. No other public or private datasets are referenced.

---

## Colab Quick Start
Open the Colab notebook used in this repo (or your own notebook) and set **exactly two** variables to point at your BraTS folders:

```python
TRAIN_SRC = "/content/drive/MyDrive/sample_data/sample_train"  # put BraTS training folder here
TEST_SRC  = "/content/drive/MyDrive/sample_data/sample_test"   # put BraTS validation folder here
```

Both folders must contain the original sub‑directories with the pattern:

```
BraTS-xxxxx-000/
```

Then **run all cells** from top to bottom.

---

## Inputs & File Types
| Suffix     | Meaning                          | Used by |
|------------|----------------------------------|---------|
| `.nii`     | Uncompressed NIfTI‑1 image       | Copy step that arranges the cases |
| `.nii.gz`  | GZip‑compressed NIfTI‑1          | All subsequent stages (nnU‑Net planning, training, inference) |
| `.npz/.npy`| NumPy archives                    | nnU‑Net runtime only |
| `.pt/.pth` | PyTorch checkpoints               | Saved automatically by the trainer |

---

## One‑Time Environment Setup
Mount Google Drive and choose a working directory in your Drive (the code will create the required nnU‑Net folders there):

```python
from google.colab import drive
drive.mount('/content/drive')

WORK_DIR = '/content/drive/MyDrive/nnunet_colab'
```

Running the setup will create the following tree and export nnU‑Net environment variables:

```
nnUNet_raw_data_base/
preprocessed/
nnUNet_trained_models/
```

Environment variables exported:

- `nnUNet_raw_data_base`
- `nnUNet_preprocessed`
- `RESULTS_FOLDER`

---

## Step‑by‑Step Pipeline

| Step | Cell / File                    | What happens |
|-----:|--------------------------------|--------------|
| 1 | **Mount & install** section | Checks GPU & CUDA, clones 3D‑TransUNet, installs nnU‑Net and dependencies |
| 2 | **Folder & env** section | Creates the `nnUNet_raw_data_base` tree and exports env vars |
| 3 | `copy_case()` loop | Renames each modality, copies masks, populates `imagesTr/`, `labelsTr/`, and `imagesTs/` |
| 4 | Compression loop | Converts `.nii → .nii.gz` and deletes originals |
| 5 | `dataset.json` generator | Writes task metadata for nnU‑Net |
| 6 | `nnUNet_plan_and_preprocess` | Computes patch size etc. and caches dataset under `preprocessed/` |
| 7 | Sanity‑check plot | Displays image vs label slabs |
| 8 | Overwrite configs & trainer | Drops in `decoder_only.yaml`, custom `nnUNetTrainerV2.py`, and custom `nnUNetTrainerV2_HD95.py` |
| 9 | `train.py` | Starts training. Adjust `--batch_size`, `--max_num_epochs`, `--fold` as needed |
| 10 | Last cell | Generates `loss_curves.png` and other figures |

---

## Hardware & Runtime Hints

| Component | Recommended |
|-----------|-------------|
| GPU | 16 GB + (A100 / L4) |
| CPU & RAM | 8 cores / 32 GB |
| Epoch time | ~6 min (A100‑40) |
| Total wall‑time | ~8 h (80 epochs) |

Mixed precision is **enabled by default**. Disable with `--fp32` if needed.

---

## Regenerating the Plots
The very last code cell finds the most recent training log and produces three figures: **loss curve**, **Dice score**, and **HD95 (mm)**.

It first sets the base path and target folder:

```python
from pathlib import Path
BASE = Path('/content/drive/MyDrive')
FOLDER = (
    BASE
    / 'nnunet_colab'
    / 'nnUNet_trained_models'
    / 'UNet_IN_NANFang'
    / 'Task180_BraTSMet'
    / 'nnUNetTrainerV2__nnUNetPlansv2.1'
    / 'fold_0'
)
```

After retrieving the latest log from `FOLDER`, the script automatically generates:

- Loss curve
- Dice‑score plot
- HD95‑distance (mm) plot

> The images are saved alongside your training outputs for easy download.

---

## Notes & Scope
- This pipeline uses **only** the BraTS‑2019 dataset. No other public or private datasets are referenced.
- All instructions assume **Google Colab Pro** (with GPU) and **Google Drive** storage.
- Adjust training flags (`--batch_size`, `--max_num_epochs`, `--fold`, `--fp32`) to match your GPU memory/time budget.

---

## Citations & Acknowledgements
Please cite the following resources as appropriate in your work:

- **BraTS 2019**: Brain Tumor Segmentation Challenge dataset.
- **nnU‑Net**: Self‑adapting framework for U‑Net‑based medical image segmentation.
- **3D‑TransUNet**: Transformer‑based U‑Net architecture for volumetric medical image segmentation.

_This README is intentionally Colab‑centric to make reproduction straightforward._
