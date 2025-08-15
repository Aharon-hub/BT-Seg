README — BT-Seg: 3D-TransUNet for Brain Tumor Segmentation (with HD95)
Authors: Ari Aharon Shemesh, Itay Asael
=============================================================

This document explains what you need to do to reproduce the pipeline and the plots contained in the presentation.
All shell commands are meant to be executed inside **Google Colab Pro**.

----------------------------------------------------------------
1. Dataset
----------------------------------------------------------------
Name: BraTS‑2019 brain‑tumour segmentation dataset
Where to get it: Kaggle website. Download and unzip locally or directly to your Google Drive. The archive contains two folders: `BraTS-Training` (234 cases) and `BraTS-Validation` (31 cases).
Licence / embargo: Research‑only.
Modality files inside each case:
    * *-t1n.nii
    * *-t1c.nii
    * *-t2w.nii
    * *-t2f.nii
    * *-seg.nii (ground‑truth mask)

Important: Only this dataset is used. No other public or private datasets are referenced.

----------------------------------------------------------------
2. Where to tell the notebook where the data live
----------------------------------------------------------------
Open the Colab notebook and adjust exactly two variables:

    TRAIN_SRC = '/content/drive/MyDrive/sample_data/sample_train'  #  put BraTS training folder here
    TEST_SRC  = '/content/drive/MyDrive/sample_data/sample_test'   #  put BraTS validation folder here

Both folders must contain the original sub‑directories named `BraTS-xxxxx-000/`.

----------------------------------------------------------------
3. File types consumed by the code
----------------------------------------------------------------
Suffix    Meaning                             Consumed by
------    -----------------------------       -----------------------------
.nii      Uncompressed NIfTI‑1 image          Copy step that arranges the cases
.nii.gz   GZip‑compressed NIfTI‑1            All subsequent stages (nnU‑Net planning, training, inference)
.npz/.npy NumPy archives                      nnU‑Net runtime only
.pt/.pth  PyTorch checkpoints                 Saved automatically by the trainer

----------------------------------------------------------------
4. One‑time environment setup
----------------------------------------------------------------
    from google.colab import drive
    drive.mount('/content/drive')

    WORK_DIR = '/content/drive/MyDrive/nnunet_colab' 

The script then creates:

    nnUNet_raw_data_base/
    preprocessed/
    nnUNet_trained_models/

and exports the environment variables `nnUNet_raw_data_base`, `nnUNet_preprocessed`, and `RESULTS_FOLDER`.

----------------------------------------------------------------
5. Step‑by‑step command flow
----------------------------------------------------------------
Step  Cell / file                       What happens
----  -------------------------------   ---------------------------------------------
1     Mount & install section           Checks GPU & CUDA, clones 3D‑TransUNet, installs nnU‑Net and deps
2     Folder & env section              Creates nnUNet_raw_data_base tree and exports env vars
3     Copy loop (copy_case())           Renames each modality, copies masks, populates imagesTr/labelsTr/imagesTs
4     Compression loop                  Converts .nii → .nii.gz and deletes originals
5     dataset.json generator            Writes task metadata for nnU‑Net
6     nnUNet_plan_and_preprocess        Computes patch‑size etc. caches dataset in preprocessed/
7     Sanity‑check plot                 Shows image vs. label slabs
8     Overwrite configs & trainer       Drops in decoder_only.yaml and custom nnUNetTrainerV2.py and custom nnUNetTrainerV2_HD95.py
9     train.py                          Starts training. Adjust --batch_size, --max_num_epochs, --fold as needed
10    Last cell                         Generates loss‑curves plot (loss_curves.png)

----------------------------------------------------------------
6. Hardware & runtime hints
----------------------------------------------------------------
Component        Recommended
---------        ---------------   
GPU              16 GB + (A100 / L4)
CPU & RAM        8 cores / 32 GB
Epoch time       ~6 min (A100‑40)
Total wall‑time  8 h (80 epochs)

Mixed precision is enabled by default; disable with --fp32 if needed.

----------------------------------------------------------------
7. Regenerating the plots
----------------------------------------------------------------
The very last code cell locates the newest training-log file by first setting `BASE = Path('/content/drive/MyDrive')` 
and defining `FOLDER` as `BASE / 'nnunet_colab' / 'nnUNet_trained_models' / 'UNet_IN_NANFang' / 'Task180_BraTSMet' / 'nnUNetTrainerV2__nnUNetPlansv2.1' / 'fold_0'`; 
after retrieving the log, it automatically generates both the loss curve, the Dice-score plot and the HD95-distance (mm) plot.
----------------------------------------------------------------

