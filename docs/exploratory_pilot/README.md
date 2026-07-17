# Exploratory Pilot: Early HCI Hybrid Implementation

This directory preserves an earlier, exploratory implementation of the hybrid
architecture that predates the final version in `src/model/`. It is included
as a record of the design iteration process.

## What it is

An earlier attempt at combining FS-GAF's `Net_view` (coarse SAI synthesis)
with a SADenseNet-style correlation-based refinement, trained and evaluated
on the HCI 4D Light Field dataset at 7x7 angular resolution.

## Key configuration (from `run_cfg.json`)

- Dataset: HCI (train_HCI.h5 / test_HCI.h5)
- Angular grid: 7x7, 4 corner input SAIs
- Epochs: 60
- Batch size: 1
- Learning rate: 2e-4
- Patch size: 112
- PSV range: 3, PSV step: 20
- Mixed-precision training (AMP): enabled

## Result

Evaluated on 4 HCI test scenes (see `results_hci_hybrid.csv`):

- Mean PSNR: 30.08 dB
- Mean SSIM: 0.8506

## Differences from the final implementation (`src/model/`)

The pilot architecture is structurally similar to the final SADenseRefine but
differs in several ways:

- Uses `Conv2d` with tensor reshaping tricks instead of `Conv3d`
- Uses `BatchNorm2d` after each convolution (the final version has no batch norm)
- Different reduce-tail structure (1x1 compress conv from 49 -> growth channels)
- Trained on HCI (not Kalantari)
- Uses AMP (mixed precision), shorter training, smaller patch config

## Why this is preserved

The pilot demonstrates that the final architecture emerged from iteration
rather than being adopted whole from a single design choice. It is not the
result reported in the thesis (see `results/stage2_8x8/` and `results/stage2_7x7/`
for the final experiments).

## Files

- `hybrid/` : pilot source code
- `run_cfg.json` : training configuration used
- `results_hci_hybrid.csv` : evaluation results on 4 HCI scenes

The pilot checkpoint (`hci_best.pth`, 11 MB) is not included in the git tree.
Contact the author if you need it.
