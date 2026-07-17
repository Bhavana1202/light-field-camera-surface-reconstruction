# Master's Thesis: Investigating Shearing and Hybrid Architectures for Sparse-to-Dense Light Field View Synthesis

Universität Heidelberg, 2026.

## Overview

This repository contains the code and results for a Master's thesis investigating two architectural interventions for sparse-to-dense light field view synthesis:

1. **Stage 1: Shearing** — effect of pre-shearing input SAIs on the reconstruction quality of FS-GAF and SADenseNet.
2. **Stage 2: A hybrid architecture** combining FS-GAF's plane-sweep-volume coarse stage with a PyTorch port of SADenseNet's correlation-based refinement module.

## Headline results

Evaluated on Kalantari 30 Scenes under matched protocol (Y-channel PSNR, 22-px boundary crop, 60 synthesized views):

| Method | Mean PSNR | Mean SSIM |
|---|---|---|
| Hybrid @ 8x8 (epoch 120) | 34.48 dB | 0.9500 |
| Hybrid @ 7x7 (epoch 280) | 36.37 dB | 0.9655 |
| SADenseNet @ 8x8 (matched re-evaluation) | 41.30 dB | 0.9834 |
| FS-GAF paper (published, 7x7) | 42.75 dB | 0.986 |

Full per-scene tables are in `results/`.

## Repository structure

- `src/` — original code contributions
  - `model/` — PyTorch port of SADenseNet correlation refinement
  - `utils/` — upstream FS-GAF utilities (util.py has apply_shear added)
  - `test_hybrid.py` — hybrid model evaluation script
  - `convert_kalantari_to_fsgaf.py` — data bridge (SADenseNet h5 to FS-GAF format)
  - `test_benchmark_yonly.py` — SADenseNet matched-protocol Y-only evaluation
- `upstream_modified/` — modified upstream files with .orig backups
  - `fsgaf/` — FS-GAF train.py and model_lfasr.py (patched)
  - `sadensenet/` — test_benchmark_shear.py (patched)
- `results/` — all experimental results
  - `stage1_shearing/` — shearing sweeps on both models
  - `stage2_8x8/` — hybrid at 8x8 angular resolution
  - `stage2_7x7/` — hybrid at 7x7 angular resolution
  - `plots/` — generated plots
- `figures/` — TikZ architecture diagrams for the thesis
- `docs/development_history/` — transient patch/probe scripts (reproducibility record)
- `docs/exploratory_pilot/` — earlier HCI-based hybrid implementation, preserved for methodological transparency

## How to reproduce

Trained model checkpoints are NOT included (too large for git). Contact the author or see GitLab release attachments.

### Environments

- `LFASR`: PyTorch 1.4, Python 3.6 (for the hybrid model)
- `sadense`: TensorFlow 1.15 + Keras, Python 3.6 (for original SADenseNet evaluation)

### Training the hybrid (8x8)

Convert Kalantari data first:

    python src/convert_kalantari_to_fsgaf.py

Then train:

    python upstream_modified/fsgaf/train.py \
        --arb_sample 0 --angular_in 2 --angular_out 8 \
        --dataset Kalantari --dataset_path ./LFData/Kalantari_train_fsgaf.h5 \
        --psv_range 4 --psv_step 50 --patch_size 64 \
        --num_cp 10 --max_epoch 301 --lr 1e-4 --step 100 --reduce 0.5 \
        --shear_amount 0.0

For 7x7, change `--angular_out 7` and `--dataset_path ./LFData/Kalantari_train_fsgaf_7x7.h5`.

### Evaluation

    python src/test_hybrid.py \
        --model_path <path_to_model.pth> \
        --test_path ./LFData/Kalantari_test_fsgaf.h5 \
        --angular_out 8 --input_ind 0 7 56 63

For 7x7 evaluation use `--angular_out 7 --input_ind 0 6 42 48` and the 7x7 test h5.

## References

- Jin et al. Deep coarse-to-fine dense light field reconstruction with flexible sampling and geometry-aware fusion. IEEE TPAMI, 2022.
- Yeung et al. Light field spatial super-resolution using deep efficient spatial-angular separable convolution. IEEE TIP, 2019.
- Kalantari et al. Learning-based view synthesis for light field cameras. ACM TOG, 2016.

## Author

Bhavana Krishna, Master's student, Universität Heidelberg.

## Supervisor

Prof. Jürgen Hesser, Universität Heidelberg.
