"""
Matched-protocol evaluation of SADenseNet on Kalantari 30Scenes.

Differences from the original test_benchmark.py:
- PSNR/SSIM computed on Y channel only (not BGR-averaged)
- Same 22-px boundary crop as the original
- Same 60-view averaging (synthesized views only, excluding 2x2 input corners)
- Outputs a CSV directly comparable to our hybrid's psnr_60view_synthesis column
"""
import argparse
import json
import os
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from components.datasets import get_dataset
from components.generator import Generator
from components.model import create_model
from components.config import Config
from components.utils import sai_io_idx, path2img_name
from components.utils.evaluate import postprocess


CROP = 22


def score_y_only(y_pred, ycrcb_gt_full, a_in, a_out, crop=CROP):
    in_sai, out_sai = sai_io_idx(ycrcb_gt_full.shape[0:2], a_in)
    ycrcb_flat = ycrcb_gt_full.reshape(
        [a_out[0] * a_out[1], ycrcb_gt_full.shape[2],
         ycrcb_gt_full.shape[3], ycrcb_gt_full.shape[4]]
    )[out_sai, ...]
    y_gt = ycrcb_flat[..., 0]
    assert y_pred.shape == y_gt.shape, f"pred {y_pred.shape} vs gt {y_gt.shape}"

    psnrs, ssims = [], []
    for i in range(y_pred.shape[0]):
        p = y_pred[i, crop:-crop, crop:-crop]
        g = y_gt[i, crop:-crop, crop:-crop]
        psnrs.append(compute_psnr(g, p, data_range=1.0))
        ssims.append(compute_ssim(
            (g * 255.0).astype(np.uint8),
            (p * 255.0).astype(np.uint8),
            gaussian_weights=True, sigma=1.5,
            use_sample_covariance=False
        ))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def main():
    parser = argparse.ArgumentParser("SADenseNet Y-only matched-protocol eval.")
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpuid', default='0')
    parser.add_argument('--out_csv', default='tmp/test/sadense_yonly_matched_30Scenes.csv')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    tfc = tf.ConfigProto()
    tfc.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=tfc))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    ds = '30Scenes'
    test_path = get_dataset(ds, Config).get_path_test()
    print(f"loading test set from {test_path}")

    with h5py.File(test_path, 'r') as h5:
        names = json.loads(h5.attrs['names'])
        n = len(names)
        print(f"{n} scenes to evaluate")

        psnr_lst = np.zeros(n)
        ssim_lst = np.zeros(n)

        for i in tqdm(range(n)):
            ycrcb = h5[names[i] + '/ycrcb'][:]
            lf = ycrcb[..., 0]

            model = create_model((ycrcb.shape[2], ycrcb.shape[3]), Config)
            model.load_weights(args.model)

            x, _ = Generator.get_xy(lf, Config.a_in)
            x = np.expand_dims(x, 0)
            y_pred = model.predict(x, batch_size=1)
            y_pred = postprocess(y_pred[0])

            psnr_v, ssim_v = score_y_only(y_pred, ycrcb,
                                          Config.a_in, Config.a_out, crop=CROP)
            psnr_lst[i] = psnr_v
            ssim_lst[i] = ssim_v
            if i == 0:
                print(f"\nFirst scene ({path2img_name(names[i])}):")
                print(f"  y_pred range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
                print(f"  PSNR_Y (60-view, crop 22): {psnr_v:.2f}")
                print(f"  SSIM_Y (60-view, crop 22): {ssim_v:.4f}")

    psnr_mean = float(np.mean(psnr_lst))
    ssim_mean = float(np.mean(ssim_lst))

    print(f"\n=== SADenseNet matched-protocol Y-only result ===")
    print(f"  Mean PSNR (Y, 60-view, crop {CROP}): {psnr_mean:.2f} dB")
    print(f"  Mean SSIM (Y, 60-view, crop {CROP}): {ssim_mean:.4f}")

    with open(args.out_csv, 'w') as f:
        f.write('sample,psnr_y_60view,ssim_y_60view\n')
        for i in range(n):
            f.write(f'{path2img_name(names[i])},{psnr_lst[i]:.4f},{ssim_lst[i]:.4f}\n')
        f.write(f'avg,{psnr_mean:.4f},{ssim_mean:.4f}\n')
    print(f"Saved per-scene CSV to {args.out_csv}")


if __name__ == "__main__":
    main()
