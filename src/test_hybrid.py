"""Evaluate the hybrid (FS-GAF Net_view + SADenseRefine) on Kalantari test set."""
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
import os
from os.path import join
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim

from utils import dataset, util
from model.model_lfasr import Net_LFASR


def build_parser():
    p = argparse.ArgumentParser("Evaluate the hybrid on Kalantari test set.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--test_path", type=str, default="./LFData/Kalantari_test_fsgaf.h5")
    p.add_argument("--save_dir", type=str, default="results_hybrid")
    p.add_argument("--angular_in", type=int, default=2)
    p.add_argument("--angular_out", type=int, default=8)
    p.add_argument("--arb_sample", type=int, default=0)
    p.add_argument("--input_ind", action=util.Store_as_array, type=int, nargs='+',
                   default=np.array([0, 7, 56, 63]))
    p.add_argument("--psv_range", type=int, default=4)
    p.add_argument("--psv_step", type=int, default=50)
    p.add_argument("--layer_num", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=22)
    p.add_argument("--test_crop_size", type=int, default=0)
    p.add_argument("--single_scene", type=int, default=-1)
    p.add_argument("--save_img", type=int, default=0)
    return p


def main():
    opt = build_parser().parse_args()
    print(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.num_source = opt.angular_in if opt.arb_sample else opt.angular_in * opt.angular_in

    print("building net")
    model = Net_LFASR(opt).to(device)

    print(f"loading checkpoint {opt.model_path}")
    ckpt = torch.load(opt.model_path, map_location=device)
    pt_dict = ckpt['model']

    missing, unexpected = model.load_state_dict(pt_dict, strict=False)
    print(f"  missing keys: {len(missing)} (should be 0)")
    print(f"  unexpected keys: {len(unexpected)} (should be 0)")
    if missing: print("  first missing:", missing[:3])
    if unexpected: print("  first unexpected:", unexpected[:3])
    if missing or unexpected:
        raise RuntimeError("checkpoint key mismatch -- aborting")
    print("  all keys loaded correctly")

    test_set = dataset.TestDataFromHdf5(opt.test_path, opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print(f"loaded {len(test_loader)} test LFIs from {opt.test_path}")

    os.makedirs(opt.save_dir, exist_ok=True)
    csv_path = join(opt.save_dir, f"hybrid_eval_{os.path.basename(opt.model_path).replace('.pth','')}.csv")

    model.net_view.eval()
    model.net_refine.eval()

    rows = []
    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            if opt.single_scene >= 0 and k != opt.single_scene:
                continue
            input_views, target_y, lfi_ycbcr = batch[0], batch[1].numpy(), batch[2].numpy()
            input_views = input_views.to(device)

            inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input_views, opt)
            # release Net_view's intermediate PSV memory before refinement
            torch.cuda.empty_cache()

            # width-strip processing with smaller strips + per-strip cleanup
            length = 90
            crop = 20
            input_l, input_m, input_r = util.CropPatches_w(inter_lf, length, crop)

            def _refine_strip(x):
                with torch.no_grad():
                    y = model.net_refine(x).detach()
                torch.cuda.empty_cache()
                return y

            pred_l = _refine_strip(input_l)
            pred_m = torch.empty(input_m.shape[0], opt.angular_out * opt.angular_out,
                                 input_m.shape[2], input_m.shape[3], device=device)
            for i in range(input_m.shape[0]):
                pred_m[i:i + 1] = _refine_strip(input_m[i:i + 1])
            pred_r = _refine_strip(input_r)

            pred_y = util.MergePatches_w(pred_l, pred_m, pred_r,
                                         inter_lf.shape[2], inter_lf.shape[3],
                                         length, crop)
            pred_y = util.crop_boundary(pred_y, opt.test_crop_size).cpu().numpy()

            # free the big intermediates before next scene
            del inter_lf, input_l, input_m, input_r, pred_l, pred_m, pred_r
            torch.cuda.empty_cache()

            bd = opt.crop_size + opt.test_crop_size
            target_yc = target_y[:, :, bd:-bd, bd:-bd]

            an2 = opt.angular_out * opt.angular_out
            psnrs_all, ssims_all, psnrs_syn, ssims_syn = [], [], [], []
            for i in range(an2):
                t = target_yc[0, i]; p = pred_y[0, i]
                psnr = util.compt_psnr(t, p)
                ssim = compare_ssim((t * 255.0).astype(np.uint8),
                                    (p * 255.0).astype(np.uint8),
                                    gaussian_weights=True, sigma=1.5,
                                    use_sample_covariance=False)
                psnrs_all.append(psnr); ssims_all.append(ssim)
                if i not in opt.input_ind:
                    psnrs_syn.append(psnr); ssims_syn.append(ssim)

            row = {
                'scene': k,
                'psnr_60view_synthesis': float(np.mean(psnrs_syn)),
                'ssim_60view_synthesis': float(np.mean(ssims_syn)),
                'psnr_64view_all':       float(np.mean(psnrs_all)),
                'ssim_64view_all':       float(np.mean(ssims_all)),
            }
            rows.append(row)
            print(f"  scene {k:2d}: PSNR 60-view = {row['psnr_60view_synthesis']:.2f}  "
                  f"PSNR 64-view = {row['psnr_64view_all']:.2f}  "
                  f"SSIM 60-view = {row['ssim_60view_synthesis']:.4f}")

            if opt.single_scene >= 0:
                break

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.loc['mean'] = ['avg',
                          df['psnr_60view_synthesis'].mean(),
                          df['ssim_60view_synthesis'].mean(),
                          df['psnr_64view_all'].mean(),
                          df['ssim_64view_all'].mean()]
    df.to_csv(csv_path, index=False)
    print(f"\nresults saved to {csv_path}")
    if len(rows) > 0:
        print(f"FINAL: mean PSNR 60-view (vs SADenseNet 40.31): "
              f"{np.mean([r['psnr_60view_synthesis'] for r in rows]):.2f} dB")
        print(f"FINAL: mean PSNR 64-view (FS-GAF convention):    "
              f"{np.mean([r['psnr_64view_all'] for r in rows]):.2f} dB")


if __name__ == "__main__":
    main()
