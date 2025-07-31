#!/usr/bin/env python3
# =========================================================
#  hybrid/test_hybrid.py
#  ---------------------------------------------------------
#  Evaluate a trained HybridLFRec checkpoint on HCI.
#  Outputs a CSV with per-scene PSNR and SSIM.
# =========================================================
import argparse, csv, math, os, time, types, h5py
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from hybrid.model import HybridLFRec
from hybrid.train_hybrid import HCIpatchDataset, make_opts   # reuse!

# -------------------- metrics ------------------------------
def psnr(pred, tgt):
    return (-10 * torch.log10(F.mse_loss(pred, tgt) + 1e-10)).item()

def ssim_np(pred, tgt):
    pred = pred.cpu().numpy()
    tgt  = tgt.cpu().numpy()
    return ssim(tgt, pred, data_range=1.0)

# -------------------- main --------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='path to hci_best.pth')
    ap.add_argument('--test_h5', required=True,
                    help='e.g. LFASR-FS-GAF/LFData/test_HCI.h5')
    ap.add_argument('--patch', type=int, default=128,
                    help='same crop size used in training')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--csv', default='results_hci_hybrid.csv')
    args = ap.parse_args()

    # ---- model -------------------------------------------
    print('loading checkpoint ?')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    opts = make_opts(K=4, A=7, patch=args.patch, psv_range=3, psv_step=20)
    model = HybridLFRec(opts).to(args.device)
    model.load_state_dict(ckpt['state'])
    model.eval()

    # ---- dataset (centre crop only ? determinstic) -------
    ds = HCIpatchDataset(args.test_h5, args.patch, train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1,
                                         shuffle=False, num_workers=2)

    # ---- evaluation -------------------------------------
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for idx, (sparse, gt) in enumerate(loader):
            sparse, gt = sparse.to(args.device), gt.to(args.device)
            dense, *_ = model(sparse)
            p  = psnr(dense, gt)
            s  = ssim_np(dense.squeeze(0), gt.squeeze(0))
            rows.append((idx, p, s))
            print(f'[{idx:02}]  PSNR {p:.2f}  SSIM {s:.4f}')

    # ---- CSV --------------------------------------------
    with open(args.csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['scene_idx', 'PSNR(dB)', 'SSIM'])
        w.writerows(rows)
        w.writerow(['mean',
                    np.mean([r[1] for r in rows]),
                    np.mean([r[2] for r in rows])])
    print(f'done in {time.time()-t0:.1f}s  -> {args.csv}')

if __name__ == '__main__':
    main()
