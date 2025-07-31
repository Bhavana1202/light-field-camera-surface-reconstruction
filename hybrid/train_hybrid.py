#!/usr/bin/env python3
# ==============================================================
#  hybrid/train_hybrid.py   ?   final corrected version
# ==============================================================

import argparse, json, math, os, time, types, h5py
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from hybrid.model import HybridLFRec

# ------------------------- helpers ------------------------------
def psnr(pred, tgt):
    return (-10 * torch.log10(F.mse_loss(pred, tgt) + 1e-10)).item()

def smooth_grad(disp):
    if disp is None or disp.numel() == 0:
        return 0.0
    dy = torch.abs(disp[:, :, 1:] - disp[:, :, :-1]).mean()
    dx = torch.abs(disp[:, :, :, 1:] - disp[:, :, :, :-1]).mean()
    return (dx + dy)

# ------------------------- dataset ------------------------------
class HCIpatchDataset(Dataset):
    _K_IDX = [10, 16, 64, 70]                            # 4 corners of 7×7
    _DENSE_IDX = [r*9+c for r in range(1,8) for c in range(1,8)]

    def __init__(self, h5_path, patch=128, train=True):
        super().__init__()
        self.h5 = h5py.File(h5_path, 'r')
        key = '/LFI_ycbcr' if '/LFI_ycbcr' in self.h5 else '/LFI'
        self.lfi   = self.h5[key]
        self.color = (key == '/LFI_ycbcr')
        self.patch = patch
        self.train = train

    def __len__(self): return self.lfi.shape[0]

    def _crop(self, img):
        H, W = img.shape[-2:]
        p = self.patch
        if self.train:
            t = torch.randint(0, H-p+1, (1,)).item()
            l = torch.randint(0, W-p+1, (1,)).item()
        else:
            t, l = (H-p)//2, (W-p)//2
        return img[..., t:t+p, l:l+p]

    def __getitem__(self, idx):
        lf = self.lfi[idx]
        if self.color: lf = lf[..., 0]                   # keep Y only
        lf = torch.from_numpy(lf).float().div(255.)
        lf = lf.view(81, *lf.shape[-2:])                 # flatten 9×9 ? 81
        lf = self._crop(lf)                              # crop patch
        return lf[self._K_IDX], lf[self._DENSE_IDX]      # sparse, dense

    def close(self): self.h5.close()

# ------------------------- Net_view opts ------------------------
def make_opts(K, A, patch, psv_range, psv_step):
    opt = types.SimpleNamespace()
    opt.angular_in  = int(math.sqrt(K))
    opt.num_source  = K
    opt.angular_out = A
    opt.patch_size  = patch
    opt.crop_size   = 0
    opt.psv_range   = psv_range
    opt.psv_step    = psv_step
    opt.psv_planes  = int(psv_range * psv_step + 1)
    opt.max_disp    = opt.psv_planes - 1
    opt.train_stage = 'coarse'
    opt.isTrain     = False
    opt.use_confidence = True
    opt.arb_sample     = True
    opt.ind_source  = [0, 6, 42, 48]                     # 7×7 corners
    return opt

# ------------------------- training loop ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_h5', required=True)
    ap.add_argument('--val_h5',   required=True)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch',  type=int, default=2)
    ap.add_argument('--lr',     type=float, default=2e-4)
    ap.add_argument('--warm',   type=int, default=10)
    ap.add_argument('--patch',  type=int, default=128)
    ap.add_argument('--psv_range', type=int, default=3)
    ap.add_argument('--psv_step',  type=int, default=20)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--save_dir', default='./checkpoints')
    ap.add_argument('--device',   default='cuda')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'run_cfg.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    train_ds = HCIpatchDataset(args.train_h5, args.patch, train=True)
    val_ds   = HCIpatchDataset(args.val_h5,   args.patch, train=False)

    # ------------ corrected DataLoader calls -----------------
    train_ld = DataLoader(train_ds,
                          batch_size=args.batch,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)
    val_ld   = DataLoader(val_ds,
                          batch_size=1,
                          shuffle=False,
                          num_workers=2,
                          pin_memory=True)
    # ---------------------------------------------------------

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    opts = make_opts(K=4, A=7, patch=args.patch,
                     psv_range=args.psv_range, psv_step=args.psv_step)
    model = HybridLFRec(opts).to(device)
    optim = torch.optim.Adam(model.parameters(), args.lr)
    sched = CosineAnnealingLR(optim, args.epochs - args.warm, 1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = 0.0
    print(f'>> Training on HCI   (#train {len(train_ds)}, #val {len(val_ds)})')
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()

        if epoch == 1 and args.warm > 0:
            for p in model.net_view.parameters(): p.requires_grad_(False)
        if epoch == args.warm + 1:
            for p in model.net_view.parameters(): p.requires_grad_(True)

        losses, p_train = [], []
        for sparse, gt in train_ld:
            sparse, gt = sparse.to(device), gt.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                dense, coarse, disp = model(sparse)
                loss = (F.mse_loss(dense, gt) +
                        0.2 * F.mse_loss(coarse, gt) +
                        0.01 * smooth_grad(disp))
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update(); optim.zero_grad()
            losses.append(loss.item())
            p_train.append(psnr(dense.detach(), gt))
        if epoch > args.warm: sched.step()

        model.eval(); p_val = []
        with torch.no_grad():
            for sparse, gt in val_ld:
                sparse, gt = sparse.to(device), gt.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    dense, *_ = model(sparse)
                p_val.append(psnr(dense, gt))
        vpsnr = np.mean(p_val)

        print(f'E{epoch:03}  '
              f'loss {np.mean(losses):.4f}  '
              f'train {np.mean(p_train):.2f}dB  '
              f'val {vpsnr:.2f}dB  '
              f't {time.time()-t0:.1f}s')

        if vpsnr > best_val:
            best_val = vpsnr
            torch.save({'epoch': epoch,
                        'state': model.state_dict(),
                        'val_psnr': best_val},
                       os.path.join(args.save_dir, 'hci_best.pth'))
            print('   ? new best, saved.')

    train_ds.close(); val_ds.close()
    print(f'>> Done.  Best val PSNR {best_val:.2f} dB')

if __name__ == '__main__':
    main()
