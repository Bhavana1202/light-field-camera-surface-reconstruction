# ------------------------------------------------------------------
# Find LFASR repo so we can import  model.model_lfasr -> Net_view
# ------------------------------------------------------------------
import importlib, pathlib, sys, os
try:
    from model.model_lfasr import Net_view          # ? normal case
except ModuleNotFoundError:
    # try ../LFASR or ../LFASR-FS-GAF relative to this file
    here = pathlib.Path(__file__).resolve().parent
    candidates = ['LFASR', 'LFASR-FS-GAF']
    for cand in candidates:
        repo = here.parent / cand
        if (repo / 'model').is_dir():
            sys.path.append(str(repo))              # expose as top-level
            break
    # re-attempt import (will raise if still missing)
    from model.model_lfasr import Net_view

# hybrid/model.py
# =========================================================
#  Hybrid LF Reconstruction:
#    LFASR Net_view  (coarse geometry)
#    ? SADense-style refinement
# =========================================================
import torch
import torch.nn as nn

from hybrid.layers_torch import DenseCorrelation, SADenseHead


class HybridLFRec(nn.Module):
    def __init__(self,
                 opt_lfasr,
                 a_out=7,
                 growth=32,
                 n_blocks=6,
                 n_S=5):
        super().__init__()
        self.opt_lfasr = opt_lfasr           # keep for forward()
        self.views = a_out ** 2              # V = 49

        # coarse reconstruction network from LFASR
        self.net_view = Net_view(opt_lfasr)

        # 1×1 compress conv: 49-chan coarse LF ? growth
        self.compress = nn.Conv3d(1, growth, 1)

        # SADense core
        self.sadense = DenseCorrelation(in_ch=growth,
                                        growth=growth,
                                        n_blocks=n_blocks,
                                        n_S=n_S)

        # residual head
        self.head = SADenseHead(in_ch=self.sadense.out_ch,
                                mid_ch=growth * 2)

    # -----------------------------------------------------
    def forward(self, sparse_in):
        """
        sparse_in : [B, 4, H, W]  (corner SAIs)
        returns   : dense, coarse, disp
        """
        B, _, H, W = sparse_in.shape
        device = sparse_in.device

        # ---- correct first argument: indices tensor (length 4) ----------
        ind_src = torch.tensor(self.opt_lfasr.ind_source,
                            dtype=torch.long,
                            device=device)     # [4]

        # ---- run coarse LFASR module ------------------------------------
        out = self.net_view(ind_src, sparse_in, self.opt_lfasr)

        # Net_view (train mode) returns (disp_lf, inter_lf)
        if isinstance(out, (list, tuple)) and len(out) == 2:
            disp, coarse = out
        else:                                     # some repos return only LF
            coarse = out
            disp   = None

        # ---- SADense refinement -----------------------------------------
        feat  = self.compress(coarse.unsqueeze(1))       # B,g,V,H,W
        feat  = self.sadense(feat)                       # B,C',V,H,W
        resid = self.head(feat)                          # B,V,H,W
        dense = coarse + resid
        return dense, coarse, disp
