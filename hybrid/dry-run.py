# ------------------------------------------------------------------
#  hybrid/dry_run.py
#
#  Quick smoke-test for the HybridLFRec network.
#  Builds the model with a fully-featured dummy opts object,
#  pushes a random sparse LF through it, and prints tensor shapes.
# ------------------------------------------------------------------
import os, sys, types, torch

# --- make sure the project root is on PYTHONPATH -------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hybrid.model import HybridLFRec


# ------------------------------------------------------------------
# 1. minimal but complete option set for Net_view (LFASR)
# ------------------------------------------------------------------
def dummy_opts():
    import types, math
    opt = types.SimpleNamespace()
    opt.angular_in   = 2
    opt.num_source   = 4
    opt.angular_out  = 7
    opt.psv_range    = 3
    opt.psv_step     = 20
    opt.psv_planes   = opt.psv_range * opt.psv_step + 1
    opt.patch_size   = 64
    opt.crop_size    = 0
    opt.max_disp     = opt.psv_planes - 1
    opt.train_stage  = 'coarse'
    opt.isTrain      = False
    opt.use_confidence = True
    opt.arb_sample      = True          # free-form pattern enabled
    # -------- NEW: 4 corner views in the 7×7 grid ----------
    opt.ind_source     = [0, 6, 42, 48]
    return opt



# ------------------------------------------------------------------
# 2. build model and push a random batch through
# ------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt   = dummy_opts()
    model = HybridLFRec(opt).to(device).eval()

    B, K  = 1, opt.num_source
    H = W = opt.patch_size
    sparse = torch.rand(B, K, H, W, device=device)

    with torch.no_grad():
        dense, coarse, disp = model(sparse)

print('dense : ',  dense.shape,
      '  coarse :', coarse.shape,
      '  disp : ', disp.shape if disp is not None else 'None')
