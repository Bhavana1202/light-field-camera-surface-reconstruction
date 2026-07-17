"""GPU memory probe for the hybrid at angular_out=8.
Tries decreasing patch sizes until one fits. Measures peak GPU memory.
"""
import torch
from types import SimpleNamespace
from model.model_lfasr import Net_LFASR

opt = SimpleNamespace(
    angular_in=2,
    angular_out=8,
    num_source=4,
    psv_range=4,
    psv_step=50,
    crop_size=6,
    layer_num=4,
)

assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")

# Build model once, reuse across patch-size attempts
torch.manual_seed(0)
model = Net_LFASR(opt).to(device).train()
ind_source = torch.tensor([[0, 7, 56, 63]], device=device)

patch_sizes_to_try = [64, 48, 32]
print(f"GPU: {torch.cuda.get_device_name(0)}, total mem {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

for ps in patch_sizes_to_try:
    print(f"\n=== trying patch_size={ps} ===")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        img_source = torch.rand(1, opt.num_source, ps, ps, device=device, requires_grad=False)

        # forward
        out = model(ind_source, img_source, opt)
        disp_lf, inter_lf, lf = out
        # fake loss & backward (uses train-time path)
        loss = lf.mean() + inter_lf.mean()
        loss.backward()

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  forward+backward OK")
        print(f"  shapes: disp_lf={tuple(disp_lf.shape)}, inter_lf={tuple(inter_lf.shape)}, lf={tuple(lf.shape)}")
        print(f"  peak GPU memory: {peak_mb:.0f} MB")
        print(f"  --> patch_size={ps} is FEASIBLE")
        break  # success, no need to try smaller
    except torch.cuda.OutOfMemoryError as e:
        print(f"  OOM at patch_size={ps}")
        # clear references before next attempt
        try: del img_source, out, disp_lf, inter_lf, lf, loss
        except Exception: pass
        # zero grads to free their tensors
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR (not OOM): {type(e).__name__}: {e}")
        raise
else:
    print("\nNONE of the tried patch sizes fit. Will need smaller psv_step or other reduction.")
