"""GPU memory probe for the hybrid at angular_out=7."""
import torch
from types import SimpleNamespace
from model.model_lfasr import Net_LFASR

opt = SimpleNamespace(
    angular_in=2,
    angular_out=7,
    num_source=4,
    psv_range=4,
    psv_step=50,
    crop_size=6,
    layer_num=4,
)

assert torch.cuda.is_available(), "CUDA not available"
device = torch.device("cuda")

torch.manual_seed(0)
model = Net_LFASR(opt).to(device).train()
ind_source = torch.tensor([[0, 6, 42, 48]], device=device)   # 2x2 corners of 7x7

patch_sizes_to_try = [64, 48, 32]
print(f"GPU: {torch.cuda.get_device_name(0)}, total {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

for ps in patch_sizes_to_try:
    print(f"\n=== patch_size={ps}, angular_out=7 ===")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        img_source = torch.rand(1, opt.num_source, ps, ps, device=device, requires_grad=False)

        out = model(ind_source, img_source, opt)
        disp_lf, inter_lf, lf = out
        loss = lf.mean() + inter_lf.mean()
        loss.backward()

        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  forward+backward OK")
        print(f"  shapes: disp_lf={tuple(disp_lf.shape)}, inter_lf={tuple(inter_lf.shape)}, lf={tuple(lf.shape)}")
        print(f"  peak GPU: {peak_mb:.0f} MB")
        print(f"  -> patch_size={ps} FEASIBLE")
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at patch_size={ps}")
        for p in model.parameters():
            if p.grad is not None: p.grad = None
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        raise
