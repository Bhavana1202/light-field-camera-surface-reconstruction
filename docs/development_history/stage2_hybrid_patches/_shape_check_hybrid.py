"""End-to-end shape check for hybrid Net_LFASR at angular_out=8 (training path, CPU, tiny input)."""
import torch
from types import SimpleNamespace
from model.model_lfasr import Net_LFASR

opt = SimpleNamespace(
    angular_in=2,
    angular_out=8,
    num_source=4,
    psv_range=4,
    psv_step=50,
    crop_size=6,        # training path uses crop_boundary; FS-GAF default-ish
    layer_num=4,
)

torch.manual_seed(0)
model = Net_LFASR(opt)
model.train()   # training path: Net_view returns (disp_target, inter_lf)

# tiny fake input. NOTE training path builds a big PSV, so keep spatial small.
N, h, w = 1, 32, 32
img_source = torch.rand(N, opt.num_source, h, w)
ind_source = torch.tensor([[0, 7, 56, 63]])   # 2x2 corners of 8x8 grid

print("running forward (training path, CPU)... this builds the PSV, may take a moment")
out = model(ind_source, img_source, opt)

print("returned", len(out), "tensors:")
names = ["disp_lf", "inter_lf", "lf"]
for nm, t in zip(names, out):
    print(f"  {nm}: {tuple(t.shape)}")

# sanity: lf should be (N, 64, h_cropped, w_cropped)
lf = out[2]
expected_views = opt.angular_out * opt.angular_out
assert lf.shape[0] == N and lf.shape[1] == expected_views, f"unexpected lf shape {tuple(lf.shape)}"
print(f"OK: final lf has {expected_views} views as expected for 8x8")
print("HYBRID FORWARD (training path) RAN SUCCESSFULLY")
