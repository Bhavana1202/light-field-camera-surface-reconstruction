"""SADenseRefine: drop-in replacement for FS-GAF's Net_refine."""
import torch
import torch.nn as nn
from model.dense_correlation import DenseCorrelation
from model.sadense_layers import SpatialConv


class SADenseRefine(nn.Module):
    def __init__(self, an=7, flt=32, n_blocks=6, n_s=5, n_a=1,
                 dense_s=True, dense_a=True, dense_i=True):
        super().__init__()
        self.an = an
        self.an2 = an * an
        self.dense = DenseCorrelation(in_ch=1, flt=flt, n_blocks=n_blocks,
                                      n_s=n_s, n_a=n_a,
                                      dense_s=dense_s, dense_a=dense_a, dense_i=dense_i)
        agg_ch = self.dense.out_channels
        self.reduce0 = SpatialConv(agg_ch, 64, kernel_size=3)
        self.reduce1 = SpatialConv(64, 64, kernel_size=3)
        self.reduce2 = SpatialConv(64, 1, kernel_size=3, activation=False)

    def forward(self, inter_lf):
        N, an2, h, w = inter_lf.shape
        assert an2 == self.an2, f"expected {self.an2} views, got {an2}"
        x = inter_lf.reshape(N, 1, self.an, self.an, h, w)
        x = self.dense(x)
        x = self.reduce0(x)
        x = self.reduce1(x)
        x = self.reduce2(x)
        res = x.reshape(N, self.an2, h, w)
        return inter_lf + res


if __name__ == "__main__":
    for an in (7, 8):
        torch.manual_seed(0)
        N, h, w = 2, 16, 16
        inter_lf = torch.randn(N, an * an, h, w)
        refine = SADenseRefine(an=an)
        lf = refine(inter_lf)
        assert tuple(lf.shape) == (N, an * an, h, w), (an, lf.shape)
        assert torch.isfinite(lf).all()
        lf.mean().backward()
        assert refine.dense.blocks[0].spatial[0].conv.weight.grad is not None
        nparams = sum(p.numel() for p in refine.parameters())
        print(f"an={an}: out={tuple(lf.shape)}  params={nparams:,}  OK")
    print("ALL SADENSE REFINE TESTS PASSED (an=7 and an=8)")
