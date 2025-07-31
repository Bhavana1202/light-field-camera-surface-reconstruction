# =========================================================
#  PyTorch re-implementation of SADenseNet building blocks
# =========================================================
import math, torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint

# ---------- 3×3 Conv on spatial plane (runs on each SAI) ----------
class SpatialConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):                              # B,C,N,H,W
        B,C,N,H,W = x.shape
        x = self.op(x.view(B*N, C, H, W))
        C2 = x.shape[1]
        return x.view(B, C2, N, H, W)

# ---------- 3×3 Conv on angular grid (runs per pixel) ----------
class AngularConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):                              # B,C,N,H,W
        B,C,N,H,W = x.shape
        A = int(math.sqrt(N)); assert A*A == N
        x = x.view(B, C, A, A, H, W)                   # B,C,A,A,H,W
        x = x.permute(0,4,5,1,2,3).contiguous()        # B,H,W,C,A,A
        x = self.op(x.view(B*H*W, C, A, A))            # conv
        C2 = x.shape[1]
        x = x.view(B, H, W, C2, A, A).permute(0,3,4,5,1,2).contiguous()
        return x.view(B, C2, N, H, W)

# ---------- one Correlation Block (n_S spatial ? 1 angular) -------
class CorrelationBlock(nn.Module):
    def __init__(self, in_ch, growth, n_S=5, dense_S=True):
        super().__init__()
        self.dense_S = dense_S
        self.spatial = nn.ModuleList()
        ch = in_ch
        for _ in range(n_S):
            self.spatial.append(SpatialConv(ch, growth))
            ch = ch + growth if dense_S else growth
        self.angular = AngularConv(ch, growth)
        self.out_ch  = growth
    def forward(self, x):
        feat = x
        for sc in self.spatial:
            new = sc(feat)
            feat = torch.cat([feat, new], 1) if self.dense_S else new
        return self.angular(feat)

# ---------- stack of Correlation Blocks with A-links -------------
class DenseCorrelation(nn.Module):
    def __init__(self, in_ch, growth=32, n_blocks=6,
                 n_S=5, dense_S=True, dense_A=True):
        super().__init__()
        self.dense_A = dense_A
        self.blocks  = nn.ModuleList()
        ch = in_ch
        for _ in range(n_blocks):
            blk = CorrelationBlock(ch, growth, n_S, dense_S)
            self.blocks.append(blk)
            ch = ch + blk.out_ch if dense_A else blk.out_ch
        self.out_ch = ch
    def forward(self, x):
        feat = x
        for blk in self.blocks:
            out = checkpoint(blk, feat)
            feat = torch.cat([feat, out], 1) if self.dense_A else out
        return feat

# ---------- 1×1 Conv head returning residual ----------------------
class SADenseHead(nn.Module):
    def __init__(self, in_ch, mid_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, 1), nn.ReLU(True),
            nn.Conv3d(mid_ch, 1, 1))          # 1 chan per view
    def forward(self, x):                     # B,C,N,H,W  ?  B,N,H,W
        return self.head(x).squeeze(1)
