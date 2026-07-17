"""PyTorch port of SADenseNet's SpatialConv and AngularConv. Layout: (b, C, a, a, s, s)."""
import torch
import torch.nn as nn


class SpatialConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=None, activation=True):
        super().__init__()
        k = kernel_size
        p = (k // 2) if padding is None else padding
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, k, k),
                              stride=(1, 1, 1), padding=(0, p, p))
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        b, C, a1, a2, s1, s2 = x.shape
        x = x.reshape(b, C, a1 * a2, s1, s2)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        oC = x.shape[1]
        x = x.reshape(b, oC, a1, a2, s1, s2)
        return x


class AngularConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=None, activation=True):
        super().__init__()
        k = kernel_size
        p = (k // 2) if padding is None else padding
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(1, k, k),
                              stride=(1, 1, 1), padding=(0, p, p))
        self.act = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        b, C, a1, a2, s1, s2 = x.shape
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous().reshape(b, C, s1 * s2, a1, a2)
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        oC = x.shape[1]
        x = x.reshape(b, oC, s1, s2, a1, a2).permute(0, 1, 4, 5, 2, 3).contiguous()
        return x
