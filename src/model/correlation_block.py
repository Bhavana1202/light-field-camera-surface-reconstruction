"""PyTorch port of SADenseNet's CorrelationBlock (n_s spatial + n_a angular, spatial dense skip)."""
import torch
import torch.nn as nn
from model.sadense_layers import SpatialConv, AngularConv


class CorrelationBlock(nn.Module):
    def __init__(self, in_ch, flt=32, n_s=5, n_a=1, dense_s=True):
        super().__init__()
        self.n_s = n_s
        self.n_a = n_a
        self.dense_s = dense_s

        self.spatial = nn.ModuleList()
        cur = in_ch
        accum = 0
        for k in range(n_s):
            self.spatial.append(SpatialConv(cur, flt, kernel_size=3))
            accum += flt
            cur = accum if dense_s else flt

        ang_in = (n_s * flt) if dense_s else flt
        self.angular = nn.ModuleList()
        cur = ang_in
        for k in range(n_a):
            self.angular.append(AngularConv(cur, flt, kernel_size=3))
            cur = flt

    def forward(self, x):
        dense_list = []
        for k in range(self.n_s):
            x = self.spatial[k](x)
            if self.dense_s:
                dense_list.append(x)
                x = torch.cat(dense_list, dim=1) if len(dense_list) > 1 else dense_list[0]
        for k in range(self.n_a):
            x = self.angular[k](x)
        return x
