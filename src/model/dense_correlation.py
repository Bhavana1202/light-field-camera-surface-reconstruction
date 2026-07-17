"""PyTorch port of SADenseNet's DenseCorrelation: n_blocks blocks + angular & image skips."""
import torch
import torch.nn as nn
from model.correlation_block import CorrelationBlock


class DenseCorrelation(nn.Module):
    def __init__(self, in_ch=1, flt=32, n_blocks=6, n_s=5, n_a=1,
                 dense_s=True, dense_a=True, dense_i=True):
        super().__init__()
        self.n_blocks = n_blocks
        self.flt = flt
        self.dense_a = dense_a
        self.dense_i = dense_i

        self.blocks = nn.ModuleList()
        cur_in = in_ch
        list_ch = [in_ch] if dense_i else []
        for i in range(n_blocks):
            self.blocks.append(CorrelationBlock(in_ch=cur_in, flt=flt,
                                                n_s=n_s, n_a=n_a, dense_s=dense_s))
            if dense_a:
                list_ch.append(flt)
            else:
                list_ch = [flt] + ([in_ch] if dense_i else [])
            cur_in = sum(list_ch) if len(list_ch) > 1 else list_ch[0]

        self.out_channels = cur_in

    def forward(self, x):
        inp = x
        dense_list = [inp] if self.dense_i else []
        for i in range(self.n_blocks):
            F = self.blocks[i](x)
            if self.dense_a:
                dense_list.append(F)
            else:
                dense_list = [F] + ([inp] if self.dense_i else [])
            x = torch.cat(dense_list, dim=1) if len(dense_list) > 1 else dense_list[0]
        return x
