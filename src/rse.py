import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict


class SwitchLayer(nn.Module):
    def __init__(self, m, r=0.9):
        super().__init__()
        self.unit = nn.Sequential(
            nn.Linear(2*m, 4*m, bias=False),
            nn.LayerNorm(4*m),
            nn.GELU(),
            nn.Linear(4*m, 2*m, bias=True)
        )

        self.s = nn.Parameter(torch.empty(2*m))
        self.h = nn.Parameter(torch.empty(1))

        nn.init.constant_(self.s, np.log(r / (1 - r)))
        nn.init.constant_(self.h, 0.25*np.sqrt(1 - r**2))

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1] // 2, x.shape[2] * 2)
        c = self.unit(x)
        out = torch.mul(torch.sigmoid(self.s), x) + torch.mul(self.h, c)
        return out.view(out.shape[0], out.shape[1] * 2, out.shape[2] // 2)


class ShuffleLayer(nn.Module):
    def __init__(self, reverse=False):
        super().__init__()
        self.reverse = reverse

    @staticmethod
    def ror(x, n, p=1):
        """Bitwise rotation right p positions
        n is the bit length of the number
        """
        return (x >> p) + ((x & ((1 << p) - 1)) << (n - p))

    @staticmethod
    def rol(x, n, p=1):
        """Bitwise rotation left p positions
        n is the bit length of the number
        """
        return ((x << p) & ((1 << n) - 1)) | (x >> (n - p))

    def forward(self, x):
        length = x.shape[1]
        n_bits = (length - 1).bit_length()
        if self.reverse:
            rev_indices = [self.ror(i, n_bits) for i in range(length)]
        else:
            rev_indices = [self.rol(i, n_bits) for i in range(length)]
        return x[..., rev_indices, :]


class BenesBlock(nn.Module):
    def __init__(self, m, r=0.9):
        super().__init__()
        self.regular_switch = SwitchLayer(m, r)
        self.regular_shuffle = ShuffleLayer(reverse=False)
        self.reverse_switch = SwitchLayer(m, r)
        self.reverse_shuffle = ShuffleLayer(reverse=True)

    def forward(self, x):
        k = x.shape[1].bit_length()
        for _ in range(k-1):
            x = self.regular_switch(x)
            x = self.regular_shuffle(x)
        for _ in range(k-1):
            x = self.regular_switch(x)
            x = self.reverse_shuffle(x)
        return x


class ResidualShuffleExchangeNetwork(nn.Module):
    def __init__(self, m, n_blocks=1, r=0.9):
        super().__init__()
        self.blocks = nn.Sequential(
            OrderedDict({f"benes_block_{i}": BenesBlock(m, r) for i in range(n_blocks)})
        )
        self.final_switch = SwitchLayer(m, r)

    def forward(self, x):
        n = 1 << (x.shape[1] - 1).bit_length()
        x = F.pad(x, (0, 0, 0, n-x.shape[1]))
        x = self.blocks(x)
        x = self.final_switch(x)
        return x
