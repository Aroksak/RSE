import torch.nn as nn

from rse import ResidualShuffleExchangeNetwork


class MusicNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=4, stride=2, bias=False),
            nn.GroupNorm(1, 96),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 384, kernel_size=4, stride=2, bias=False),
            nn.GroupNorm(1, 384),
            nn.GELU()
        )

        self.pre_linear = nn.Linear(384, 192)

        self.rse = ResidualShuffleExchangeNetwork(192, n_blocks=2)

        self.post_linear = nn.Linear(192, 128)

    def forward(self, x):
        x = self.conv1(x)   # B x C x L
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # B x L x C
        x = self.pre_linear(x)
        x = self.rse(x)
        x = self.post_linear(x)
        return x
