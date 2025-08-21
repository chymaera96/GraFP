import torch
import torch.nn as nn

class NAFPEncoder(nn.Module):
    def __init__(self, in_channels=1, kernel_size=3):
        super().__init__()
        k = kernel_size
        d, h = 128, 1024
        channels = [d, d, 2*d, 2*d, 4*d, 4*d, h, h]

        strides = [
            ((1,2),(2,1)),
            ((1,2),(2,1)),
            ((1,2),(2,1)),
            ((1,2),(2,1)),
            ((1,1),(2,1)),
            ((1,2),(2,1)),
            ((1,1),(2,1)),
            ((1,2),(2,1)),
        ]


        layers, in_ch = [], in_channels
        for out_ch, (st1, st2) in zip(channels, strides):
            # --- first conv: (1 x k), stride st1 ---
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=(1, k), stride=st1, padding=(0,1)))
            layers.append(nn.ELU())
            layers.append(nn.GroupNorm(1, out_ch))

            # --- second conv: (k x 1), stride st2 ---
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=(k, 1), stride=st2, padding=(1,0)))
            layers.append(nn.ELU())
            layers.append(nn.GroupNorm(1, out_ch))

            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, F, T)
        x = self.conv(x)      # (B, C, F, T)
        return self.flatten(x)
