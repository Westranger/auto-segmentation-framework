import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, use_norm: bool = True):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size has to be odd"
        pad = kernel_size // 2

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=pad, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=not use_norm),
        ]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
