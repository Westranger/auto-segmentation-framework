from typing import List, Tuple
import torch
import torch.nn as nn

from src.net.double_conv import DoubleConv


class UNet(nn.Module):
    def __init__(
            self,
            input_size: Tuple[int, int],
            encoder_specs: List[Tuple[int, int]],
            in_channels: int = 3,
            num_classes: int = 1,
            use_norm: bool = True,
    ):
        super().__init__()
        assert len(encoder_specs) >= 1, "at least one encoder-block required"

        H, W = input_size
        num_pools = len(encoder_specs) - 1
        div = 2 ** num_pools
        assert H % div == 0 and W % div == 0, (
            f"input_size {input_size} has to be dividable by {div} (=2**{num_pools})"
        )

        assert (H // div) >= 1 and (W // div) >= 1, "too many block for the given inputsize"
        for ks, _ in encoder_specs:
            assert ks % 2 == 1, "kernel_size has to be odd (3/5/7/...)."

        self.input_size = input_size
        self.encoder_specs = encoder_specs
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_norm = use_norm

        # --- Encoder ---
        enc_blocks = []
        pools = []
        prev_ch = in_channels
        self.enc_out_ch = []
        for i, (ks, out_ch) in enumerate(encoder_specs):
            enc_blocks.append(DoubleConv(prev_ch, out_ch, kernel_size=ks, use_norm=use_norm))
            self.enc_out_ch.append(out_ch)
            prev_ch = out_ch
            if i < num_pools:
                pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encoder = nn.ModuleList(enc_blocks)
        self.pools = nn.ModuleList(pools)

        # --- Decoder ---
        curr_ch = self.enc_out_ch[-1]
        upconvs = []
        dec_blocks = []
        for i in reversed(range(len(encoder_specs) - 1)):
            ks_i, out_ch_i = encoder_specs[i]
            upconvs.append(nn.ConvTranspose2d(curr_ch, out_ch_i, kernel_size=2, stride=2))
            dec_blocks.append(DoubleConv(in_ch=2 * out_ch_i, out_ch=out_ch_i, kernel_size=ks_i, use_norm=use_norm))
            curr_ch = out_ch_i
        self.upconvs = nn.ModuleList(upconvs)
        self.decoder = nn.ModuleList(dec_blocks)

        self.head = nn.Conv2d(self.enc_out_ch[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for i, enc in enumerate(self.encoder):
            h = enc(h)
            if i < len(self.pools):
                skips.append(h)
                h = self.pools[i](h)

        # Decoder
        for up, dec, skip in zip(self.upconvs, self.decoder, reversed(skips)):
            h = up(h)
            assert h.shape[-2:] == skip.shape[-2:], (
                f"shape-mismatch im decoder: up={tuple(h.shape[-2:])} vs skip={tuple(skip.shape[-2:])}. "
                "check input_size & encoder_specs."
            )
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        return self.head(h)
