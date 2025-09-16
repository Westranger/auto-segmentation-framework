from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch.nn as nn

from src.net.unet import UNet


@dataclass
class UNetFactory:
    in_channels: int
    num_classes: int

    def build(self, input_size: tuple[int, int], encoder_specs: List[Tuple[int, int]]) -> nn.Module:
        model = UNet(
            input_size=input_size,
            encoder_specs=encoder_specs,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            use_norm=True,
        )
        return model
