from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_IMGS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class TileGeneratorConfig:
    images_dir: Path
    masks_dir: Path
    out_dir: Path
    patch_size: int
    overlap_pct: float  # 0.0 .. <1.0

    def validate(self) -> None:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"masks_dir not found: {self.masks_dir}")
        if not (0.0 <= self.overlap_pct < 1.0):
            raise ValueError("overlap_pct must be in [0.0, 1.0).")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be > 0.")
