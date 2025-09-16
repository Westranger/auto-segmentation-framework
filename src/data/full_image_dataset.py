from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Sequence, Dict, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.util import apply_mask_mapping


class FullImageDataset(Dataset):

    def __init__(
            self,
            images_dir: Path,
            masks_dir: Path,
            *,
            num_classes: Optional[int] = None,
            mask_value_mapping: Optional[Dict[int, int]] = None,
            binary_auto: bool = True,
            dtype: torch.dtype = torch.float32,
            exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.num_classes = num_classes
        self.mask_value_mapping = mask_value_mapping
        self.binary_auto = binary_auto
        self.dtype = dtype
        self.exts = tuple(e.lower() for e in exts)

        img_map: Dict[str, Path] = {}
        for ext in self.exts:
            for p in self.images_dir.rglob(f"*{ext}"):
                img_map[p.stem] = p

        msk_map: Dict[str, Path] = {}
        for ext in self.exts:
            for p in self.masks_dir.rglob(f"*{ext}"):
                msk_map[p.stem] = p

        stems = sorted(set(img_map.keys()) & set(msk_map.keys()))
        if not stems:
            raise FileNotFoundError(f"No proper pairs found in {self.images_dir} and {self.masks_dir}")

        self.pairs: List[Tuple[str, Path, Path]] = [(s, img_map[s], msk_map[s]) for s in stems]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        stem, img_path, msk_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.uint8, copy=True)  # [H,W,3]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(self.dtype) / 255.0  # [C,H,W] float

        msk = Image.open(msk_path)
        if msk.mode not in ("L", "P"):
            msk = msk.convert("L")
        msk_np = np.array(msk, copy=True)  # [H,W]
        msk_np = apply_mask_mapping(
            msk_np,
            num_classes=self.num_classes,
            mapping=self.mask_value_mapping,
            binary_auto=self.binary_auto,
        )
        msk_t = torch.from_numpy(msk_np).unsqueeze(0)  # [1,H,W] long

        return img_t, msk_t, stem
