from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Sequence
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.data.util import apply_mask_mapping


class TileDataset(Dataset):

    def __init__(
            self,
            tiles_dir: Path,
            manifest: Optional[List[Dict]] = None,
            img_subdir: str = "images",
            msk_subdir: str = "masks",
            joint_transform: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
            image_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            mask_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            dtype: torch.dtype = torch.float32,
            num_classes: Optional[int] = None,
            mask_value_mapping: Optional[Dict[int, int]] = None,
            binary_auto: bool = True,
            exts: Sequence[str] = (".png",),
    ) -> None:
        self.tiles_dir = Path(tiles_dir)
        self.img_dir = self.tiles_dir / img_subdir
        self.msk_dir = self.tiles_dir / msk_subdir
        self.joint_transform = joint_transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.dtype = dtype

        self.num_classes = num_classes
        self.mask_value_mapping = mask_value_mapping
        self.binary_auto = binary_auto
        self.exts = tuple(e.lower() for e in exts)

        if manifest is not None:
            self.pairs = [(Path(r["image_path"]), Path(r["mask_path"])) for r in manifest]
        else:
            img_files = []
            for ext in self.exts:
                img_files.extend(sorted(self.img_dir.glob(f"*{ext}")))
            msk_map = {}
            for ext in self.exts:
                for p in self.msk_dir.glob(f"*{ext}"):
                    msk_map[p.name] = p

            self.pairs: List[Tuple[Path, Path]] = []
            for ip in sorted(img_files):
                mp = msk_map.get(ip.name)
                if mp is not None:
                    self.pairs.append((ip, mp))

            if not self.pairs:
                raise FileNotFoundError(f"no tile pairs found in {self.img_dir} / {self.msk_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, msk_path = self.pairs[idx]

        img = Image.open(img_path).convert("RGB")  # (H,W,3), uint8
        img_np = np.array(img, dtype=np.uint8, copy=True)  # writeable
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(self.dtype) / 255.0  # [3,H,W], float

        msk = Image.open(msk_path)
        if msk.mode not in ("L", "P"):
            msk = msk.convert("L")
        msk_np = np.array(msk, copy=True)
        msk_np = apply_mask_mapping(msk_np, self.num_classes, self.mask_value_mapping, self.binary_auto)
        msk_t = torch.from_numpy(msk_np).unsqueeze(0)  # [1,H,W], long

        if self.joint_transform is not None:
            img_t, msk_t = self.joint_transform(img_t, msk_t)
        if self.image_transform is not None:
            img_t = self.image_transform(img_t)
        if self.mask_transform is not None:
            msk_t = self.mask_transform(msk_t)

        return img_t, msk_t
