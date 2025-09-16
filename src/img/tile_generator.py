from __future__ import annotations
from typing import Iterable, Tuple, List
import csv
import shutil
from pathlib import Path

from PIL import Image
import numpy as np

from src.img.tile_generator_config import TileGeneratorConfig

SUPPORTED_IMGS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class TileGenerator:

    def __init__(self, cfg: TileGeneratorConfig) -> None:
        self.cfg = cfg
        self.cfg.validate()

    def run(self) -> None:
        self._prepare_out()
        pairs = self._pair_images_and_masks()
        stride = max(1, int(round(self.cfg.patch_size * (1.0 - self.cfg.overlap_pct))))

        for img_path, msk_path in pairs:
            stem = img_path.stem
            img = Image.open(img_path).convert("RGB")
            msk = Image.open(msk_path)

            if msk.mode not in ("L", "P"):
                msk = msk.convert("L")
            img_np = np.asarray(img)  # (H,W,3), uint8
            msk_np = np.asarray(msk)  # (H,W),   uint8/uint16 depends on src

            H, W = img_np.shape[:2]
            if msk_np.shape[0] != H or msk_np.shape[1] != W:
                raise ValueError(f"Size mismatch image vs mask for stem '{stem}': {img_np.shape} vs {msk_np.shape}")

            S = self.cfg.patch_size
            for y in range(0, max(0, H - S + 1), stride):
                for x in range(0, max(0, W - S + 1), stride):
                    img_patch = img_np[y:y + S, x:x + S, :]
                    msk_patch = msk_np[y:y + S, x:x + S]

                    if img_patch.shape[0] != S or img_patch.shape[1] != S:
                        continue
                    if msk_patch.shape[0] != S or msk_patch.shape[1] != S:
                        continue

                    img_out = Image.fromarray(img_patch)
                    msk_out = Image.fromarray(msk_patch)

                    img_name = f"{stem}_y{y}_x{x}.png"
                    msk_name = f"{stem}_y{y}_x{x}.png"
                    img_out.save(self.cfg.out_dir / "images" / img_name)
                    msk_out.save(self.cfg.out_dir / "masks" / msk_name)

    def _prepare_out(self) -> None:
        if self.cfg.out_dir.exists():
            shutil.rmtree(self.cfg.out_dir)
        (self.cfg.out_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.cfg.out_dir / "masks").mkdir(parents=True, exist_ok=True)

    def _pair_images_and_masks(self) -> Iterable[Tuple[Path, Path]]:
        imgs = self._collect(self.cfg.images_dir)
        msks = self._collect(self.cfg.masks_dir)
        msk_map = {p.stem: p for p in msks}

        pairs = []
        for img in imgs:
            stem = img.stem
            if stem not in msk_map:
                raise FileNotFoundError(f"No matching mask for image stem '{stem}'")
            pairs.append((img, msk_map[stem]))
        return pairs

    @staticmethod
    def _collect(root: Path) -> list[Path]:
        files = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMGS:
                files.append(p)
        files.sort()
        return files

    def list_image_tiles(self) -> List[Path]:
        p = self.cfg.out_dir / "images"
        return sorted(p.glob("*.png")) if p.exists() else []

    def list_mask_tiles(self) -> List[Path]:
        p = self.cfg.out_dir / "masks"
        return sorted(p.glob("*.png")) if p.exists() else []

    def iter_tile_pairs(self) -> Iterable[Tuple[Path, Path]]:
        img_dir = self.cfg.out_dir / "images"
        msk_dir = self.cfg.out_dir / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            return
        msk_map = {p.name: p for p in self.list_mask_tiles()}
        for img_p in self.list_image_tiles():
            msk_p = msk_map.get(img_p.name)
            if msk_p is not None:
                yield (img_p, msk_p)

    def build_manifest(self, save_csv: bool = True, filename: str = "manifest.csv") -> List[dict]:
        rows: List[dict] = []
        for img_p, msk_p in self.iter_tile_pairs():
            name = img_p.stem
            y = x = None
            parts = name.split("_")
            try:
                for part in parts:
                    if part.startswith("y") and part[1:].isdigit():
                        y = int(part[1:])
                    if part.startswith("x") and part[1:].isdigit():
                        x = int(part[1:])
            except Exception:
                pass

            rows.append({
                "original_stem": "_".join(parts[:-2]) if y is not None and x is not None else parts[0],
                "y": y,
                "x": x,
                "image_path": str(img_p),
                "mask_path": str(msk_p),
            })

        if save_csv:
            out_csv = self.cfg.out_dir / filename
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["original_stem", "y", "x", "image_path", "mask_path"])
                writer.writeheader()
                writer.writerows(rows)
        return rows
