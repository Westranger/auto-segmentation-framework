from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class InMemRepSelectorConfig:
    images_dir: Path
    masks_dir: Path
    patch_size: int
    overlap_pct: float = 0.25  # 0..1
    target_k: Optional[int] = None  # if None -> k auto
    min_k: int = 50  # if target_k=None
    max_k: int = 200  # if target_k=None
    device: str = "cuda"
    similarity_prune: float = 0.995

    # output
    index_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    write_manifest: Optional[Path] = None

    # performance
    batch_embed: int = 128
    pretrained: bool = True
    progress: bool = True
    mask_mapping: Optional[Dict[int, int]] = None
    num_classes_override: Optional[int] = None

    prefilter_by_entropy: bool = True
    prefilter_entropy_quantile: float = 0.2
    max_tiles_per_image: int = 0

    dedupe_method: str = "lsh"
    lsh_bits: int = 64
    lsh_bands: int = 4
    lsh_band_bits: int = 16
    lsh_bucket_limit: int = 5000

    # re-use index/embeddings
    recompute_embeddings: bool = False
