from __future__ import annotations
from typing import Optional, Dict, Sequence
import numpy as np
from pathlib import Path


def apply_mask_mapping(
        arr: np.ndarray,
        num_classes: Optional[int],
        mapping: Optional[Dict[int, int]],
        binary_auto: bool,
) -> np.ndarray:
    """
    arr: mask as numpy array (H,W), arbitrary dtype
    - If mapping is given, remap exact values via table.
    - Else if num_classes==2 and binary_auto: map >0 -> 1 (tolerates 0/255, 0/100, ...).
    - Else if num_classes provided: validate range [0..num_classes-1], raise on mismatch.
    Returns int64 array.
    """
    if mapping is not None:
        out = np.zeros_like(arr, dtype=np.int64)
        for src, dst in mapping.items():
            out[arr == src] = int(dst)
        return out

    if num_classes == 2 and binary_auto:
        return (arr > 0).astype(np.int64)

    arr64 = arr.astype(np.int64, copy=False)
    if num_classes is not None:
        vmin, vmax = int(arr64.min()), int(arr64.max())
        if vmin < 0 or vmax >= num_classes:
            uniq = np.unique(arr64)
            raise ValueError(
                f"Mask labels out of range for num_classes={num_classes}. "
                f"Found min={vmin}, max={vmax}, unique (up to 20): {uniq[:20]}."
            )
    return arr64


def _collect_pairs(
        images_dir: Path,
        masks_dir: Path,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
        mask_exts: Sequence[str] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"),
        mask_suffixes: Sequence[str] = ("_mask", "-mask", "_label", "-label"),
        casefold: bool = True,
) -> list[tuple[Path, Path]]:
    def _norm_name(p: Path) -> str:
        s = p.stem
        if casefold:
            s = s.casefold()
        return s

    imgs = []
    for ext in image_exts:
        imgs.extend(images_dir.glob(f"*{ext}"))
    msks = []
    for ext in mask_exts:
        msks.extend(masks_dir.glob(f"*{ext}"))

    mask_map: dict[str, Path] = {}
    for mp in msks:
        base = _norm_name(mp)
        mask_map.setdefault(base, mp)
        for suf in mask_suffixes:
            if base.endswith(suf):
                base2 = base[: -len(suf)]
                mask_map.setdefault(base2, mp)

    pairs: list[tuple[Path, Path]] = []
    missing_imgs = []
    for ip in imgs:
        key = _norm_name(ip)
        mp = mask_map.get(key)
        if mp is not None:
            pairs.append((ip, mp))
        else:
            missing_imgs.append(ip.name)

    print(f"[Pairing] images={len(imgs)} masks={len(msks)} -> paired={len(pairs)}")
    if missing_imgs:
        print(f"[Pairing] no mask for {len(missing_imgs)} images (first 5): {missing_imgs[:5]}")

    if not pairs:
        raise FileNotFoundError(
            f"no image/mask pairs found. Check extensions/suffices. images_dir={images_dir} masks_dir={masks_dir}"
        )
    return pairs


def _jsonify(o):
    from pathlib import Path
    import numpy as np
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, dict):
        return {k: _jsonify(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonify(v) for v in o]
    return o
