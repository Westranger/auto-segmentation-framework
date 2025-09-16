from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Optional
import json

from PIL import Image


class RepresentativeMaterializer:

    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        for fn in ["config.json", "meta.jsonl", "embeddings.npy", "features.npy", "clusters.json"]:
            if not (self.index_dir / fn).exists():
                raise FileNotFoundError(f"Index unvollständig, fehlt: {fn}")
        with open(self.index_dir / "clusters.json", "r", encoding="utf-8") as f:
            self.cl = json.load(f)
        self.meta: List[Dict] = []
        with open(self.index_dir / "meta.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.meta.append(json.loads(line))

    def top_medoid_indices(self, n: int) -> List[int]:
        order = self.cl["medoid_order"]
        meds = self.cl["medoid_indices"]
        ranked = [int(meds[i]) for i in order]
        return ranked[:min(n, len(ranked))]

    def materialize(self, n: int, out_dir: Path, write_manifest: Optional[Path] = None) -> List[Dict]:
        out_dir = Path(out_dir)
        out_img = out_dir / "images";
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk = out_dir / "masks";
        out_msk.mkdir(parents=True, exist_ok=True)

        idxs = self.top_medoid_indices(n)
        selected: List[Dict] = []
        for i in idxs:
            rec = self.meta[i]
            ip = Path(rec["src_image"]);
            mp = Path(rec["src_mask"])
            x, y, w, h = rec["x"], rec["y"], rec["w"], rec["h"]
            img = Image.open(ip).convert("RGB").crop((x, y, x + w, y + h))
            msk = Image.open(mp)
            if msk.mode not in ("L", "P"):
                msk = msk.convert("L")
            msk = msk.crop((x, y, x + w, y + h))
            name = f"{Path(ip).stem}_x{x}_y{y}_w{w}_h{h}.png"
            img.save(out_img / name)
            msk.save(out_msk / name)
            out_rec = {
                **rec,
                "image_path": str(out_img / name),
                "mask_path": str(out_msk / name),
                "tile_index": int(i),
                "cluster_id": int(self.cl["assign"][i]),
            }
            selected.append(out_rec)

        if write_manifest is not None:
            write_manifest.parent.mkdir(parents=True, exist_ok=True)
            with open(write_manifest, "w", encoding="utf-8") as f:
                json.dump(selected, f, ensure_ascii=False, indent=2)
        return selected
