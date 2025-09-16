from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from tqdm.auto import tqdm

from src.data.rep_selector_conf import InMemRepSelectorConfig
from src.data.util import _collect_pairs, _jsonify


def _pbar(iterable, cfg: InMemRepSelectorConfig, **kwargs):
    return tqdm(iterable, **kwargs) if cfg.progress else iterable


class InMemoryRepresentativeIndexer:
    def __init__(self, cfg: InMemRepSelectorConfig):
        self.cfg = cfg
        self.images_dir = Path(cfg.images_dir)
        self.masks_dir = Path(cfg.masks_dir)
        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError("images_dir/masks_dir do not exist")
        self._build_backbone()

    def _build_backbone(self):
        try:
            weights = models.ResNet18_Weights.DEFAULT if self.cfg.pretrained else None
        except Exception:
            weights = None if not self.cfg.pretrained else None
        model = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1]).to(self.cfg.device).eval()
        if weights is not None:
            self.pre = weights.transforms(antialias=True, crop_size=224, resize_size=256)
        else:
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            self.pre = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    @staticmethod
    def _tile_grid(w: int, h: int, ps: int, overlap: float) -> List[Tuple[int, int, int, int]]:
        step = max(1, int(ps * (1.0 - overlap)))
        xs = list(range(0, max(1, w - ps + 1), step))
        ys = list(range(0, max(1, h - ps + 1), step))
        if xs[-1] + ps < w: xs.append(w - ps)
        if ys[-1] + ps < h: ys.append(h - ps)
        return [(x, y, ps, ps) for y in ys for x in xs]

    @staticmethod
    def _entropy_gray(img: Image.Image) -> float:
        g = img.convert("L")
        arr = np.asarray(g, dtype=np.uint8)
        counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        total = counts.sum()
        if total == 0:
            return 0.0
        p = counts / total
        nz = p[p > 0]
        return float(-(nz * np.log2(nz)).sum())

    @staticmethod
    def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        A = A.astype(np.float32, copy=False);
        B = B.astype(np.float32, copy=False)
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        return A @ B.T

    @torch.no_grad()
    def _embed_batch(self, imgs: List[Image.Image]) -> np.ndarray:
        dev = self.cfg.device
        tensors = [self.pre(im.convert("RGB")) for im in imgs]
        X = torch.stack(tensors, 0).to(dev, non_blocking=True)
        f = self.backbone(X).squeeze(-1).squeeze(-1)
        return f.cpu().numpy()

    def _simhash(self, emb: np.ndarray, rnd_dim: int = 64, seed: int = 1234) -> np.ndarray:
        rng = np.random.default_rng(seed)
        R = rng.standard_normal(size=(rnd_dim, emb.shape[1]), dtype=np.float32)  # [64,D]
        proj = emb @ R.T
        bits = (proj >= 0).astype(np.uint8)
        sig = np.zeros((emb.shape[0],), dtype=np.uint64)
        for b in range(64):
            sig |= (bits[:, b].astype(np.uint64) << np.uint64(b))
        return sig

    def _lsh_buckets(self, sig: np.ndarray, bands: int, band_bits: int = 16):
        sig = np.asarray(sig, dtype=np.uint64)
        assert bands * band_bits == 64, "bands*band_bits must be 64"
        mask = (np.uint64(1) << np.uint64(band_bits)) - np.uint64(1)
        buckets = [dict() for _ in range(bands)]
        for i, s in enumerate(sig):
            s = np.uint64(s)
            for b in range(bands):
                key = int((s >> np.uint64(b * band_bits)) & mask)
                buckets[b].setdefault(key, []).append(i)
        return buckets

    def _prune_duplicates_lsh(self, emb: np.ndarray, sim_threshold: float,
                              bands: int = 4, band_bits: int = 16) -> np.ndarray:
        print(f"[Dedupe] LSH (N={emb.shape[0]:,}, thr={sim_threshold})")
        sig = self._simhash(emb)
        buckets = self._lsh_buckets(sig, bands=bands, band_bits=band_bits)
        N = emb.shape[0]
        keep = np.ones(N, dtype=bool)
        X = emb.astype(np.float32, copy=False)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        for table in buckets:
            for _, idxs in table.items():
                if len(idxs) < 2:
                    continue
                base = next((j for j in idxs if keep[j]), idxs[0])
                for j in idxs:
                    if j == base or not keep[base] or not keep[j]:
                        continue
                    sim = float(X[base] @ X[j])
                    if sim >= sim_threshold:
                        keep[j] = False
        return np.flatnonzero(keep).astype(np.int64)

    def build_index(self) -> Dict:
        if self.cfg.index_dir is None:
            raise ValueError("cfg.index_dir must be set for persistence")
        idxdir = Path(self.cfg.index_dir)
        idxdir.mkdir(parents=True, exist_ok=True)

        meta: List[Dict] = []

        if (not self.cfg.recompute_embeddings and
                (idxdir / "embeddings.npy").exists() and
                (idxdir / "meta.jsonl").exists()):
            print("[Index] re-use existing embeddings/meta")
            with open(idxdir / "meta.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    meta.append(json.loads(line))
            emb = np.load(idxdir / "embeddings.npy").astype(np.float32, copy=False)
        else:
            print("[Index] re build embeddings/meta")
            ps = self.cfg.patch_size
            ov = self.cfg.overlap_pct

            pairs = _collect_pairs(
                images_dir=self.images_dir,
                masks_dir=self.masks_dir,
                image_exts=(".jpg", ".jpeg", ".png"),
                mask_exts=(".png", ".jpg", ".jpeg"),
                mask_suffixes=("_mask", "-mask", "_label", "-label"),
                casefold=True,
            )

            emb_list: List[np.ndarray] = []

            for ip, mp in _pbar(pairs, self.cfg, desc="Images", unit="img"):
                img = Image.open(ip).convert("RGB")
                W, H = img.size
                boxes = self._tile_grid(W, H, ps, ov)

                tiles_imgs: List[Image.Image] = []
                tiles_boxes: List[Tuple[int, int, int, int]] = []
                tiles_entropy: List[float] = []

                for x, y, w, h in boxes:
                    crop = img.crop((x, y, x + w, y + h))
                    tiles_imgs.append(crop)
                    tiles_boxes.append((x, y, w, h))
                    tiles_entropy.append(self._entropy_gray(crop) if self.cfg.prefilter_by_entropy else 1.0)

                idx_all = np.arange(len(tiles_imgs))
                if self.cfg.prefilter_by_entropy and len(idx_all) > 0:
                    q = np.quantile(np.array(tiles_entropy), self.cfg.prefilter_entropy_quantile)
                    idx_all = np.array([i for i in idx_all if tiles_entropy[i] >= q], dtype=int)

                if self.cfg.max_tiles_per_image > 0 and len(idx_all) > self.cfg.max_tiles_per_image:
                    ent = np.array([tiles_entropy[i] for i in idx_all])
                    order = np.argsort(-ent)
                    idx_all = idx_all[order[: self.cfg.max_tiles_per_image]]

                batch: List[Image.Image] = []
                batch_map: List[int] = []
                for i in idx_all.tolist():
                    batch.append(tiles_imgs[i])
                    batch_map.append(i)
                    if len(batch) >= self.cfg.batch_embed:
                        e = self._embed_batch(batch)
                        emb_list.append(e)
                        for j in batch_map:
                            bx = tiles_boxes[j]
                            meta.append({
                                "src_image": str(ip),
                                "src_mask": str(mp),
                                "x": bx[0], "y": bx[1], "w": bx[2], "h": bx[3],
                            })
                        batch.clear();
                        batch_map.clear()
                if batch:
                    e = self._embed_batch(batch)
                    emb_list.append(e)
                    for j in batch_map:
                        bx = tiles_boxes[j]
                        meta.append({
                            "src_image": str(ip),
                            "src_mask": str(mp),
                            "x": bx[0], "y": bx[1], "w": bx[2], "h": bx[3],
                        })

            if not meta:
                raise RuntimeError("no tiles created")

            emb = np.concatenate(emb_list, 0).astype(np.float32)

            with open(idxdir / "meta.jsonl", "w", encoding="utf-8") as f:
                for i, rec in enumerate(meta):
                    rec2 = dict(rec);
                    rec2["tile_index"] = i
                    f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            np.save(idxdir / "embeddings.npy", emb)

            empty_features = np.zeros((emb.shape[0], 0), dtype=np.float32)
            np.save(idxdir / "features.npy", empty_features)

        if self.cfg.dedupe_method == "lsh" and self.cfg.similarity_prune < 1.0 and emb.shape[0] > 1:
            idx_keep = self._prune_duplicates_lsh(
                emb, self.cfg.similarity_prune,
                bands=self.cfg.lsh_bands, band_bits=self.cfg.lsh_band_bits
            )
            emb = emb[idx_keep]

        X = emb.astype(np.float32, copy=False)
        N = X.shape[0]
        if N == 0:
            raise RuntimeError("no data after pruning.")

        if self.cfg.target_k is None:
            k = self._auto_k_via_silhouette(X, min(self.cfg.min_k, N), min(self.cfg.max_k, N))
        else:
            k = min(self.cfg.target_k, N)

        centers_idx = self._greedy_k_centers(X, k)

        sims = self._cosine_sim(X, X[centers_idx])  # [N,k]
        assign = np.argmax(sims, axis=1)  # [N]
        medoid_indices: List[int] = []
        medoid_scores: List[float] = []
        for c in range(k):
            members = np.where(assign == c)[0]
            if members.size == 0:
                medoid_indices.append(int(centers_idx[c]))
                medoid_scores.append(0.0)
                continue
            Xm = X[members]
            center = Xm.mean(axis=0, keepdims=True)
            cs = self._cosine_sim(X[members], center).ravel()
            mi_local = members[int(np.argmax(cs))]
            medoid_indices.append(int(mi_local))
            medoid_scores.append(float(cs.max()))

        cluster_sizes = np.bincount(assign, minlength=k).astype(int)
        size_norm = (cluster_sizes / max(1, cluster_sizes.max())).astype(np.float32)
        centrality = np.array(medoid_scores, dtype=np.float32)
        centrality = (centrality - centrality.min()) / (centrality.ptp() + 1e-12)
        medoid_priority = 0.7 * size_norm + 0.3 * centrality
        medoid_order = np.argsort(-medoid_priority).tolist()

        (idxdir / "images").mkdir(exist_ok=True)

        summary = {
            "num_tiles": int(N),
            "feature_dim": int(X.shape[1]),
            "embed_dim": int(emb.shape[1]),
            "num_classes": 0,
            "k": int(k),
        }

        cfg_dump = asdict(self.cfg)
        cfg_dump["images_dir"] = str(self.images_dir)
        cfg_dump["masks_dir"] = str(self.masks_dir)
        cfg_dump = _jsonify(cfg_dump)
        payload = _jsonify({**cfg_dump, **summary})

        for k in ("index_dir", "out_dir", "write_manifest"):
            if cfg_dump.get(k) is not None:
                cfg_dump[k] = str(cfg_dump[k])

        with open(idxdir / "config.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        cl_dict = {
            "assign": assign.tolist(),
            "centers_idx": [int(x) for x in centers_idx],
            "medoid_indices": medoid_indices,
            "medoid_priority": medoid_priority.tolist(),
            "medoid_order": medoid_order,
            "cluster_sizes": cluster_sizes.tolist(),
        }
        with open(idxdir / "clusters.json", "w", encoding="utf-8") as f:
            json.dump(cl_dict, f, ensure_ascii=False, indent=2)

        return {**summary, "index_dir": str(idxdir)}

    def _greedy_k_centers(self, X: np.ndarray, k: int) -> List[int]:
        sims_to_mean = self._cosine_sim(X, X.mean(axis=0, keepdims=True)).ravel()
        first = int(np.argmax(sims_to_mean))
        selected = [first]
        if k == 1:
            return selected
        minDist = 1.0 - self._cosine_sim(X, X[selected]).ravel()
        while len(selected) < k:
            cand = int(np.argmax(minDist))
            selected.append(cand)
            d = 1.0 - self._cosine_sim(X, X[[cand]]).ravel()
            minDist = np.minimum(minDist, d)
        return selected

    def _silhouette_score(self, X: np.ndarray, centers_idx: List[int]) -> float:
        Cc = X[centers_idx]
        sims = self._cosine_sim(X, Cc)
        assign = np.argmax(sims, axis=1)
        a = 1.0 - sims[np.arange(X.shape[0]), assign]
        sims_sorted = np.sort(sims, axis=1)
        b = 1.0 - sims_sorted[:, -2] if Cc.shape[0] > 1 else np.ones_like(a)
        s = (b - a) / np.maximum(a, b, 1e-12)
        return float(np.mean(s))

    def _auto_k_via_silhouette(self, X: np.ndarray, kmin: int, kmax: int) -> int:
        best_k, best_s = kmin, -1e9
        step = max(1, (kmax - kmin) // 5 or 1)
        for k in _pbar(range(kmin, max(kmin, kmax) + 1, step), self.cfg, desc="Scan k", unit="k"):
            if k < 2:
                continue
            idx = self._greedy_k_centers(X, k)
            s = self._silhouette_score(X, idx)
            s_reg = s - 0.001 * k
            if s_reg > best_s:
                best_s, best_k = s_reg, k
        return best_k
