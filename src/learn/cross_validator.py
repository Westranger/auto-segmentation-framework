from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Optional, Callable, Any
import numpy as np
import torch
import time
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from src.learn.trainer import Trainer, TrainResult, TrainConfig, MetricFn


class ModelFactory(Protocol):
    def build(self) -> Any: ...


@dataclass
class CVConfig:
    k_splits: int = 5
    batch_size: int = 4
    num_workers: int = 0
    seed: int = 42
    shuffle: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: Optional[bool] = None  # None → is True when num_workers>0
    drop_last: bool = False
    collate_fn: Optional[Callable] = None


@dataclass
class CVResult:
    mean_score: float
    fold_scores: List[float]
    best_fold: int
    best_state_dict: dict


class CrossValidator:
    def __init__(self, cfg: CVConfig, train_cfg: TrainConfig, metric_fn: MetricFn):
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.metric_fn = metric_fn

    def _make_loader(self, dataset_subset, shuffle: bool) -> DataLoader:
        persistent = self.cfg.persistent_workers
        if persistent is None:
            persistent = self.cfg.num_workers > 0

        pin = self.cfg.pin_memory and (self.train_cfg.device.startswith("cuda"))

        g = torch.Generator()
        g.manual_seed(self.cfg.seed)

        kwargs = dict(
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=pin,
            drop_last=self.cfg.drop_last,
            generator=g,
        )

        if self.cfg.num_workers > 0:
            kwargs.update(
                dict(
                    prefetch_factor=max(2, int(self.cfg.prefetch_factor)),
                    persistent_workers=persistent,
                )
            )

        if self.cfg.collate_fn is not None:
            kwargs["collate_fn"] = self.cfg.collate_fn

        return DataLoader(dataset_subset, **kwargs)

    def run(self, model_factory: ModelFactory, dataset) -> CVResult:
        kf = KFold(n_splits=self.cfg.k_splits, shuffle=self.cfg.shuffle, random_state=self.cfg.seed)
        indices = np.arange(len(dataset))
        fold_scores: List[float] = []
        best_score = -1.0
        best_state = {}
        best_fold = -1

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(indices)):
            t_fold_start = time.perf_counter()

            t_build0 = time.perf_counter()
            train_loader = DataLoader(
                Subset(dataset, tr_idx),
                batch_size=self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=(self.cfg.num_workers > 0),
                prefetch_factor=2 if self.cfg.num_workers > 0 else None  # <- optional
            )
            val_loader = DataLoader(
                Subset(dataset, va_idx),
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                pin_memory=self.cfg.pin_memory,
                persistent_workers=(self.cfg.num_workers > 0),
                prefetch_factor=2 if self.cfg.num_workers > 0 else None  # <- optional
            )

            t_build1 = time.perf_counter()
            print(f"[CV] Fold {fold_idx + 1}/{self.cfg.k_splits} "
                  f"| build_loaders={t_build1 - t_build0:.2f}s "
                  f"(train={len(tr_idx)} val={len(va_idx)} bs={self.cfg.batch_size} workers={self.cfg.num_workers})")

            model = model_factory.build()
            trainer = Trainer(self.train_cfg, self.metric_fn)
            t_train0 = time.perf_counter()
            result: TrainResult = trainer.train(model, train_loader, val_loader)
            t_train1 = time.perf_counter()

            print(f"[CV] Fold {fold_idx + 1}: train_time={t_train1 - t_train0:.2f}s  "
                  f"best_val={result.best_score:.4f}")

            fold_scores.append(result.best_score)
            if result.best_score > best_score:
                best_score = result.best_score
                best_state = result.best_state_dict
                best_fold = fold_idx

            t_fold_end = time.perf_counter()
            print(f"[CV] Fold {fold_idx + 1} total={t_fold_end - t_fold_start:.2f}s")

        mean_score = float(sum(fold_scores) / max(1, len(fold_scores)))
        return CVResult(mean_score=mean_score, fold_scores=fold_scores,
                        best_fold=best_fold, best_state_dict=best_state)
