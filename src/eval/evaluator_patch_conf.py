from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import torch

from src.learn.metrics import pixel_accuracy

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass
class PatchCVEvaluatorConfig:
    # data
    batch_size: int = 4
    num_workers: int = 0
    # train
    epochs: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    grad_clip: float | None = None
    # CV
    k_splits: int = 5
    seed: int = 42
    shuffle: bool = True
    pin_memory: bool = True
    # metric
    metric_fn: MetricFn = pixel_accuracy
