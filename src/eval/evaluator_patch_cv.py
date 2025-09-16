from __future__ import annotations
from typing import List, Tuple

from src.eval.evaluator_patch_conf import PatchCVEvaluatorConfig
from src.learn.cross_validator import CrossValidator, CVConfig
from src.learn.trainer import TrainConfig


class PatchCVEvaluator:
    """
    end-to-end evaluator for Optuna:
      evaluate(input_size, lst_tuples[(ks,ch),...]) -> CV-score (float)
    input:
      - tile_dataset: PyTorch Dataset mit (img[3,H,W], mask[1,H,W])
      - model_factory: object with .build(input_size, encoder_specs) -> nn.Module
    """

    def __init__(self, tile_dataset, model_factory, cfg: PatchCVEvaluatorConfig):
        self.tile_dataset = tile_dataset
        self.model_factory = model_factory
        self.cfg = cfg

    def evaluate(self, input_size: int, lst_tuples: List[Tuple[int, int]]) -> float:
        cv = CrossValidator(
            CVConfig(
                k_splits=self.cfg.k_splits,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.num_workers,
                seed=self.cfg.seed,
                shuffle=self.cfg.shuffle,
                pin_memory=self.cfg.pin_memory,
            ),
            TrainConfig(
                epochs=self.cfg.epochs,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                device=self.cfg.device,
                grad_clip=self.cfg.grad_clip,
            ),
            self.cfg.metric_fn,
        )

        class _ParamBoundFactory:
            def __init__(self, outer):
                self.outer = outer

            def build(self):
                return self.outer.model_factory.build((input_size, input_size), lst_tuples)

        result = cv.run(_ParamBoundFactory(self), self.tile_dataset)
        return float(result.mean_score)
