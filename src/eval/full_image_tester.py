from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
import torch

from src.model.inference_tiler import InferenceTiler

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


@dataclass
class FullImageTester:
    tiler: InferenceTiler
    metric_fn: MetricFn
    device: str = "cpu"

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module, dataset) -> Dict:
        """
        dataset: FullImageDataset (returns (img[C,H,W], mask[1,H,W], stem))
        output:
          {
            "mean_score": float,
            "per_image": [
                {"stem": str, "score": float},
                ...
            ]
          }
        """
        model = model.to(self.device).eval()
        per_image: List[Dict] = []
        scores: List[float] = []

        for i in range(len(dataset)):
            img_t, gt_t, stem = dataset[i]  # [C,H,W], [1,H,W]
            img_t = img_t.to(self.device, non_blocking=True)

            pred_mask = self.tiler.predict_mask(model, img_t)  # [1,H,W]

            score = self.metric_fn(pred_mask.unsqueeze(0).cpu(), gt_t.unsqueeze(0).cpu())
            per_image.append({"stem": stem, "score": float(score)})
            scores.append(float(score))

        mean_score = float(sum(scores) / max(1, len(scores)))
        return {"mean_score": mean_score, "per_image": per_image}
