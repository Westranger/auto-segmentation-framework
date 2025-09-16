from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class InferenceTiler:
    patch_size: int
    overlap: float = 0.25
    blend: Literal["mean"] = "mean"

    def _compute_steps(self, length: int) -> list[int]:
        S = self.patch_size
        stride = max(1, int(round(S * (1.0 - self.overlap))))
        if stride <= 0:
            stride = 1
        starts = list(range(0, max(1, length - S + 1), stride))
        if not starts or starts[-1] != (length - S):
            last = max(0, length - S)
            if (not starts) or (last != starts[-1]):
                starts.append(last)
        return starts

    @torch.no_grad()
    def predict_logits(self, model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
        """
        image: [C,H,W] float tensor on same device as model
        returns logits: [K,H,W] (no softmax)
        """
        assert image.ndim == 3, "image must be [C,H,W]"
        device = next(model.parameters()).device
        C, H, W = image.shape
        S = self.patch_size
        assert S <= H and S <= W, f"patch_size {S} must be <= image size {(H, W)}"

        cy = max(0, (H - S) // 2)
        cx = max(0, (W - S) // 2)
        sample = image[:, cy:cy + S, cx:cx + S].unsqueeze(0).to(device)
        sample_logits = model(sample)  # [1,K,S,S]
        K = sample_logits.shape[1]

        logits_full = torch.zeros((K, H, W), device=device, dtype=sample_logits.dtype)
        weight_full = torch.zeros((1, H, W), device=device, dtype=sample_logits.dtype)

        ys = self._compute_steps(H)
        xs = self._compute_steps(W)

        for y in ys:
            for x in xs:
                patch = image[:, y:y + S, x:x + S].unsqueeze(0).to(device)  # [1,C,S,S]
                out = model(patch)  # [1,K,S,S]
                logits_full[:, y:y + S, x:x + S] += out.squeeze(0)
                weight_full[:, y:y + S, x:x + S] += 1.0

        weight_full = torch.clamp(weight_full, min=1.0)
        logits_full = logits_full / weight_full
        return logits_full

    @torch.no_grad()
    def predict_mask(self, model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
        """
        returns predicted mask as [1,H,W] long tensor (argmax over K)
        """
        logits = self.predict_logits(model, image)  # [K,H,W]
        pred = torch.argmax(logits, dim=0, keepdim=True).to(torch.long)  # [1,H,W]
        return pred
