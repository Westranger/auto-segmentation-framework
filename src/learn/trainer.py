from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional
import time
import torch
from torch.utils.data import DataLoader

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]  # (pred[B,1,H,W], gt[B,1,H,W]) -> float


def _sync_if_cuda(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    grad_clip: Optional[float] = None
    mixed_precision: bool = False
    max_train_batches: Optional[int] = None
    max_val_batches: Optional[int] = None


@dataclass
class TrainResult:
    best_score: float
    best_state_dict: dict
    history: List[Dict]  # list of {"epoch": int, "train_loss": float, "val_score": float, ...}


class Trainer:
    """
    Minimaler, trainer for SemSeg with CrossEntropy.
    input:
      - Model: forward(x)->logits [B,K,H,W]
      - Targets: [B,1,H,W] long (will get [B,H,W])
      - metric_fn: works on argmax(logits) against target
    """

    def __init__(self, cfg: TrainConfig, metric_fn: MetricFn):
        self.cfg = cfg
        self.metric_fn = metric_fn
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> TrainResult:
        device = torch.device(self.cfg.device)
        model = model.to(device)
        amp_enabled = (self.cfg.mixed_precision and device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        optim = torch.optim.Adam(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        best_score = -1.0
        best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        history: List[Dict] = []

        devcheck_done = False

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()

            run_loss = 0.0
            nitems = 0
            nbatches = 0
            data_time_sum = 0.0
            compute_time_sum = 0.0

            end_prev = time.perf_counter()

            # -------- Training --------
            for xb, yb in train_loader:
                t_batch_start = time.perf_counter()

                data_time_sum += (t_batch_start - end_prev)

                xb = xb.to(device, non_blocking=True)
                y = yb[:, 0].to(device, non_blocking=True).long()

                if not devcheck_done:
                    print("[DEV] model on:", next(model.parameters()).device)
                    print("[DEV] x on:", xb.device, " y on:", y.device)
                    devcheck_done = True

                optim.zero_grad(set_to_none=True)

                _sync_if_cuda(device)
                t0 = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(xb)  # [B,K,H,W]
                    n_classes = logits.shape[1]
                    if y.min() < 0 or y.max() >= n_classes:
                        with torch.no_grad():
                            uniq = torch.unique(y)
                        raise ValueError(
                            f"Target labels out of range for CrossEntropy: "
                            f"min={int(y.min())}, max={int(y.max())}, n_classes={n_classes}. "
                            f"Unique(first20)={uniq[:20].tolist()}. "
                            f"-> Prüfe num_classes, mask_value_mapping, binary_auto im Dataset."
                        )
                    loss = self.criterion(logits, y)

                if amp_enabled:
                    scaler.scale(loss).backward()
                    if self.cfg.grad_clip is not None:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.cfg.grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                    optim.step()

                _sync_if_cuda(device)
                t1 = time.perf_counter()

                compute_time_sum += (t1 - t0)
                bs = xb.size(0)
                run_loss += loss.item() * bs
                nitems += bs
                nbatches += 1
                end_prev = time.perf_counter()

                if nbatches <= 3:
                    data_ms = (t_batch_start - (end_prev - (t1 - t0))) * 1000.0
                    comp_ms = (t1 - t0) * 1000.0
                    print(f"[B{nbatches:02d}] data={data_ms:.1f}ms  fwd+bwd={comp_ms:.1f}ms  "
                          f"imgs={bs}  loss={loss.item():.4f}")

                if nbatches % 20 == 0:
                    total_batches = len(train_loader)
                    avg_bt = (data_time_sum + compute_time_sum) / max(1, nbatches)
                    eta = avg_bt * (total_batches - nbatches)
                    print(f"[Train] progress {nbatches}/{total_batches}  ETA ~{eta:.1f}s  "
                          f"avg_batch={(avg_bt * 1000):.1f}ms")

                if self.cfg.max_train_batches and nbatches >= self.cfg.max_train_batches:
                    break

            train_loss = run_loss / max(1, nitems)

            print("[start validation]")
            t_val0 = time.perf_counter()

            model.eval()
            val_scores = []
            nbatches_val = 0

            ctx_infer = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            with ctx_infer():
                t_prev = t_val0
                for xb, yb in val_loader:
                    t_data_in = time.perf_counter()
                    data_ms = (t_data_in - t_prev) * 1000.0

                    xb = xb.to(device, non_blocking=True)
                    y = yb[:, 0].to(device, non_blocking=True).long()  # [B,H,W]

                    t0 = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        logits = model(xb)  # [B,K,H,W]
                        pred = logits.argmax(dim=1)  # [B,H,W] (long)

                    K = logits.shape[1]
                    if K == 2:
                        p1 = (pred == 1)
                        y1 = (y == 1)
                        inter = (p1 & y1).sum(dtype=torch.float32)
                        denom = p1.sum(dtype=torch.float32) + y1.sum(dtype=torch.float32)
                        dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
                        score = float(dice.detach().item())
                    else:
                        score = self.metric_fn(pred.detach().unsqueeze(1).cpu(),
                                               y.detach().unsqueeze(1).cpu())

                    val_scores.append(score)

                    t1 = time.perf_counter()
                    fwd_ms = (t1 - t0) * 1000.0
                    nbatches_val += 1

                    if nbatches_val <= 3:
                        print(f"[Val B{nbatches_val:02d}] data={data_ms:.1f}ms  fwd={fwd_ms:.1f}ms  "
                              f"imgs={xb.size(0)}  score={score:.4f}")

                    if nbatches_val % 20 == 0:
                        total_v = len(val_loader)
                        avg_bt = (t1 - t_val0) / max(1, nbatches_val)
                        eta = avg_bt * (total_v - nbatches_val)
                        print(f"[Val] progress {nbatches_val}/{total_v}  ETA ~{eta:.1f}s  "
                              f"avg_batch={avg_bt * 1000:.1f}ms")

                    t_prev = time.perf_counter()

                    if self.cfg.max_val_batches and nbatches_val >= self.cfg.max_val_batches:
                        break

            t_val1 = time.perf_counter()
            val_time = t_val1 - t_val0
            val_score = float(sum(val_scores) / max(1, len(val_scores)))
            print(f"[Val] batches={nbatches_val} time={val_time:.2f}s mean={val_score:.4f}")

            # report
            imgs_per_sec = (nitems / compute_time_sum) if compute_time_sum > 0 else 0.0
            data_ratio = data_time_sum / max(1e-6, (data_time_sum + compute_time_sum))
            if device.type == "cuda":
                mem_gb = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"[Train] Epoch {epoch}/{self.cfg.epochs} "
                      f"loss={train_loss:.4f} val={val_score:.4f} | "
                      f"GPU~{mem_gb:.2f}GB | "
                      f"data%={100 * data_ratio:.1f} imgs/s={imgs_per_sec:.1f}")
            else:
                print(f"[Train] Epoch {epoch}/{self.cfg.epochs} "
                      f"loss={train_loss:.4f} val={val_score:.4f} | "
                      f"data%={100 * data_ratio:.1f} imgs/s={imgs_per_sec:.1f}")

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_score": val_score,
                "data_ratio": data_ratio,
                "imgs_per_sec": imgs_per_sec,
                "nitems": nitems,
                "nbatches": nbatches,
                "val_time": val_time,
            })

            if val_score > best_score:
                best_score = val_score
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

        return TrainResult(best_score=best_score, best_state_dict=best_state, history=history)
