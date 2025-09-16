import math
import time, torch
from contextlib import nullcontext


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_latency_ms(model, input_size, device="cuda", warmup=5, iters=10, amp=False):
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    autocast_ctx = torch.cuda.amp.autocast() if (amp and device.startswith("cuda")) else nullcontext()
    # warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    for _ in range(warmup):
        with autocast_ctx:
            _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    # timing
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with autocast_ctx:
            _ = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    # robust mean
    return times[len(times) // 2]


def composite_score(val_score: float, n_params: int, lat_ms: float,
                    alpha: float = 0.06,  # wgt on param loop
                    beta: float = 0.12,  # wgt on latency loop
                    p_ref: int = 1_000_000,  # 1 mio params
                    t_ref: float = 10.0):  # ref: 10 ms @ batch=1
    pen_p = max(0.0, math.log10(max(1.0, n_params) / p_ref))
    pen_t = max(0.0, math.log10(max(1e-3, lat_ms) / t_ref))
    return float(val_score - alpha * pen_p - beta * pen_t)
