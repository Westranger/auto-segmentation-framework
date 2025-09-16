import torch


@torch.no_grad()
def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    pred_mask, gt_mask: [B,1,H,W] oder [B,H,W] (ints)
    """
    if pred_mask.dim() == 4: pred_mask = pred_mask[:, 0]
    if gt_mask.dim() == 4:   gt_mask = gt_mask[:, 0]
    correct = (pred_mask == gt_mask).sum().item()
    total = pred_mask.numel()
    return correct / max(1, total)


@torch.no_grad()
def dice_score(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int | None = None,
               average: str = "macro") -> float:
    """
    Multiclass Dice: compute Dice per class (0..num_classes-1) and compute mean.
    """
    if pred_mask.dim() == 4: pred_mask = pred_mask[:, 0]
    if gt_mask.dim() == 4:   gt_mask = gt_mask[:, 0]
    if num_classes is None:
        # binary fallback: 1 vs else
        pred_bin = (pred_mask > 0)
        gt_bin = (gt_mask > 0)
        inter = (pred_bin & gt_bin).sum().item()
        den = pred_bin.sum().item() + gt_bin.sum().item()
        return (2 * inter) / den if den > 0 else 1.0

    dices = []
    for c in range(num_classes):
        p = (pred_mask == c)
        g = (gt_mask == c)
        inter = (p & g).sum().item()
        den = p.sum().item() + g.sum().item()
        if den > 0:
            dices.append((2 * inter) / den)
    if not dices:
        return 0.0
    if average == "macro":
        return float(sum(dices) / len(dices))
    elif average == "micro":
        # micro dice == 2*global_inter / (sum pred + sum gt)
        inter = 0
        pred_sum = 0
        gt_sum = 0
        for c in range(num_classes):
            p = (pred_mask == c)
            g = (gt_mask == c)
            inter += (p & g).sum().item()
            pred_sum += p.sum().item()
            gt_sum += g.sum().item()
        den = pred_sum + gt_sum
        return (2 * inter) / den if den > 0 else 0.0
    else:
        raise ValueError("average must be 'macro' or 'micro'")
