from typing import Dict, Any, Counter


def infer_classes_and_mapping(counter: Counter, max_classes_warn: int = 50) -> Dict[str, Any]:
    """
    rules:
      - {0,255} -> binary, mapping {0:0, 255:1}
      - {0,1}   -> binary, no mapping
      - {0..K-1} consecutive -> multiclass K
    """
    uniq = sorted(counter.keys())
    out: Dict[str, Any] = {"num_classes": None, "mapping": None, "note": ""}

    if uniq == [0, 255]:
        out["num_classes"] = 2
        out["mapping"] = {0: 0, 255: 1}
        out["note"] = "Detected binary 0/255 → mapping to {0,1}."
        return out
    if uniq == [0, 1]:
        out["num_classes"] = 2
        out["mapping"] = None
        out["note"] = "Detected binary {0,1}."
        return out

    if uniq and uniq[0] == 0 and uniq[-1] == len(uniq) - 1 and len(uniq) <= max_classes_warn:
        out["num_classes"] = len(uniq)
        out["mapping"] = None
        out["note"] = f"Detected contiguous classes 0..{len(uniq) - 1}."
        return out

    out["note"] = (f"Non-contiguous / unexpected label set: {uniq[:20]} "
                   f"(showing up to 20). Provide explicit mapping.")
    return out
