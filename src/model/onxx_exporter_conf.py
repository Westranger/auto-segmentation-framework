from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


@dataclass
class OnnxExportConfig:
    onnx_path: Path
    input_shape: Tuple[int, int, int] = (3, 128, 128)
    dynamic_batch: bool = True
    dynamic_hw: bool = True
    opset: int = 13
    do_constant_folding: bool = True
    external_data: bool = False
    simplify: bool = False
    device: str = "cpu"
    input_name: str = "input"
    output_name: str = "logits"
    meta: Optional[Dict[str, Any]] = None
    validate_with_ort: bool = True
    export_fp16: bool = False
