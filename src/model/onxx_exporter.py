from __future__ import annotations
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
import onnx
from onnxsim import simplify

from src.model.onxx_exporter_conf import OnnxExportConfig


class OnnxExporter:
    def __init__(self, cfg: OnnxExportConfig):
        self.cfg = cfg

    @torch.no_grad()
    def export(self, model: torch.nn.Module) -> Path:
        model = model.to(self.cfg.device).eval()

        C, H, W = self.cfg.input_shape
        dtype = torch.float16 if self.cfg.export_fp16 else torch.float32
        dummy = torch.zeros((1, C, H, W), dtype=dtype, device=self.cfg.device)

        dynamic_axes = {}
        if self.cfg.dynamic_batch or self.cfg.dynamic_hw:
            dyn_in = {}
            dyn_out = {}
            if self.cfg.dynamic_batch:
                dyn_in[0] = "batch"
                dyn_out[0] = "batch"
            if self.cfg.dynamic_hw:
                dyn_in[2] = "height"
                dyn_in[3] = "width"
                dyn_out[2] = "height"
                dyn_out[3] = "width"
            dynamic_axes = {
                self.cfg.input_name: dyn_in,
                self.cfg.output_name: dyn_out,
            }

        self.cfg.onnx_path.parent.mkdir(parents=True, exist_ok=True)

        export_kwargs = dict(
            args=(dummy,),
            f=str(self.cfg.onnx_path),
            export_params=True,
            opset_version=self.cfg.opset,
            do_constant_folding=self.cfg.do_constant_folding,
            input_names=[self.cfg.input_name],
            output_names=[self.cfg.output_name],
            dynamic_axes=dynamic_axes if dynamic_axes else None,
        )

        if self.cfg.external_data:
            export_kwargs.update(dict(
                use_external_data_format=True,
            ))

        if not self.cfg.export_fp16 and next(model.parameters()).dtype != torch.float32:
            model = model.float()

        try:
            y = model(dummy)
            if isinstance(y, (tuple, list)):
                y = y[0]
            # expects: [B, C, H, W]
            _ = tuple(y.shape)
        except Exception as e:
            raise RuntimeError(f"Dry-run forward failed before export: {e}")

        torch.onnx.export(model, **export_kwargs)

        if self.cfg.meta:
            try:
                m = onnx.load(str(self.cfg.onnx_path))
                for k, v in self.cfg.meta.items():
                    m.metadata_props.add(key=str(k), value=str(v))
                onnx.save(m, str(self.cfg.onnx_path))
            except Exception as e:
                print(f"[ONNX][warn] metadata inject failed: {e}")

        if self.cfg.simplify:
            try:
                m = onnx.load(str(self.cfg.onnx_path))
                m_simplified, ok = simplify(m)
                if ok:
                    onnx.save(m_simplified, str(self.cfg.onnx_path))
                else:
                    print("[ONNX][warn] onnx-simplify returned ok=False")
            except Exception as e:
                print(f"[ONNX][warn] simplify failed: {e}")

        if self.cfg.validate_with_ort:
            try:
                providers = ["CPUExecutionProvider"]
                sess = ort.InferenceSession(str(self.cfg.onnx_path), providers=providers)

                def _run(h, w):
                    x = torch.zeros((1, C, h, w), dtype=dtype).numpy().astype(
                        "float16" if self.cfg.export_fp16 else "float32"
                    )
                    out = sess.run([self.cfg.output_name], {self.cfg.input_name: x})[0]
                    return out.shape

                s1 = _run(H, W)
                print(f"[ONNX] ORT shape check OK @ {H}x{W}: {s1}")
                if self.cfg.dynamic_hw:
                    h2, w2 = max(32, H // 2), max(32, W // 2)
                    s2 = _run(h2, w2)
                    print(f"[ONNX] ORT shape check OK @ {h2}x{w2}: {s2}")
            except Exception as e:
                print(f"[ONNX][warn] onnxruntime validation skipped/failed: {e}")

        print(f"[ONNX] exported → {self.cfg.onnx_path}")
        return self.cfg.onnx_path
