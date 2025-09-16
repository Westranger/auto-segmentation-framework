from pathlib import Path
from collections import Counter
from typing import Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import torch
import optuna
from tqdm.auto import tqdm

from src.data.rep_materializer import RepresentativeMaterializer
from src.data.rep_selector_inmemory import InMemRepSelectorConfig, \
    InMemoryRepresentativeIndexer
from src.data.tile_dataset import TileDataset
from src.data.full_image_dataset import FullImageDataset
from src.model.onxx_exporter import OnnxExporter
from src.model.onxx_exporter_conf import OnnxExportConfig
from src.net.unet_factory import UNetFactory
from src.eval.evaluator_patch_cv import PatchCVEvaluator, PatchCVEvaluatorConfig
from src.net.unet_parameter_generator import UNetParamGenerator
from src.learn.cross_validator import CrossValidator, CVConfig
from src.learn.trainer import TrainConfig
from src.eval.full_image_tester import FullImageTester
from src.model.inference_tiler import InferenceTiler
from src.learn.metrics import dice_score

TRAIN_IMAGES = Path(r"E:\SegmentDronePatches\images_train")
TRAIN_MASKS = Path(r"E:\SegmentDronePatches\masks_train")
TEST_IMAGES = Path(r"E:\SegmentDronePatches\images_test")
TEST_MASKS = Path(r"E:\SegmentDronePatches\masks_test")
TILES_DIR = Path(r"E:\SegmentDronePatches\tiles")

# Patch/Tiling
PATCH_SIZE = 128
OVERLAP = 0.25  # 0..1

IN_CHANNELS = 3
NUM_CLASSES = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training/CV
N_REPS = 200
EPOCHS = 3
BATCH_SIZE = 16
KFOLD = 5
LR = 1e-3
NUM_WORKERS = 4

# Optuna Budget
N_TRIALS_PER_SETTING = 10

NET_INPUT_SIZES = (PATCH_SIZE,)
NUM_LAYER_CHANNELS = (4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 256, 512)
NUM_LAYERS = (2,)
FILTER_SIZES = (3, 5, 7)


def _pbar(iterable, **kwargs):
    return tqdm(iterable, **kwargs)


def scan_mask_values(masks_dir: Path, limit: Optional[int] = 200) -> Tuple[Counter, int]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [p for p in Path(masks_dir).rglob("*") if p.suffix.lower() in exts]
    if limit is not None:
        files = files[:limit]
    counter = Counter()
    for p in _pbar(files, desc="Scan mask values", unit="mask"):
        m = Image.open(p)
        if m.mode not in ("L", "P"):
            m = m.convert("L")
        arr = np.array(m, copy=True)
        vals, counts = np.unique(arr, return_counts=True)
        counter.update(dict(zip(vals.tolist(), counts.tolist())))
    return counter, len(files)


def infer_classes_and_mapping(counter: Counter, max_classes_warn: int = 50) -> Dict[str, Any]:
    uniq = sorted(counter.keys())
    out: Dict[str, Any] = {"num_classes": None, "mapping": None, "note": ""}

    if uniq == [0, 255]:
        out["num_classes"] = 2
        out["mapping"] = {0: 0, 255: 1}
        out["note"] = "Detected binary masks {0,255} → mapping to {0,1}."
        return out
    if uniq == [0, 1]:
        out["num_classes"] = 2
        out["mapping"] = None
        out["note"] = "Detected binary masks {0,1}."
        return out

    if uniq and uniq[0] == 0 and uniq[-1] == len(uniq) - 1 and len(uniq) <= max_classes_warn:
        out["num_classes"] = len(uniq)
        out["mapping"] = None
        out["note"] = f"Detected contiguous classes 0..{len(uniq) - 1}."
        return out

    out["note"] = (f"Non-contiguous / unexpected label set: {uniq[:20]} (showing up to 20). "
                   f"Consider providing an explicit mapping.")
    return out


def setup_gpu():
    """Configure PyTorch backend flags for max performance on GPU."""
    print("[GPU] Available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        return

    print("[GPU] Device count:", torch.cuda.device_count())
    print("[GPU] Current device:", torch.cuda.current_device())
    print("[GPU] Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    torch.backends.cudnn.benchmark = True
    print("[GPU] cuDNN benchmark:", torch.backends.cudnn.benchmark)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[GPU] TF32 matmul:", torch.backends.cuda.matmul.allow_tf32)
    print("[GPU] TF32 cuDNN :", torch.backends.cudnn.allow_tf32)

    try:
        torch.set_float32_matmul_precision("high")
        print("[GPU] MatMul precision set to 'high'")
    except AttributeError:
        print("[GPU] MatMul precision control not available (PyTorch < 2.0)")


def main():
    setup_gpu()
    print(f"[CFG] device={DEVICE}  patch={PATCH_SIZE}  overlap={OVERLAP}")

    cnt, scanned = scan_mask_values(TRAIN_MASKS, limit=200)
    print(f"[Masks] scanned={scanned}, unique={len(cnt)}")
    preview = ", ".join(f"{k}:{v}" for k, v in list(sorted(cnt.items()))[:10])
    print(f"[Masks] first uniques: {preview} ...")
    info = infer_classes_and_mapping(cnt)
    print(f"[Masks] {info['note']}")
    num_classes = info["num_classes"] if info["num_classes"] is not None else NUM_CLASSES
    mask_mapping = info["mapping"]  # kann None sein
    print(f"[CFG] num_classes = {num_classes}")
    if mask_mapping:
        print(f"[CFG] mask mapping = {mask_mapping}")

    INDEX_DIR = TILES_DIR.parent / "tile_index"
    cfg = InMemRepSelectorConfig(
        images_dir=TRAIN_IMAGES,
        masks_dir=TRAIN_MASKS,
        patch_size=PATCH_SIZE,
        overlap_pct=OVERLAP,
        target_k=200,  # or None -> auto-k
        min_k=50,
        max_k=200,
        device=DEVICE,
        similarity_prune=0.995,
        index_dir=INDEX_DIR,
        progress=True,
        mask_mapping=mask_mapping,
        num_classes_override=num_classes,
    )
    indexer = InMemoryRepresentativeIndexer(cfg)

    if not (INDEX_DIR / "clusters.json").exists():
        summary = indexer.build_index()
        print("[Index] gebaut:", summary)
    else:
        print("[Index] gefunden – überspringe Build")

    REDUCED_TILES = TILES_DIR.parent / "tiles_small"

    mat = RepresentativeMaterializer(INDEX_DIR)
    selected_manifest = mat.materialize(
        n=N_REPS,
        out_dir=REDUCED_TILES,
        write_manifest=TILES_DIR / f"rep_manifest_top{N_REPS}.json",
    )
    print(f"[RepSel] materialisiert {len(selected_manifest)} tiles -> {REDUCED_TILES}")

    tiles_ds = TileDataset(
        tiles_dir=REDUCED_TILES,
        num_classes=num_classes,
        mask_value_mapping=mask_mapping,
        binary_auto=True,
    )
    test_ds = FullImageDataset(
        images_dir=TEST_IMAGES,
        masks_dir=TEST_MASKS,
        num_classes=num_classes,
        mask_value_mapping=mask_mapping,
        binary_auto=True,
    )
    factory = UNetFactory(in_channels=IN_CHANNELS, num_classes=num_classes)
    eval_cfg = PatchCVEvaluatorConfig(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        epochs=EPOCHS, lr=LR, device=DEVICE,
        k_splits=KFOLD, metric_fn=dice_score,
    )
    evaluator = PatchCVEvaluator(tile_dataset=tiles_ds, model_factory=factory, cfg=eval_cfg)

    gen = UNetParamGenerator(
        net_input_sizes=NET_INPUT_SIZES,
        num_layer_channels=NUM_LAYER_CHANNELS,
        num_layers=NUM_LAYERS,
        filter_sizes=FILTER_SIZES,
        in_channels=IN_CHANNELS,
        num_classes=num_classes,
        evaluator=evaluator,
    )

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)

    results = gen.generate(
        n_trials_per_setting=N_TRIALS_PER_SETTING,
        sampler=sampler,
        pruner=pruner,
        study_name_prefix="unet_param_search",
        verbose=True,
    )

    best_key = None
    best_val = None
    for k, v in results.items():
        if v["best_value"] is None:
            continue
        if best_val is None or v["best_value"] > best_val["best_value"]:
            best_key, best_val = k, v

    if best_val is None:
        print("no valid parameters found")
        return

    L, S = best_key
    best_specs = best_val["best_params"]
    print("\n===== BEST (CV) =====")
    print(f"S={S}, L={L}")
    print(f"encoder_specs: {best_specs}")
    print(f"CV-score:      {best_val['best_value']:.4f}")

    cv = CrossValidator(
        CVConfig(k_splits=KFOLD, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=123, shuffle=True),
        TrainConfig(epochs=EPOCHS, lr=LR, device=DEVICE),
        dice_score,
    )

    class _FixedFactory:
        def build(self):
            return factory.build((S, S), best_specs)

    cv_result = cv.run(_FixedFactory(), tiles_ds)
    print(f"[CV-retrain] mean={cv_result.mean_score:.4f}  best_fold={cv_result.best_fold}")

    model = factory.build((S, S), best_specs)
    model.load_state_dict(cv_result.best_state_dict)
    model.eval().to(DEVICE)

    tiler = InferenceTiler(patch_size=S, overlap=OVERLAP, blend="mean")
    tester = FullImageTester(tiler=tiler, metric_fn=dice_score, device=DEVICE)
    report = tester.evaluate(model, test_ds)

    print("\n===== TEST =====")
    print(f"mean_{tester.metric_fn.__name__}: {report['mean_score']:.4f}")
    for item in report["per_image"]:
        print(f"  {item['stem']}: {item['score']:.4f}")

    print("\n===== Export to ONXX =====")
    onnx_cfg = OnnxExportConfig(
        onnx_path=Path("artifacts/unet_best.onnx"),
        input_shape=(3, 128, 128),
        dynamic_batch=True,
        dynamic_hw=True,
        opset=13,
        do_constant_folding=True,
        external_data=False,
        simplify=True,
        device="cuda",
        meta={
            "project": "MinimalSegmentationModel",
            "specs": str(best_specs),
            "num_classes": cfg.num_classes,
        },
        validate_with_ort=True,
        export_fp16=False,
    )

    OnnxExporter(onnx_cfg).export(model)

    print("\n===== All Done =====")


if __name__ == "__main__":
    main()
