import unittest
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import torch

from src.data.tile_dataset import TileDataset
from src.eval.evaluator_patch_cv import PatchCVEvaluator, PatchCVEvaluatorConfig
from src.img.tile_generator import TileGenerator
from src.img.tile_generator_config import TileGeneratorConfig
from src.learn.metrics import pixel_accuracy
from test.img.util import make_chessboard_rgb_and_mask


class BiasOnlyNet(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.head = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x): return self.head(x)


class BiasNetFactory:
    def __init__(self, in_channels=3, num_classes=2): self.in_channels = in_channels; self.num_classes = num_classes

    def build(self, input_size, encoder_specs):
        return BiasOnlyNet(self.in_channels, self.num_classes)


class TestPatchCVEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="evalcv_test_"))
        print(f"[EvalCVTest] temp dir: {self.tmpdir}")
        self.images_dir = self.tmpdir / "raw_images"
        self.masks_dir = self.tmpdir / "raw_masks"
        self.tiles_dir = self.tmpdir / "tiles"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        img1, msk1 = make_chessboard_rgb_and_mask(128, 128, 32)
        Image.fromarray(img1).save(self.images_dir / "a.png")
        Image.fromarray(msk1).save(self.masks_dir / "a.png")
        img2, msk2 = make_chessboard_rgb_and_mask(96, 96, 16)
        Image.fromarray(img2).save(self.images_dir / "b.png")
        Image.fromarray(msk2).save(self.masks_dir / "b.png")

        TileGenerator(TileGeneratorConfig(
            images_dir=self.images_dir, masks_dir=self.masks_dir,
            out_dir=self.tiles_dir, patch_size=64, overlap_pct=0.5
        )).run()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        print(f"[EvalCVTest] removed: {self.tmpdir}")

    def test_evaluator_runs_and_returns_score(self):
        ds = TileDataset(self.tiles_dir)
        factory = BiasNetFactory(in_channels=3, num_classes=2)
        evaluator = PatchCVEvaluator(ds, factory, PatchCVEvaluatorConfig(
            batch_size=4, num_workers=0, epochs=2, lr=1e-1, device="cpu", k_splits=3, metric_fn=pixel_accuracy
        ))
        score = evaluator.evaluate(input_size=64, lst_tuples=[(3, 16), (5, 32)])
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
