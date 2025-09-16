import unittest
from pathlib import Path
import tempfile
import shutil
from PIL import Image

import torch

from src.data.full_image_dataset import FullImageDataset
from src.eval.full_image_tester import FullImageTester
from src.learn.metrics import pixel_accuracy
from src.model.inference_tiler import InferenceTiler
from test.img.util import make_chessboard_rgb_and_mask


class ConstantModel(torch.nn.Module):

    def __init__(self, in_ch=3, num_classes=2, bias0=0.0, bias1=1.0):
        super().__init__()
        self.head = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
        torch.nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias[:] = torch.tensor([bias0, bias1])

    def forward(self, x):
        return self.head(x)


class TestFullImageTester(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="fullimg_test_"))
        print(f"[FullImageTester] temp dir: {self.tmpdir}")
        self.images_dir = self.tmpdir / "images"
        self.masks_dir = self.tmpdir / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        for stem in ("a", "b"):
            img, msk = make_chessboard_rgb_and_mask(128, 128, 32)
            Image.fromarray(img).save(self.images_dir / f"{stem}.png")
            Image.fromarray(msk).save(self.masks_dir / f"{stem}.png")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        print(f"[FullImageTester] removed: {self.tmpdir}")

    def test_full_image_evaluation_accuracy(self):
        ds = FullImageDataset(self.images_dir, self.masks_dir)
        tiler = InferenceTiler(patch_size=64, overlap=0.5)
        tester = FullImageTester(tiler=tiler, metric_fn=pixel_accuracy, device="cpu")

        model = ConstantModel().eval()
        report = tester.evaluate(model, ds)

        self.assertIn("mean_score", report)
        self.assertIn("per_image", report)
        self.assertEqual(len(report["per_image"]), 2)
        self.assertTrue(0.45 <= report["mean_score"] <= 0.55)
        for item in report["per_image"]:
            self.assertTrue(0.45 <= item["score"] <= 0.55)


if __name__ == "__main__":
    unittest.main(verbosity=2)
