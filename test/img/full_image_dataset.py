import unittest
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import torch

from src.data.full_image_dataset import FullImageDataset
from util import make_chessboard_rgb_and_mask


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ds_test_"))
        print(f"[DatasetsTest] temp dir: {self.tmpdir}")
        self.images_dir = self.tmpdir / "raw_images"
        self.masks_dir = self.tmpdir / "raw_masks"
        self.tiles_dir = self.tmpdir / "tiles"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        img1, msk1 = make_chessboard_rgb_and_mask(128, 128, 32)
        img2, msk2 = make_chessboard_rgb_and_mask(96, 160, 16)
        Image.fromarray(img1).save(self.images_dir / "a.png")
        Image.fromarray(msk1).save(self.masks_dir / "a.png")
        Image.fromarray(img2).save(self.images_dir / "b.png")
        Image.fromarray(msk2).save(self.masks_dir / "b.png")

    def tearDown(self):
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            print(f"[DatasetsTest] removed: {self.tmpdir}")

    def test_full_image_dataset(self):
        ds = FullImageDataset(self.images_dir, self.masks_dir)
        self.assertEqual(len(ds), 2)
        x, y, stem = ds[0]
        self.assertIsInstance(stem, str)
        self.assertEqual(x.ndim, 3)  # [C,H,W]
        self.assertEqual(y.ndim, 3)  # [1,H,W]
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(y.shape[0], 1)
        self.assertTrue((x >= 0).all() and (x <= 1).all())
        self.assertTrue(y.dtype in (torch.int64, torch.long))


if __name__ == "__main__":
    unittest.main(verbosity=2)
