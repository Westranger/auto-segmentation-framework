import unittest, tempfile, shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from test.img.util import make_chessboard_rgb_and_mask
from src.data.rep_selector_inmemory import InMemoryRepresentativeIndexer, InMemRepSelectorConfig


class TestInMemoryRepSelector(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="inmem_rep_"))
        print("[InMemRepTest] tmp:", self.tmp)
        imgs = self.tmp / "raw_images";
        imgs.mkdir()
        msks = self.tmp / "raw_masks";
        msks.mkdir()

        def _to_pil_img(x):
            if isinstance(x, Image.Image):
                return x
            # x ist vermutlich ein np.ndarray
            arr = np.asarray(x)
            if arr.ndim == 2:  # grayscale
                return Image.fromarray(arr.astype(np.uint8), mode="L")
            elif arr.ndim == 3:  # H,W,C
                return Image.fromarray(arr.astype(np.uint8))
            else:
                raise ValueError(f"Unexpected array shape for image: {arr.shape}")

        for i in range(2):
            img_x, msk_x = make_chessboard_rgb_and_mask(1024, 768, 8 + 2 * i)
            img = _to_pil_img(img_x)  # RGB
            msk = _to_pil_img(msk_x)  # L
            if msk.mode not in ("L", "P"):
                msk = msk.convert("L")
            img.save(imgs / f"im_{i}.png")
            msk.save(msks / f"im_{i}.png")
        self.imgs, self.msks = imgs, msks

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        print("[InMemRepTest] removed:", self.tmp)

    def test_select_and_write(self):
        out_dir = self.tmp / "tiles_small"
        man = self.tmp / "manifest.json"
        man = self.tmp / "manifest.json"
        cfg = InMemRepSelectorConfig(
            images_dir=self.imgs, masks_dir=self.msks,
            patch_size=256, overlap_pct=0.5,
            target_k=12,
            device="cuda" if torch.cuda.is_available() else "cpu",
            similarity_prune=0.995,
            out_dir=out_dir, write_manifest=man,
            batch_embed=64,
            pretrained=True,  # <--- NEU: vermeidet Weights-Download im Test
        )

        selector = InMemoryRepresentativeIndexer(cfg)
        manifest = selector.run()
        self.assertTrue(1 <= len(manifest) <= 12)
        # Dateien existieren & haben richtige Größe
        for rec in manifest:
            ip = Path(rec["image_path"]);
            mp = Path(rec["mask_path"])
            self.assertTrue(ip.exists() and mp.exists())
            with Image.open(ip) as im, Image.open(mp) as mm:
                self.assertEqual(im.size, (256, 256))
                self.assertEqual(mm.size, (256, 256))
                self.assertIn(mm.mode, ("L", "P"))

        self.assertTrue(man.exists())


if __name__ == "__main__":
    unittest.main()
