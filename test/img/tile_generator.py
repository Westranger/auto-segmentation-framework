import unittest
from pathlib import Path
import tempfile
import shutil

import numpy as np
from PIL import Image

from src.img.tile_generator import TileGenerator
from src.img.tile_generator_config import TileGeneratorConfig
from util import make_chessboard_rgb_and_mask


class TestTileGenerator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="tilegen_test_"))
        print(f"[TileGeneratorTest] Using temporary dir: {self.tmpdir}")

        self.images_dir = self.tmpdir / "images"
        self.masks_dir = self.tmpdir / "masks"
        self.out_dir = self.tmpdir / "out"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        img_np, msk_np = make_chessboard_rgb_and_mask(H=128, W=128, cell=32)
        Image.fromarray(img_np).save(self.images_dir / "board.png")
        Image.fromarray(msk_np).save(self.masks_dir / "board.png")

    def tearDown(self):
        if self.tmpdir.exists():
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            print(f"[TileGeneratorTest] Removed temporary dir: {self.tmpdir}")

    def test_tiles_overlap_50pct_content_and_count(self):
        cfg = TileGeneratorConfig(
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            out_dir=self.out_dir,
            patch_size=64,
            overlap_pct=0.5,
        )
        gen = TileGenerator(cfg)
        gen.run()

        tiles_img_dir = self.out_dir / "images"
        tiles_msk_dir = self.out_dir / "masks"
        self.assertTrue(tiles_img_dir.exists())
        self.assertTrue(tiles_msk_dir.exists())

        expected_coords = [(y, x) for y in (0, 32, 64) for x in (0, 32, 64)]
        expected_filenames = [f"board_y{y}_x{x}.png" for (y, x) in expected_coords]

        files_img = sorted([p.name for p in tiles_img_dir.glob("*.png")])
        files_msk = sorted([p.name for p in tiles_msk_dir.glob("*.png")])

        self.assertEqual(files_img, expected_filenames)
        self.assertEqual(files_msk, expected_filenames)

        orig_img = np.asarray(Image.open(self.images_dir / "board.png"))
        orig_msk = np.asarray(Image.open(self.masks_dir / "board.png"))
        for y, x in expected_coords:
            patch_img = np.asarray(Image.open(tiles_img_dir / f"board_y{y}_x{x}.png"))
            patch_msk = np.asarray(Image.open(tiles_msk_dir / f"board_y{y}_x{x}.png"))

            self.assertEqual(tuple(patch_img[0, 0]), tuple(orig_img[y, x]))
            self.assertEqual(int(patch_msk[0, 0]), int(orig_msk[y, x]))

            self.assertEqual(patch_img.shape, (64, 64, 3))
            self.assertEqual(patch_msk.shape, (64, 64))

        for (y, x) in [(0, 32), (32, 0), (32, 32), (64, 64)]:
            patch = np.asarray(Image.open(tiles_msk_dir / f"board_y{y}_x{x}.png"))
            self.assertEqual(int(patch[31, 31]), int(orig_msk[y + 31, x + 31]))


def test_multiple_inputs_listing_and_manifest(self):
    img2, msk2 = make_chessboard_rgb_and_mask(H=96, W=96, cell=24)
    Image.fromarray(img2).save(self.images_dir / "board2.png")
    Image.fromarray(msk2).save(self.masks_dir / "board2.png")

    cfg = TileGeneratorConfig(
        images_dir=self.images_dir,
        masks_dir=self.masks_dir,
        out_dir=self.out_dir,
        patch_size=32,
        overlap_pct=0.5,
    )
    gen = TileGenerator(cfg)
    gen.run()

    img_tiles = gen.list_image_tiles()
    msk_tiles = gen.list_mask_tiles()
    self.assertGreater(len(img_tiles), 0)
    self.assertEqual([p.name for p in img_tiles], [p.name for p in msk_tiles])

    pairs = list(gen.iter_tile_pairs())
    self.assertEqual(len(pairs), len(img_tiles))
    self.assertTrue(all(a.name == b.name for a, b in pairs))

    rows = gen.build_manifest(save_csv=False)
    self.assertEqual(len(rows), len(pairs))
    for r in rows[:5]:
        self.assertIn("original_stem", r)
        self.assertIn("y", r)
        self.assertIn("x", r)
        self.assertIn("image_path", r)
        self.assertIn("mask_path", r)
        self.assertEqual(Path(r["image_path"]).name, Path(r["mask_path"]).name)

    any_pair = pairs[0]
    name = any_pair[0].name
    matched = [r for r in rows if Path(r["image_path"]).name == name]
    self.assertEqual(len(matched), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
