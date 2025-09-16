import unittest
import torch

from src.learn.metrics import pixel_accuracy, dice_score


class TestMetrics(unittest.TestCase):
    def test_pixel_accuracy(self):
        pred = torch.tensor([[0, 1], [1, 1]]).unsqueeze(0)  # [1,2,2]
        gt = torch.tensor([[0, 0], [1, 1]]).unsqueeze(0)
        acc = pixel_accuracy(pred, gt)
        self.assertAlmostEqual(acc, 0.75, places=6)

    def test_binary_dice(self):
        pred = torch.tensor([[0, 1], [1, 0]]).unsqueeze(0)
        gt = torch.tensor([[0, 1], [0, 1]]).unsqueeze(0)
        # binär: Klasse >0
        # pred= [[0,1],[1,0]], gt = [[0,1],[0,1]]
        # inter = 2 (Positionen (0,1) und (1,0) nicht beide 1; nur (0,1) und (1,1)? -> Rechnen wir:)
        # pred_bin = [[0,1],[1,0]], gt_bin = [[0,1],[0,1]]
        # inter = 1 (nur (0,1)), sum= (pred:2 + gt:2)=4 -> dice = 2*1/4 = 0.5
        d = dice_score(pred, gt, num_classes=None)
        self.assertAlmostEqual(d, 0.5, places=6)

    def test_multiclass_dice_macro(self):
        pred = torch.tensor([[0, 1, 2], [0, 1, 2]]).unsqueeze(0)  # [1,2,3]
        gt = torch.tensor([[0, 2, 2], [0, 1, 1]]).unsqueeze(0)
        # Klasse 0: pred==gt==0 an 2 Stellen → Dice 1.0
        # Klasse 1: pred hat 2, gt hat 2, Schnitt 1 → dice = 2*1/(2+2)=0.5
        # Klasse 2: pred hat 2, gt hat 2, Schnitt 1 → dice = 0.5
        # macro = (1.0 + 0.5 + 0.5)/3 = 0.666...
        d = dice_score(pred, gt, num_classes=3, average="macro")
        self.assertAlmostEqual(d, 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
