import unittest
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

from metrics import pixel_accuracy
from src.learn.cross_validator import CrossValidator, CVConfig
from src.learn.trainer import Trainer, TrainConfig


class AllOnesSegDataset(Dataset):
    def __init__(self, n: int = 32, H: int = 32, W: int = 32, in_ch: int = 3):
        self.n, self.H, self.W, self.in_ch = n, H, W, in_ch

    def __len__(self): return self.n

    def __getitem__(self, idx):
        x = torch.rand(self.in_ch, self.H, self.W)  # [C,H,W] in 0..1
        y = torch.ones(1, self.H, self.W, dtype=torch.long)
        return x, y


class BiasOnlyNet(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=2):
        super().__init__()
        self.head = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x): return self.head(x)


@dataclass
class BiasNetFactory:
    in_ch: int = 3
    num_classes: int = 2

    def build(self):
        return BiasOnlyNet(self.in_ch, self.num_classes)


class TestTrainerAndCV(unittest.TestCase):
    def test_trainer_improves_on_all_ones(self):
        ds = AllOnesSegDataset(n=16)
        loader_tr = DataLoader(ds, batch_size=4, shuffle=True)
        loader_va = DataLoader(ds, batch_size=4, shuffle=False)

        model = BiasOnlyNet()
        trainer = Trainer(TrainConfig(epochs=3, lr=1e-1, device="cpu"), pixel_accuracy)
        result = trainer.train(model, loader_tr, loader_va)

        self.assertGreaterEqual(result.best_score, 0.9)
        self.assertTrue(isinstance(result.best_state_dict, dict))
        self.assertGreater(len(result.history), 0)

    def test_cross_validator_aggregates(self):
        ds = AllOnesSegDataset(n=20)
        cv = CrossValidator(
            CVConfig(k_splits=5, batch_size=4, num_workers=0, seed=123, shuffle=True),
            TrainConfig(epochs=2, lr=1e-1, device="cpu"),
            pixel_accuracy
        )
        mf = BiasNetFactory()
        result = cv.run(mf, ds)

        self.assertEqual(len(result.fold_scores), 5)
        self.assertTrue(0.0 <= result.mean_score <= 1.0)
        self.assertGreaterEqual(result.mean_score, 0.8)  # sollte gut sein
        self.assertIsInstance(result.best_state_dict, dict)
        self.assertGreaterEqual(result.best_fold, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
