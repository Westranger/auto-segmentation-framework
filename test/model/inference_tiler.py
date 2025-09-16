import unittest
import torch

from src.model.inference_tiler import InferenceTiler


class ConstantModel(torch.nn.Module):
    def __init__(self, in_ch=3, num_classes=2, bias0=0.0, bias1=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.head = torch.nn.Conv2d(in_ch, num_classes, kernel_size=1, bias=True)
        torch.nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias[:] = torch.tensor([bias0, bias1])

    def forward(self, x):
        return self.head(x)


class TestInferenceTiler(unittest.TestCase):
    def test_constant_model_predicts_all_ones(self):
        model = ConstantModel().eval()
        image = torch.rand(3, 190, 260)
        tiler = InferenceTiler(patch_size=64, overlap=0.5)
        pred = tiler.predict_mask(model, image)
        self.assertEqual(pred.shape, (1, 190, 260))
        self.assertTrue(torch.all(pred == 1))

    def test_full_coverage(self):
        model = ConstantModel().eval()
        image = torch.rand(3, 101, 113)
        tiler = InferenceTiler(patch_size=32, overlap=0.25)
        logits = tiler.predict_logits(model, image)
        self.assertEqual(logits.shape, (2, 101, 113))  # [K,H,W]
        pred = torch.argmax(logits, dim=0)
        self.assertTrue(torch.all(pred == 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
