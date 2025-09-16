import unittest
import torch

from src.net.double_conv import DoubleConv


class TestDoubleConv(unittest.TestCase):
    def test_forward_shape(self):
        in_ch = 3
        out_ch = 16
        kernel_size = 3
        H, W = 64, 64

        block = DoubleConv(in_ch, out_ch, kernel_size=kernel_size)
        x = torch.randn(2, in_ch, H, W)
        y = block(x)

        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (2, out_ch, H, W))


if __name__ == "__main__":
    unittest.main()
