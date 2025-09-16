import unittest
import torch
import torch.nn as nn

from src.net.unet import UNet


class TestUNet(unittest.TestCase):

    def test_forward_shape_four_blocks(self):
        specs = [(3, 64), (3, 128), (3, 256), (3, 512)]
        model = UNet(input_size=(256, 256), encoder_specs=specs, in_channels=3, num_classes=21)
        x = torch.randn(2, 3, 256, 256)
        y = model(x)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(y.shape, (2, 21, 256, 256))

    def test_invalid_input_size_raises(self):
        specs = [(3, 32), (3, 64), (3, 128), (3, 256)]
        with self.assertRaises(AssertionError):
            _ = UNet(input_size=(250, 250), encoder_specs=specs, in_channels=1, num_classes=2)

    def test_even_kernel_raises(self):
        specs = [(4, 32), (3, 64)]  # 4 ist invalid
        with self.assertRaises((AssertionError, ValueError)):
            _ = UNet(input_size=(128, 128), encoder_specs=specs, in_channels=3, num_classes=2)

    def test_single_block_no_pool(self):
        specs = [(3, 16)]
        model = UNet(input_size=(123, 77), encoder_specs=specs, in_channels=3, num_classes=5)
        x = torch.randn(1, 3, 123, 77)
        y = model(x)
        self.assertEqual(y.shape, (1, 5, 123, 77))

    def test_decoder_mirror_counts(self):
        specs = [(3, 32), (3, 64), (3, 128)]
        model = UNet(input_size=(128, 128), encoder_specs=specs, in_channels=3, num_classes=2)
        self.assertEqual(len(model.upconvs), len(specs) - 1)
        self.assertEqual(len(model.decoder), len(specs) - 1)

    def test_gradients_flow(self):
        specs = [(3, 32), (3, 64)]
        model = UNet(input_size=(128, 128), encoder_specs=specs, in_channels=3, num_classes=4)
        x = torch.randn(2, 3, 128, 128, requires_grad=False)
        y = model(x).sum()
        y.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None for g in grads), "Keine Gradienten – Backprop gescheitert.")

    def setUp(self):
        self.in_channels = 3
        self.num_classes = 2
        self.input_size = (64, 64)
        self.encoder_specs = [(3, 16), (3, 32)]
        self.batch_size = 2

        self.model = UNet(
            input_size=self.input_size,
            encoder_specs=self.encoder_specs,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            use_norm=True,
        )

    def test_forward_shape(self):
        x = torch.randn(self.batch_size, self.in_channels, *self.input_size)
        y = self.model(x)
        self.assertEqual(y.shape, (self.batch_size, self.num_classes, *self.input_size))

    def test_backward_one_step(self):
        x = torch.randn(self.batch_size, self.in_channels, *self.input_size)
        target = torch.ones(self.batch_size, 1, *self.input_size, dtype=torch.long)
        logits = self.model(x)
        loss_fn = nn.CrossEntropyLoss()
        loss_before = loss_fn(logits, target[:, 0])

        opt = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.0)
        opt.zero_grad(set_to_none=True)
        loss_before.backward()
        opt.step()

        with torch.no_grad():
            logits_after = self.model(x)
            loss_after = loss_fn(logits_after, target[:, 0])

        self.assertLessEqual(loss_after.item(), loss_before.item() * 1.10)

    def test_invalid_geometry_raises(self):
        with self.assertRaises(Exception):
            UNet(
                input_size=(64, 64),
                encoder_specs=[(4, 16), (3, 32)],
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            )

        with self.assertRaises(Exception):
            UNet(
                input_size=(66, 66),  # 66 % 4 != 0
                encoder_specs=[(3, 16), (3, 32), (3, 64)],
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
