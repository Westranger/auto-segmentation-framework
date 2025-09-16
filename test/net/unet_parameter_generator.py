# test_unet_param_generator_injected_unittest.py
import unittest
import optuna
from optuna.samplers import GridSampler

from dummy_evaluator import DummyProductEvaluator
from src.net.unet_parameter_generator import UNetParamGenerator


class TestUNetParamGeneratorWithEvaluator(unittest.TestCase):
    def test_finds_global_max_with_grid(self):
        net_input_sizes = (16, 32)
        num_layer_channels = (16, 32)
        filter_sizes = (3, 5)
        num_layers = (2,)

        evaluator = DummyProductEvaluator()
        gen = UNetParamGenerator(
            net_input_sizes=net_input_sizes,
            num_layer_channels=num_layer_channels,
            num_layers=num_layers,
            filter_sizes=filter_sizes,
            evaluator=evaluator,
        )

        tuple_comb = [(f, c) for f in filter_sizes for c in num_layer_channels]
        search_space = {
            "layer_0_ks": filter_sizes,
            "layer_0_ch": num_layer_channels,
            "layer_1_ks": filter_sizes,
            "layer_1_ch": num_layer_channels,
        }
        sampler = GridSampler(search_space)

        n_trials = (
                len(filter_sizes)  # layer_0_ks
                * len(num_layer_channels)  # layer_0_ch
                * len(filter_sizes)  # layer_1_ks
                * len(num_layer_channels)  # layer_1_ch
        )

        results = gen.generate(
            n_trials_per_setting=n_trials,
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            verbose=False,
        )

        best = results[(2, 32)]
        self.assertIsNotNone(best["best_value"])
        self.assertIsNotNone(best["best_params"])

        expected = ((5, 16), (5, 32))
        self.assertEqual(expected, best["best_params"])

        worse = results[(2, 16)]
        self.assertLess(worse["best_value"], best["best_value"])

    def test_sanity_rejects_even_kernel_or_non_increasing_channels(self):
        gen = UNetParamGenerator((32,), (16, 32), (3,), (3, 4))  # 4 = gerade → invalid
        self.assertFalse(gen.perform_param_sanity_check(32, [(3, 16), (4, 32), (3, 32)]))
        self.assertFalse(gen.perform_param_sanity_check(32, [(3, 16), (5, 16), (3, 32)]))
        self.assertTrue(gen.perform_param_sanity_check(32, [(3, 16), (5, 32), (3, 64)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
