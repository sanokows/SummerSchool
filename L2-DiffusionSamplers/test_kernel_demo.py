"""Checks for the interactive exercises and bundled learned score."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

import torch

from lesson2_kernel_demo import (
    DemoConfig, forward_demo, load_pretrained_demo, marginal_parameters,
    mixture_score, reverse_demo, simulate_forward, simulate_reverse,
)
from test_diffusion_kernels import load_notebook_kernels


ROOT = Path(__file__).resolve().parent


class KernelDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "diffusion_sampler_solution.ipynb").exists():
            raise unittest.SkipTest("Instructor kernel checks require the local solution notebook.")
        torch.set_num_threads(1)
        cls.kernels = load_notebook_kernels()
        cls.pretrained = load_pretrained_demo(ROOT / "assets/bimodal_score.pt")

    def test_forward_controls_reach_the_student_kernel(self):
        for schedule in ("constant", "linear", "cosine"):
            calls = []
            def record(state, delta, **kwargs):
                calls.append((float(delta), kwargs["prior_std"], state.shape[-1]))
                return self.kernels.forward_sde_step(state, delta, **kwargs)

            config = DemoConfig(prior_std=2.3, num_steps=40, schedule=schedule,
                                delta_start=0.003, delta_end=0.04)
            path = simulate_forward(record, config, num_samples=256)
            self.assertEqual(path.states.shape, (41, 256, 1))
            self.assertEqual(calls, [(float(d), 2.3, 1) for d in config.deltas()[:-1]])
            # This path must be actual student output, not an analytic replacement.
            unchanged = simulate_forward(lambda state, *args, **kwargs: state, config, num_samples=256)
            torch.testing.assert_close(unchanged.states[0], unchanged.states[-1])
            self.assertFalse(torch.equal(path.states[-1], unchanged.states[-1]))

    def test_forward_endpoint_reaches_the_gaussian_prior(self):
        config = DemoConfig()
        path = simulate_forward(self.kernels.forward_sde_step, config, num_samples=8192)
        mean, variance = marginal_parameters(config)
        self.assertLess(float(mean[-1]), 0.01)
        self.assertAlmostEqual(float(variance[-1]), config.prior_std**2, delta=0.01)
        endpoint = path.states[-1, :, 0]
        self.assertLess(abs(float(endpoint.mean())), 0.05)
        self.assertAlmostEqual(float(endpoint.std()), 1.0, delta=0.05)
        self.assertTrue(torch.equal(path.states, simulate_forward(
            self.kernels.forward_sde_step, config, num_samples=8192).states))

    def test_checkpoint_is_frozen_and_approximates_the_score(self):
        score = self.pretrained.score
        self.assertFalse(score.training)
        self.assertTrue(all(not p.requires_grad for p in score.parameters()))
        with self.assertRaises(FrozenInstanceError):
            self.pretrained.config.prior_std = 7
        means, variances = marginal_parameters(self.pretrained.config)
        rng = torch.Generator().manual_seed(23)
        k = torch.randint(0, 601, (4096, 1), generator=rng)
        signs = 2 * torch.randint(0, 2, k.shape, generator=rng) - 1
        states = signs * means[k] + variances[k].sqrt() * torch.randn(k.shape, generator=rng)
        error = score(states, k, 600) - mixture_score(states, means[k], variances[k])
        self.assertLess(float((variances[k] * error.square()).mean().sqrt()), 0.03)

    def test_notebook_presets_match_the_pretrained_linear_schedule(self):
        config = self.pretrained.config
        self.assertEqual(config, DemoConfig())
        self.assertEqual(config.schedule, "linear")
        self.assertGreater(config.delta_end, config.delta_start)
        for role in ("student", "solution"):
            notebook = json.loads((ROOT / f"diffusion_sampler_{role}.ipynb").read_text())
            panel = next("".join(cell["source"]) for cell in notebook["cells"]
                         if "".join(cell["source"]).startswith("# HYPERPARAMETERS"))
            settings = {}
            exec(compile(panel, f"{role} hyperparameters", "exec"), settings)
            preset = DemoConfig(
                prior_std=settings["DEMO_PRIOR_STD"], num_steps=settings["DEMO_DIFFUSION_STEPS"],
                schedule=settings["DEMO_SCHEDULE"], delta_start=settings["DEMO_DELTA_START"],
                delta_end=settings["DEMO_DELTA_END"], mode_location=settings["DEMO_MODE_LOCATION"],
                component_std=settings["DEMO_COMPONENT_STD"],
            )
            self.assertEqual(preset, config)
            self.assertEqual(settings["DEMO_PARTICLES"], 4096)

    def test_reverse_recovers_both_modes_with_the_checkpoint_settings(self):
        calls = []
        def record(score, state, k, delta, total, **kwargs):
            calls.append((k, float(delta), total, kwargs["prior_std"]))
            return self.kernels.reverse_sde_step(score, state, k, delta, total, **kwargs)

        path = simulate_reverse(record, self.pretrained, num_samples=4096)
        config = self.pretrained.config
        self.assertEqual([c[0] for c in calls], list(range(600, 0, -1)))
        self.assertTrue(all(c[2:] == (600, 1.0) for c in calls))
        torch.testing.assert_close(torch.tensor([c[1] for c in calls]), config.deltas()[1:].flip(0))
        samples = path.states[0, :, 0]
        fraction_left = float((samples < 0).float().mean())
        self.assertAlmostEqual(fraction_left, 0.5, delta=0.04)
        for subset, expected_mean in [(samples[samples < 0], -3), (samples[samples >= 0], 3)]:
            self.assertAlmostEqual(float(subset.mean()), expected_mean, delta=0.12)
            self.assertAlmostEqual(float(subset.std()), 0.55, delta=0.10)
        self.assertLess(float((samples.abs() < 1).float().mean()), 0.015)

    def test_widget_callbacks_scrubbing_and_unimplemented_kernels(self):
        explorer = forward_demo(self.kernels.forward_sde_step,
                                config=DemoConfig(num_steps=20), num_samples=128, frame_count=9)
        initial = bytes(explorer.image.value)
        explorer.frame.value = explorer.frame.max
        self.assertNotEqual(initial, bytes(explorer.image.value))
        explorer.controls["prior_std"].value = 2
        explorer.controls["schedule"].value = "linear"
        explorer.controls["delta_start"].value = 0.005
        self.assertEqual(explorer.path.config.prior_std, 2)
        self.assertEqual(explorer.path.config.delta_start, 0.005)
        self.assertTrue(bytes(explorer.image.value).startswith(b"\x89PNG"))
        fixed = reverse_demo(self.kernels.reverse_sde_step, self.pretrained,
                             num_samples=32, frame_count=9)
        self.assertEqual(fixed.controls, {})
        self.assertEqual(fixed.path.config, self.pretrained.config)
        self.assertIn("linear δ₀=0.005 → δK=0.035", fixed.widget.children[0].value)
        fixed.frame.value = fixed.frame.max
        self.assertEqual(fixed._steps[-1], 0)
        student = load_notebook_kernels("diffusion_sampler_student.ipynb")
        missing = forward_demo(student.forward_sde_step, config=DemoConfig(num_steps=20), num_samples=32)
        self.assertIsNone(missing.path)
        self.assertIn("NotImplementedError", missing.status.value)
        self.assertTrue(missing.frame.disabled)


if __name__ == "__main__":
    unittest.main()
