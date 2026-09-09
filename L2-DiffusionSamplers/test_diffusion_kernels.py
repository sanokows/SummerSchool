"""Instructor regression checks for the discrete VP equations and training.

Run: .venv/bin/python -m unittest discover -s L2-DiffusionSamplers -v
Requires the local solution notebook, which is intentionally excluded from Git.
"""

import ast
import inspect
import json
import math
from pathlib import Path
import unittest

import torch
from torch.distributions import Independent, Normal

import lesson2_common as common


ROOT = Path(__file__).resolve().parent
KERNEL_NAMES = tuple(common.DiffusionKernels.__dataclass_fields__)


def load_notebook_kernels(filename="diffusion_sampler_solution.ipynb", *, prior_std=common.PRIOR_STD):
    notebook = json.loads((ROOT / filename).read_text())
    namespace = {"torch": torch, "Independent": Independent, "Normal": Normal,
                 "PRIOR_STD": prior_std}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        functions = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name in KERNEL_NAMES]
        if functions:
            exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    return common.DiffusionKernels(**{name: namespace[name] for name in KERNEL_NAMES})


def gaussian_log_density(value, mean, variance):
    return -0.5 * (((value - mean).square() / variance)
                  + torch.log(2 * math.pi * variance)).sum(-1)


class DiscreteVPKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT / "diffusion_sampler_solution.ipynb").is_file():
            raise unittest.SkipTest(
                "Instructor checks require the local diffusion_sampler_solution.ipynb "
                "(intentionally excluded from Git)."
            )
        torch.set_num_threads(1)
        cls.kernels = load_notebook_kernels()

    def test_notebooks_have_matching_interfaces(self):
        student = load_notebook_kernels("diffusion_sampler_student.ipynb")
        for name in KERNEL_NAMES:
            self.assertEqual(inspect.signature(getattr(student, name)),
                             inspect.signature(getattr(self.kernels, name)))

    def test_discrete_means_variances_and_log_probabilities(self):
        previous = torch.tensor([[1.0, -2.0], [0.3, 1.7]], dtype=torch.float64)
        noise = torch.tensor([[0.2, -0.4], [0.8, -0.5]], dtype=torch.float64)
        delta = torch.tensor(0.08, dtype=torch.float64)
        variance = common.PRIOR_VARIANCE * delta
        forward_mean = (1 - delta / 2) * previous
        noisy = self.kernels.forward_sde_step(previous, delta, reparameterize=True, noise=noise)
        torch.testing.assert_close(noisy, forward_mean + variance.sqrt() * noise)
        torch.testing.assert_close(
            self.kernels.forward_kernel_log_prob(previous, noisy, delta),
            gaussian_log_density(noisy, forward_mean, variance),
        )

        def score(states, k, total):
            self.assertEqual((k, total), (3, 5))
            return 0.03 * states + 0.02

        reverse_mean = (1 + delta / 2) * noisy + variance * score(noisy, 3, 5)
        recovered = self.kernels.reverse_sde_step(score, noisy, 3, delta, 5,
                                                reparameterize=True, noise=noise)
        torch.testing.assert_close(recovered, reverse_mean + variance.sqrt() * noise)
        torch.testing.assert_close(
            self.kernels.reverse_kernel_log_prob(score, recovered, noisy, 3, delta, 5),
            gaussian_log_density(recovered, reverse_mean, variance),
        )

    def test_sampling_moments_and_gradient_flags(self):
        torch.manual_seed(21)
        count = 50000
        previous = torch.tensor([1.0, -2.0]).repeat(count, 1).requires_grad_()
        delta = torch.tensor(0.08, requires_grad=True)
        samples = self.kernels.forward_sde_step(previous, delta, reparameterize=False)
        variance = common.PRIOR_VARIANCE * delta.detach()
        mean = (1 - delta.detach() / 2) * previous[0].detach()
        self.assertTrue(torch.all((samples.mean(0) - mean).abs() < 5 * (variance / count).sqrt()))
        self.assertTrue(torch.all((samples.var(0) / variance - 1).abs() < 5 * math.sqrt(2 / count)))
        self.assertFalse(samples.requires_grad)
        small = previous[:2]
        score = lambda states, k, total: -states / common.PRIOR_VARIANCE
        for fixed_noise in (None, torch.ones_like(small)):
            for reparameterize in (False, True):
                forward = self.kernels.forward_sde_step(
                    small, delta, reparameterize=reparameterize, noise=fixed_noise)
                reverse = self.kernels.reverse_sde_step(
                    score, small, 2, delta, 4, reparameterize=reparameterize, noise=fixed_noise)
                self.assertEqual(forward.requires_grad, reparameterize)
                self.assertEqual(reverse.requires_grad, reparameterize)

    def test_kernel_derivatives_against_finite_differences(self):
        torch.manual_seed(5)
        previous = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        noisy = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        delta = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
        weight = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
        noise = torch.randn_like(previous)
        checks = [
            (lambda x, d: self.kernels.forward_sde_step(x, d, reparameterize=True, noise=noise),
             (previous, delta)),
            (self.kernels.forward_kernel_log_prob, (previous, noisy, delta)),
            (lambda x, d, w: self.kernels.reverse_sde_step(
                lambda states, k, total: w * states, x, 3, d, 5,
                reparameterize=True, noise=noise), (noisy, delta, weight)),
            (lambda y, x, d, w: self.kernels.reverse_kernel_log_prob(
                lambda states, k, total: w * states, y, x, 3, d, 5),
             (previous, noisy, delta, weight)),
        ]
        for function, inputs in checks:
            self.assertTrue(torch.autograd.gradcheck(function, inputs))

    def test_schedule_coefficients_and_signal(self):
        for learn in (False, True):
            schedule = common.VPSchedule(num_steps=4, learn_schedule=learn)
            if learn:
                with torch.no_grad():
                    schedule.increment_logits.copy_(torch.tensor([-1.0, 0.5, 1.0, -0.3]))
            betas = schedule()
            self.assertEqual(betas.shape, (5,))
            self.assertTrue(torch.all(betas.diff() > 0))
            torch.testing.assert_close(betas[[0, -1]], torch.tensor([common.BETA_START, common.BETA_END]))
            squared_signal = torch.tensor(1.0)
            expected = []
            for beta in betas[:-1]:
                squared_signal = squared_signal * (1 - beta * common.DIFFUSION_STEP_SIZE / 2)**2
                expected.append(squared_signal)
            torch.testing.assert_close(schedule.cumulative_signal(), torch.stack(expected))

    def test_reverse_path_uses_adjacent_coefficients(self):
        calls = {"sample": [], "reverse_log": [], "forward_log": []}

        def reverse_step(network, state, k, delta, total, **kwargs):
            calls["sample"].append((k, float(delta)))
            return self.kernels.reverse_sde_step(network, state, k, delta, total, **kwargs)

        def reverse_log(network, previous, state, k, delta, total, **kwargs):
            calls["reverse_log"].append((k, float(delta)))
            return self.kernels.reverse_kernel_log_prob(network, previous, state, k, delta, total, **kwargs)

        def forward_log(previous, state, delta, **kwargs):
            calls["forward_log"].append(float(delta))
            return self.kernels.forward_kernel_log_prob(previous, state, delta, **kwargs)

        kernels = common.DiffusionKernels(self.kernels.forward_sde_step, reverse_step,
                                         forward_log, reverse_log)
        sampler = common.DiffusionSampler(kernels, num_steps=4)
        deltas = sampler.schedule() * common.DIFFUSION_STEP_SIZE
        prior_noise = torch.tensor([[0.2, -0.4], [0.3, 0.7]])
        step_noises = torch.zeros(4, 2, 2)
        path = sampler.sample_reverse_path(2, reparameterize=True,
                                          prior_noise=prior_noise, step_noises=step_noises)
        self.assertEqual(path.states.shape, (5, 2, 2))
        self.assertEqual(calls["sample"], [(k, float(deltas[k])) for k in range(4, 0, -1)])
        self.assertEqual(calls["reverse_log"], calls["sample"])
        self.assertEqual(calls["forward_log"], [float(deltas[k-1]) for k in range(4, 0, -1)])
        torch.testing.assert_close(path.states[-1], common.PRIOR_STD * prior_noise)
        expected_q = common.prior_log_prob(path.states[-1])
        expected_p = torch.zeros(2)
        for k in range(1, 5):
            # At initialization the network equals the analytic prior score.
            reverse_mean = (1 - deltas[k] / 2) * path.states[k]
            torch.testing.assert_close(path.states[k-1], reverse_mean)
            expected_q = expected_q + gaussian_log_density(
                path.states[k-1], reverse_mean, common.PRIOR_VARIANCE * deltas[k])
            expected_p = expected_p + gaussian_log_density(
                path.states[k], (1-deltas[k-1]/2)*path.states[k-1],
                common.PRIOR_VARIANCE*deltas[k-1])
        torch.testing.assert_close(path.log_q, expected_q)
        torch.testing.assert_close(path.log_forward, expected_p)

    def test_both_estimators_and_training_with_all_flags(self):
        for estimator in ("reparameterization", "log_derivative"):
            for learn in (False, True):
                for langevin in (False, True):
                    with self.subTest(estimator=estimator, learn=learn, langevin=langevin):
                        torch.manual_seed(19)
                        sampler = common.DiffusionSampler(
                            self.kernels, num_steps=4, learn_schedule=learn,
                            use_langevin_preconditioning=langevin)
                        loss_fn = (common.reparameterization_path_loss if estimator == "reparameterization"
                                   else common.log_derivative_path_loss)
                        loss = loss_fn(sampler, 32)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        for parameter in sampler.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                        if learn:
                            self.assertGreater(sampler.schedule.increment_logits.grad.abs().sum().item(), 0)
                        trained, history = common.train_sampler(
                            self.kernels, gradient_estimator=estimator, learn_schedule=learn,
                            use_langevin_preconditioning=langevin, steps=8, batch_size=32,
                            log_every=4, num_animation_samples=32)
                        self.assertTrue(all(math.isfinite(v) for v in history["evaluation_loss"]))
                        self.assertTrue(all(math.isfinite(v) for v in history["loss"][1:]))
                        self.assertTrue(all(torch.isfinite(v).all() for v in history["samples"]))
                        self.assertFalse(torch.equal(history["samples"][0], history["samples"][-1]))
                        self.assertTrue(torch.all(trained.schedule().diff() > 0))
                        self.assertEqual(history["temperature"], [common.TEMPERATURE] * 3)

    def test_configurable_target_prior_schedule_and_training(self):
        """Nondefault settings must reach the target, kernels, and optimizer run."""
        try:
            common.configure_target(num_modes=3, mode_bound=5, target_variance=0.4,
                                    seed=9, plot_bound=8, grid_resolution=16,
                                    contour_levels=8, reward_plot_range=10)
            self.assertEqual(common.MODE_LOCATIONS.shape, (3, 2))
            self.assertEqual(common.GRID_X.shape, (16, 16))
            self.assertEqual(len(common.REWARD_LEVELS), 8)
            self.assertAlmostEqual(common.TARGET_STD**2, 0.4)
            expected_locations = 10 * torch.rand(3, 2, generator=torch.Generator().manual_seed(9)) - 5
            torch.testing.assert_close(common.MODE_LOCATIONS, expected_locations)

            prior_std = 7.0
            # Defaults stay at 30: the sampler must explicitly pass its own scale.
            kernels = load_notebook_kernels()
            for estimator in ("reparameterization", "log_derivative"):
                sampler, history = common.train_sampler(
                    kernels, gradient_estimator=estimator, learn_schedule=True,
                    use_langevin_preconditioning=True, num_steps=5, prior_std=prior_std,
                    beta_start=0.01, beta_end=0.04, diffusion_step_size=0.4,
                    score_width=24, langevin_gradient_clip=11, langevin_drift_clip=23,
                    temperature=0.6,
                    steps=4, batch_size=16, learning_rate=1e-3, schedule_learning_rate=2e-3,
                    max_grad_norm=2, log_every=2, num_animation_samples=12, animation_seed=17,
                )
                self.assertEqual(sampler.num_steps, 5)
                self.assertEqual(sampler.prior_std, prior_std)
                self.assertEqual(sampler.score_network.prior_variance, prior_std**2)
                self.assertEqual(sampler.score_network.residual[0].out_features, 24)
                self.assertEqual(sampler.score_network.langevin_gradient_clip, 11)
                self.assertEqual(sampler.score_network.langevin_drift_clip, 23)
                self.assertEqual(sampler.schedule.step_size, 0.4)
                self.assertEqual(history["samples"][-1].shape, (12, 2))
                torch.testing.assert_close(sampler.schedule()[[0, -1]], torch.tensor([0.01, 0.04]))
                self.assertEqual(history["temperature"], [0.6, 0.6, 0.6])
                self.assertFalse(torch.equal(history["betas"][0], history["betas"][-1]))
                self.assertTrue(all(math.isfinite(x) for x in history["evaluation_loss"]))

                prior_noise = torch.ones(12, 2)
                path = sampler.sample_reverse_path(12, reparameterize=True, prior_noise=prior_noise)
                torch.testing.assert_close(path.states[-1], prior_std * prior_noise)
                reference = Independent(Normal(torch.zeros(2), prior_std), 1)
                torch.testing.assert_close(common.prior_log_prob(path.states[-1], prior_std),
                                           reference.log_prob(path.states[-1]))
        finally:
            common.configure_target()


if __name__ == "__main__":
    unittest.main()
