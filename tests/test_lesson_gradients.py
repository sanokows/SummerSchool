"""Regression checks for the Langevin features and the full pathwise loss.

Run from the repository root: uv run --locked python -m unittest discover -s tests -v
"""

import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch


LESSON = Path(__file__).resolve().parents[1] / "L2-DiffusionSamplers"
sys.path.insert(0, str(LESSON))
torch.set_num_threads(2)
import lesson2_common as common


def solution_kernels():
    notebook = json.loads((LESSON / "diffusion_sampler_solution.ipynb").read_text())
    names = common.DiffusionKernels.__dataclass_fields__
    namespace = {
        "torch": torch,
        "Independent": torch.distributions.Independent,
        "Normal": torch.distributions.Normal,
        "PRIOR_STD": common.PRIOR_STD,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "solution kernels", "exec"), namespace)
    return common.DiffusionKernels(**{name: namespace[name] for name in names})


class LangevinGradientTests(unittest.TestCase):
    def test_target_score_and_its_derivative(self):
        states = (common.MODE_LOCATIONS[:3].double() + 0.15).requires_grad_()
        temperature = 0.7
        expected, = torch.autograd.grad(
            (common.target_reward(states) / temperature).sum(), states,
            create_graph=True,
        )
        actual = common.target_log_density_gradient(states, temperature)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(actual.requires_grad, "Pathwise updates need score derivatives.")
        self.assertTrue(torch.autograd.gradcheck(
            lambda x: common.target_log_density_gradient(x, temperature), (states,)
        ))

    def test_full_pathwise_gradient_with_a_nonzero_langevin_gate(self):
        # A single Gaussian gives a smooth target and keeps clipping inactive.
        # The gate starts at zero in class, so initialization alone misses this bug.
        with patch.multiple(common, MODE_LOCATIONS=torch.zeros(1, 2),
                            TARGET_STD=1.0, LOG_MIXTURE_WEIGHT=0.0):
            for learn_schedule in (False, True):
                with self.subTest(learn_schedule=learn_schedule):
                    torch.manual_seed(9)
                    sampler = common.DiffusionSampler(
                        solution_kernels(), num_steps=4, prior_std=2,
                        beta_start=0.01, beta_end=0.02, score_width=16,
                        use_langevin_preconditioning=True,
                        learn_schedule=learn_schedule,
                    ).double()
                    with torch.no_grad():
                        sampler.score_network.langevin_gate[-1].bias.fill_(0.2)
                    prior_noise = torch.randn(8, 2, dtype=torch.float64)
                    step_noises = torch.randn(4, 8, 2, dtype=torch.float64)

                    def objective():
                        path = sampler.sample_reverse_path(
                            8, reparameterize=True, prior_noise=prior_noise,
                            step_noises=step_noises,
                        )
                        return sampler.path_cost(path).mean()

                    parameters = [sampler.score_network.residual[-1].bias]
                    if learn_schedule:
                        parameters.append(sampler.schedule.increment_logits)
                    gradients = torch.autograd.grad(objective(), parameters)
                    epsilon = 1e-5
                    for parameter, gradient in zip(parameters, gradients):
                        value = parameter[0].item()
                        with torch.no_grad():
                            try:
                                parameter[0] = value + epsilon
                                plus = objective().item()
                                parameter[0] = value - epsilon
                                minus = objective().item()
                            finally:
                                parameter[0] = value
                        finite_difference = (plus - minus) / (2 * epsilon)
                        self.assertAlmostEqual(gradient[0].item(), finite_difference,
                                               delta=1e-7)


if __name__ == "__main__":
    unittest.main()
