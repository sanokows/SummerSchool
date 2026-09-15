"""Shared infrastructure for Lesson 2: diffusion samplers.

The four transition-kernel functions are intentionally supplied by each
notebook.  Everything else -- the reward, score network, two gradient
estimators, training loop, diagnostics, and GIF writer -- lives here so the
student exercise stays focused on the forward and reverse VP kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import animation
import torch
from torch import nn


STATE_DIMENSION = 2
NUM_DIFFUSION_STEPS = 24
TEMPERATURE = 1.0
BETA_START = 5.0e-4
BETA_END = 2.0e-3
DIFFUSION_STEP_SIZE = 1.0  # delta_k = beta_k * Delta_k
PRIOR_STD = 30.0
PRIOR_VARIANCE = PRIOR_STD**2
LANGEVIN_GRADIENT_CLIP = 1.0e2
LANGEVIN_DRIFT_CLIP = 1.0e4

NUM_MODES = 40
MODE_BOUND = 40.0
TARGET_VARIANCE = 1.0
TARGET_STD = math.sqrt(TARGET_VARIANCE)
LOG_MIXTURE_WEIGHT = -math.log(NUM_MODES)


def target_reward(states: torch.Tensor) -> torch.Tensor:
    """Log density of the configured equal-weight GMM target."""
    locations = MODE_LOCATIONS.to(device=states.device, dtype=states.dtype)
    standardized = (states.unsqueeze(-2) - locations) / TARGET_STD
    component_log_prob = (
        -0.5 * standardized.square().sum(dim=-1)
        - STATE_DIMENSION * math.log(TARGET_STD * math.sqrt(2.0 * math.pi))
        + LOG_MIXTURE_WEIGHT
    )
    return torch.logsumexp(component_log_prob, dim=-1)


def target_log_density_gradient(
    states: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Differentiable analytic gradient of log pi_T(x) = R(x) / temperature."""
    locations = MODE_LOCATIONS.to(device=states.device, dtype=states.dtype)
    differences = locations - states.unsqueeze(-2)
    component_logits = (
        -0.5 * differences.square().sum(dim=-1) / TARGET_STD**2
        + LOG_MIXTURE_WEIGHT
    )
    responsibilities = component_logits.softmax(dim=-1)
    gradient = (
        responsibilities.unsqueeze(-1) * differences / TARGET_STD**2
    ).sum(dim=-2)
    # Earlier denoising decisions affect these states. Pathwise updates need
    # the derivative of this score feature through the rest of the chain.
    return gradient / temperature


def configure_target(
    *,
    num_modes: int = NUM_MODES,
    mode_bound: float = MODE_BOUND,
    target_variance: float = TARGET_VARIANCE,
    seed: int = 0,
    plot_bound: float = 56.0,
    grid_resolution: int = 360,
    contour_levels: int = 80,
    reward_plot_range: float = 100.0,
) -> None:
    """Configure the notebook's target and rebuild its plotting grid.

    Call once in setup, before importing grid arrays or creating samplers.
    Rerun the notebook from the top after changing its hyperparameter panel.
    """
    global NUM_MODES, MODE_BOUND, TARGET_VARIANCE, TARGET_STD, LOG_MIXTURE_WEIGHT
    global MODE_LOCATIONS, PLOT_BOUND, PLOT_AXIS, GRID_X, GRID_Y, GRID_POINTS
    global REWARD_ON_GRID, DISPLAY_REWARD, REWARD_LEVELS
    if num_modes < 1 or mode_bound <= 0 or target_variance <= 0:
        raise ValueError("The target needs positive mode count, bound, and variance.")
    if plot_bound <= 0 or grid_resolution < 2 or contour_levels < 2 or reward_plot_range <= 0:
        raise ValueError("Invalid plotting grid or contour settings.")
    NUM_MODES, MODE_BOUND, TARGET_VARIANCE = num_modes, mode_bound, target_variance
    TARGET_STD = math.sqrt(target_variance)
    LOG_MIXTURE_WEIGHT = -math.log(num_modes)
    generator = torch.Generator().manual_seed(seed)
    MODE_LOCATIONS = 2 * mode_bound * torch.rand(
        num_modes, STATE_DIMENSION, generator=generator
    ) - mode_bound
    PLOT_BOUND = plot_bound
    PLOT_AXIS = torch.linspace(-plot_bound, plot_bound, grid_resolution)
    GRID_X, GRID_Y = torch.meshgrid(PLOT_AXIS, PLOT_AXIS, indexing="xy")
    GRID_POINTS = torch.stack((GRID_X, GRID_Y), dim=-1)
    REWARD_ON_GRID = target_reward(GRID_POINTS)
    DISPLAY_REWARD = REWARD_ON_GRID.clamp(min=REWARD_ON_GRID.max() - reward_plot_range)
    REWARD_LEVELS = torch.linspace(
        DISPLAY_REWARD.min().item(), DISPLAY_REWARD.max().item(), contour_levels
    )


configure_target()


def plot_reward_landscape(axis: plt.Axes) -> None:
    """Draw the GMM-40 contours without markers at the component centers."""
    axis.contour(
        GRID_X,
        GRID_Y,
        DISPLAY_REWARD,
        levels=REWARD_LEVELS,
        cmap="viridis",
        linewidths=0.55,
        alpha=0.9,
    )
    axis.set(
        xlim=(-PLOT_BOUND, PLOT_BOUND),
        ylim=(-PLOT_BOUND, PLOT_BOUND),
        xticks=(-MODE_BOUND, 0, MODE_BOUND),
        yticks=(-MODE_BOUND, 0, MODE_BOUND),
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        aspect="equal",
    )
    axis.set_facecolor("white")


class VPSchedule(nn.Module):
    """K+1 coefficients beta_0, ..., beta_K for the slides' K VP steps."""

    def __init__(
        self,
        num_steps: int = NUM_DIFFUSION_STEPS,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
        learn_schedule: bool = False,
        step_size: float = DIFFUSION_STEP_SIZE,
    ):
        super().__init__()
        if num_steps < 2:
            raise ValueError("A diffusion schedule needs at least two steps.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1.")
        if not math.isfinite(step_size) or not 0 < beta_end * step_size < 2:
            raise ValueError("Require 0 < beta_end * step_size < 2 for VP shrinkage.")

        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learn_schedule = learn_schedule
        self.step_size = step_size

        if learn_schedule:
            # K positive increments give K+1 ordered coefficients, with
            # beta_0 and beta_K fixed and beta_1, ..., beta_{K-1} learnable.
            self.increment_logits = nn.Parameter(torch.zeros(num_steps))
        else:
            self.register_buffer(
                "fixed_betas", torch.linspace(beta_start, beta_end, num_steps + 1)
            )

    def forward(self) -> torch.Tensor:
        if not self.learn_schedule:
            return self.fixed_betas

        increments = self.increment_logits.softmax(dim=0)
        interior = torch.cumsum(increments, dim=0)[:-1]
        interior_betas = self.beta_start + (
            self.beta_end - self.beta_start
        ) * interior
        endpoints = self.increment_logits.new_tensor(
            [self.beta_start, self.beta_end]
        )
        return torch.cat((endpoints[:1], interior_betas, endpoints[1:]))

    def cumulative_signal(self) -> torch.Tensor:
        """Squared forward signal multiplier through steps k=1, ..., K."""
        forward_deltas = self()[:-1] * self.step_size
        return torch.cumprod((1.0 - 0.5 * forward_deltas).square(), dim=0)


class ScoreNetwork(nn.Module):
    """Time-conditioned score with optional DDS/PIS-GRAD preconditioning."""

    def __init__(
        self, width: int = 96, *, use_langevin_preconditioning: bool = False,
        prior_std: float = PRIOR_STD,
        langevin_gradient_clip: float = LANGEVIN_GRADIENT_CLIP,
        langevin_drift_clip: float = LANGEVIN_DRIFT_CLIP,
    ):
        super().__init__()
        self.use_langevin_preconditioning = use_langevin_preconditioning
        self.prior_variance = prior_std**2
        self.langevin_gradient_clip = langevin_gradient_clip
        self.langevin_drift_clip = langevin_drift_clip
        self.target_temperature = TEMPERATURE
        self.residual = nn.Sequential(
            nn.Linear(7, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, STATE_DIMENSION),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

        if use_langevin_preconditioning:
            self.langevin_gate = nn.Sequential(
                nn.Linear(5, width // 2),
                nn.SiLU(),
                nn.Linear(width // 2, STATE_DIMENSION),
            )
            nn.init.zeros_(self.langevin_gate[-1].weight)
            nn.init.zeros_(self.langevin_gate[-1].bias)
        else:
            self.langevin_gate = None

    def forward(
        self, states: torch.Tensor, step_index: int, num_steps: int
    ) -> torch.Tensor:
        # step_index is the mathematical k in {1, ..., K}.
        time = states.new_full((*states.shape[:-1], 1), step_index / num_steps)
        time_features = torch.cat(
            (
                time,
                torch.sin(2.0 * math.pi * time),
                torch.cos(2.0 * math.pi * time),
                torch.sin(4.0 * math.pi * time),
                torch.cos(4.0 * math.pi * time),
            ),
            dim=-1,
        )
        features = torch.cat(
            (
                states,
                time_features,
            ),
            dim=-1,
        )
        drift_correction = self.residual(features)
        if self.langevin_gate is not None:
            target_gradient = target_log_density_gradient(
                states, self.target_temperature
            ).clamp(-self.langevin_gradient_clip, self.langevin_gradient_clip)
            drift_correction = drift_correction + self.langevin_gate(
                time_features
            ) * target_gradient
            drift_correction = drift_correction.clamp(
                -self.langevin_drift_clip, self.langevin_drift_clip
            )

        # The Gaussian prior score initializes a stable reverse drift.
        # Euler steps preserve its variance only approximately.
        return -states / self.prior_variance + drift_correction


ForwardStep = Callable[..., torch.Tensor]
ReverseStep = Callable[..., torch.Tensor]
KernelLogProb = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class DiffusionKernels:
    forward_sde_step: ForwardStep
    reverse_sde_step: ReverseStep
    forward_kernel_log_prob: KernelLogProb
    reverse_kernel_log_prob: KernelLogProb


@dataclass
class ReversePath:
    states: torch.Tensor
    log_q: torch.Tensor
    log_forward: torch.Tensor

    @property
    def samples(self) -> torch.Tensor:
        return self.states[0]


def prior_log_prob(states: torch.Tensor, prior_std: float = PRIOR_STD) -> torch.Tensor:
    prior_variance = prior_std**2
    return -0.5 * (
        states.square() / prior_variance
        + math.log(2.0 * math.pi * prior_variance)
    ).sum(dim=-1)


class DiffusionSampler(nn.Module):
    """A learned reverse VP chain starting from the broad GMM-40 prior."""

    def __init__(
        self,
        kernels: DiffusionKernels,
        *,
        learn_schedule: bool = False,
        use_langevin_preconditioning: bool = False,
        num_steps: int = NUM_DIFFUSION_STEPS,
        prior_std: float = PRIOR_STD,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
        diffusion_step_size: float = DIFFUSION_STEP_SIZE,
        score_width: int = 96,
        langevin_gradient_clip: float = LANGEVIN_GRADIENT_CLIP,
        langevin_drift_clip: float = LANGEVIN_DRIFT_CLIP,
    ):
        super().__init__()
        if not math.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("The fixed Gaussian prior needs a positive, finite std.")
        self.prior_std = prior_std
        self.kernels = kernels
        self.score_network = ScoreNetwork(
            width=score_width,
            use_langevin_preconditioning=use_langevin_preconditioning,
            prior_std=prior_std,
            langevin_gradient_clip=langevin_gradient_clip,
            langevin_drift_clip=langevin_drift_clip,
        )
        self.temperature = TEMPERATURE
        self.schedule = VPSchedule(
            num_steps=num_steps, learn_schedule=learn_schedule,
            beta_start=beta_start, beta_end=beta_end, step_size=diffusion_step_size,
        )

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature = float(value)
        if hasattr(self, "score_network"):
            self.score_network.target_temperature = self._temperature

    @property
    def num_steps(self) -> int:
        return self.schedule.num_steps

    @property
    def device(self) -> torch.device:
        return next(self.score_network.parameters()).device

    def sample_reverse_path(
        self,
        num_samples: int,
        *,
        reparameterize: bool,
        prior_noise: torch.Tensor | None = None,
        step_noises: torch.Tensor | None = None,
    ) -> ReversePath:
        deltas = self.schedule() * self.schedule.step_size
        if prior_noise is None:
            state = self.prior_std * torch.randn(
                num_samples, STATE_DIMENSION, device=self.device
            )
        else:
            state = self.prior_std * prior_noise.to(self.device)
            if state.shape != (num_samples, STATE_DIMENSION):
                raise ValueError("prior_noise has the wrong shape.")

        if step_noises is not None:
            expected_shape = (self.num_steps, num_samples, STATE_DIMENSION)
            if step_noises.shape != expected_shape:
                raise ValueError(f"step_noises must have shape {expected_shape}.")
            step_noises = step_noises.to(self.device)

        states: list[torch.Tensor | None] = [None] * (self.num_steps + 1)
        states[self.num_steps] = state
        log_q = prior_log_prob(state, self.prior_std)
        log_forward = state.new_zeros(num_samples)

        for step_index in range(self.num_steps, 0, -1):
            reverse_delta = deltas[step_index]
            forward_delta = deltas[step_index - 1]
            noise = None if step_noises is None else step_noises[step_index - 1]
            previous_state = self.kernels.reverse_sde_step(
                self.score_network,
                state,
                step_index,
                reverse_delta,
                self.num_steps,
                reparameterize=reparameterize,
                noise=noise,
                prior_std=self.prior_std,
            )
            log_q = log_q + self.kernels.reverse_kernel_log_prob(
                self.score_network,
                previous_state,
                state,
                step_index,
                reverse_delta,
                self.num_steps,
                prior_std=self.prior_std,
            )
            log_forward = log_forward + self.kernels.forward_kernel_log_prob(
                previous_state, state, forward_delta, prior_std=self.prior_std
            )
            states[step_index - 1] = previous_state
            state = previous_state

        return ReversePath(
            states=torch.stack([value for value in states if value is not None]),
            log_q=log_q,
            log_forward=log_forward,
        )

    def path_cost(self, path: ReversePath) -> torch.Tensor:
        """Temperature-scaled joint KL up to the constant T log Z_T."""
        return -target_reward(path.samples) + self.temperature * (
            path.log_q - path.log_forward
        )


def reparameterization_path_loss(
    sampler: DiffusionSampler, num_samples: int
) -> torch.Tensor:
    """Pathwise gradient of the diffusion path-space upper bound."""
    path = sampler.sample_reverse_path(num_samples, reparameterize=True)
    return sampler.path_cost(path).mean()


def log_derivative_path_loss(
    sampler: DiffusionSampler, num_samples: int
) -> torch.Tensor:
    """Score-function gradient with a leave-one-out path-cost baseline."""
    if num_samples < 2:
        raise ValueError("The leave-one-out baseline needs at least two paths.")

    path = sampler.sample_reverse_path(num_samples, reparameterize=False)
    path_cost = sampler.path_cost(path)
    baseline = (path_cost.sum() - path_cost) / (num_samples - 1)
    centered_cost = (path_cost - baseline).detach()

    # The second term supplies explicit derivatives of kernel densities. This
    # matters when the VP schedule itself is learnable.
    return (centered_cost * path.log_q + path_cost).mean()


def check_kernel_functions(
    kernels: DiffusionKernels, *, prior_std: float = PRIOR_STD,
    num_steps: int = NUM_DIFFUSION_STEPS,
) -> None:
    """Structural and gradient checks for the four student functions."""
    torch.manual_seed(3)
    score_network = ScoreNetwork(width=32, prior_std=prior_std)
    delta = torch.tensor(0.08, requires_grad=True)
    clean_state = torch.randn(16, STATE_DIMENSION)

    noisy_state = kernels.forward_sde_step(
        clean_state, delta, reparameterize=True, prior_std=prior_std
    )
    recovered_state = kernels.reverse_sde_step(
        score_network,
        noisy_state,
        min(4, num_steps),
        delta,
        num_steps,
        reparameterize=True,
        prior_std=prior_std,
    )
    forward_log_prob = kernels.forward_kernel_log_prob(
        clean_state, noisy_state, delta, prior_std=prior_std
    )
    reverse_log_prob = kernels.reverse_kernel_log_prob(
        score_network,
        recovered_state,
        noisy_state,
        min(4, num_steps),
        delta,
        num_steps,
        prior_std=prior_std,
    )

    assert noisy_state.shape == clean_state.shape
    assert recovered_state.shape == clean_state.shape
    assert forward_log_prob.shape == (clean_state.shape[0],)
    assert reverse_log_prob.shape == (clean_state.shape[0],)
    assert all(
        torch.isfinite(value).all()
        for value in (noisy_state, recovered_state, forward_log_prob, reverse_log_prob)
    )

    check_loss = -(forward_log_prob + reverse_log_prob).mean()
    check_loss.backward()
    gradients = [parameter.grad for parameter in score_network.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert delta.grad is not None and torch.isfinite(delta.grad)
    print("All four VP-kernel checks passed.")


def nearest_mode_fractions(samples: torch.Tensor) -> torch.Tensor:
    locations = MODE_LOCATIONS.to(samples)
    assignments = torch.cdist(samples, locations).argmin(dim=1)
    return torch.stack(
        [(assignments == index).float().mean() for index in range(len(locations))]
    )


def evaluate_mode_recovery(
    sampler: DiffusionSampler,
    *,
    num_samples: int = 10_000,
    capture_radius: float = 1.25,
    minimum_fraction: float = 0.005,
    seed: int = 31,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure mass captured near each mode and flag recovered modes."""
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(seed)
        samples = sampler.sample_reverse_path(
            num_samples, reparameterize=True
        ).samples.cpu()
    distances = torch.cdist(samples, MODE_LOCATIONS)
    nearest_distances, assignments = distances.min(dim=1)
    captured_fractions = torch.stack(
        [
            ((assignments == index) & (nearest_distances <= capture_radius))
            .float()
            .mean()
            for index in range(len(MODE_LOCATIONS))
        ]
    )
    return captured_fractions, captured_fractions >= minimum_fraction


def train_sampler(
    kernels: DiffusionKernels,
    *,
    gradient_estimator: str = "reparameterization",
    learn_schedule: bool = False,
    use_langevin_preconditioning: bool = False,
    seed: int = 11,
    steps: int = 1000,
    batch_size: int = 256,
    learning_rate: float = 5.0e-4,
    schedule_learning_rate: float = 3.0e-4,
    log_every: int = 25,
    num_animation_samples: int = 600,
    temperature: float = TEMPERATURE,
    max_grad_norm: float = 10.0,
    animation_seed: int | None = None,
    num_steps: int = NUM_DIFFUSION_STEPS,
    prior_std: float = PRIOR_STD,
    beta_start: float = BETA_START,
    beta_end: float = BETA_END,
    diffusion_step_size: float = DIFFUSION_STEP_SIZE,
    score_width: int = 96,
    langevin_gradient_clip: float = LANGEVIN_GRADIENT_CLIP,
    langevin_drift_clip: float = LANGEVIN_DRIFT_CLIP,
) -> tuple[DiffusionSampler, dict[str, list]]:
    """Train at a fixed target temperature and retain fixed-noise GIF samples."""
    if gradient_estimator not in {"reparameterization", "log_derivative"}:
        raise ValueError("Unknown gradient estimator.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Diffusion sampling requires a positive, finite temperature.")

    torch.manual_seed(seed)
    sampler = DiffusionSampler(
        kernels,
        learn_schedule=learn_schedule,
        use_langevin_preconditioning=use_langevin_preconditioning,
        num_steps=num_steps, prior_std=prior_std,
        beta_start=beta_start, beta_end=beta_end,
        diffusion_step_size=diffusion_step_size, score_width=score_width,
        langevin_gradient_clip=langevin_gradient_clip,
        langevin_drift_clip=langevin_drift_clip,
    )
    sampler.temperature = temperature
    parameter_groups = [
        {"params": sampler.score_network.parameters(), "lr": learning_rate}
    ]
    if learn_schedule:
        parameter_groups.append(
            {
                "params": sampler.schedule.parameters(),
                "lr": schedule_learning_rate,
            }
        )
    optimizer = torch.optim.Adam(parameter_groups)
    loss_function = (
        reparameterization_path_loss
        if gradient_estimator == "reparameterization"
        else log_derivative_path_loss
    )

    animation_generator = torch.Generator().manual_seed(
        seed + 10_000 if animation_seed is None else animation_seed
    )
    animation_prior = torch.randn(
        num_animation_samples, STATE_DIMENSION, generator=animation_generator
    )
    animation_noises = torch.randn(
        sampler.num_steps,
        len(animation_prior),
        STATE_DIMENSION,
        generator=animation_generator,
    )
    history: dict[str, list] = {
        "step": [],
        "samples": [],
        "mean_reward": [],
        "mode_fractions": [],
        "betas": [],
        "loss": [],
        "evaluation_loss": [],
        "temperature": [],
    }
    latest_loss = float("nan")

    for step in range(steps + 1):
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                fixed_path = sampler.sample_reverse_path(
                    len(animation_prior),
                    reparameterize=True,
                    prior_noise=animation_prior,
                    step_noises=animation_noises,
                )
                samples = fixed_path.samples.cpu()
                history["step"].append(step)
                history["samples"].append(samples)
                history["mean_reward"].append(target_reward(samples).mean().item())
                history["mode_fractions"].append(nearest_mode_fractions(samples))
                history["betas"].append(sampler.schedule().detach().cpu())
                history["loss"].append(latest_loss)
                history["evaluation_loss"].append(
                    sampler.path_cost(fixed_path).mean().item()
                )
                history["temperature"].append(sampler.temperature)

        if step == steps:
            break

        optimizer.zero_grad()
        loss = loss_function(sampler, batch_size)
        loss.backward()
        nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        latest_loss = loss.item()

    return sampler, history


def plot_training_summary(
    sampler: DiffusionSampler, history: dict[str, list]
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    plot_reward_landscape(axes[0])
    samples = history["samples"][-1]
    axes[0].scatter(
        samples[:, 0],
        samples[:, 1],
        s=16,
        color="#5E2B97",
        alpha=0.62,
        zorder=4,
        label="learned samples",
    )
    axes[0].legend(loc="lower center", fontsize=8)
    axes[0].set_title("Final diffusion samples")

    axes[1].plot(history["step"], history["mean_reward"], color="#5E2B97")
    axes[1].set(
        title="Reward during training",
        xlabel="optimization step",
        ylabel="mean reward",
    )
    axes[1].grid(alpha=0.2)

    initial_betas = history["betas"][0]
    final_betas = history["betas"][-1]
    diffusion_steps = torch.arange(len(final_betas))
    axes[2].plot(diffusion_steps, initial_betas, "--", label="initial")
    axes[2].plot(diffusion_steps, final_betas, label="final")
    axes[2].set(
        title=(
            "Learned VP schedule"
            if sampler.schedule.learn_schedule
            else "Fixed VP schedule"
        ),
        xlabel="diffusion step k",
        ylabel="discrete beta_k",
    )
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    return fig


def set_loss_axis_scale(axis, loss_histories) -> None:
    """Use log loss axes, retaining zero/negative objectives with a symlog scale."""
    if all(value > 0 for values in loss_histories for value in values):
        axis.set_yscale("log")
        axis.set_ylabel(axis.get_ylabel() + " (log scale)")
    else:
        # A path objective can be negative; preserve it without shifting/clipping.
        axis.set_yscale("symlog", linthresh=1.0)
        axis.set_ylabel(axis.get_ylabel() + " (symlog scale)")


def save_training_animation(
    history: dict[str, list],
    output_path: Path,
    *,
    gradient_estimator: str,
    learn_schedule: bool,
    use_langevin_preconditioning: bool,
) -> animation.FuncAnimation:
    """Save samples beside their synchronized fixed-noise path loss."""
    fig, (axis, loss_axis) = plt.subplots(
        1, 2, figsize=(10.8, 5.2), gridspec_kw={"width_ratios": [1.0, 0.9]}
    )
    plot_reward_landscape(axis)
    initial_samples = history["samples"][0]
    sample_artist = axis.scatter(
        initial_samples[:, 0],
        initial_samples[:, 1],
        s=19,
        color="#5E2B97",
        edgecolor="white",
        linewidth=0.25,
        alpha=0.68,
        zorder=4,
        label="diffusion samples",
    )
    axis.legend(loc="lower center", fontsize=8)
    title = axis.set_title("")

    loss_values = history["evaluation_loss"]
    loss_axis.plot(history["step"], loss_values, color="#0067B1", linewidth=2)
    loss_marker = loss_axis.scatter(
        [history["step"][0]], [loss_values[0]], s=34, color="#0067B1", zorder=3
    )
    current_step_line = loss_axis.axvline(
        history["step"][0], color="#1F2937", linestyle="--", linewidth=1.4
    )
    loss_axis.set(
        title="Fixed-noise path loss",
        xlabel="optimization step",
        ylabel="path-space objective",
        xlim=(history["step"][0], history["step"][-1]),
    )
    set_loss_axis_scale(loss_axis, [loss_values])
    loss_axis.grid(alpha=0.2, which="both")

    def update(frame: int):
        sample_artist.set_offsets(history["samples"][frame])
        current_step = history["step"][frame]
        current_step_line.set_xdata([current_step, current_step])
        loss_marker.set_offsets([[current_step, loss_values[frame]]])
        schedule_label = "learned schedule" if learn_schedule else "fixed schedule"
        preconditioner_label = (
            "Langevin preconditioned"
            if use_langevin_preconditioning
            else "unpreconditioned"
        )
        title.set_text(
            "Lesson 2 diffusion sampler\n"
            f"{gradient_estimator}, {schedule_label}, {preconditioner_label}\n"
            f"step {history['step'][frame]}, T={history['temperature'][frame]:.2f}"
        )
        return sample_artist, title, current_step_line, loss_marker

    sample_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(history["step"]),
        interval=900,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_animation.save(
        output_path, writer=animation.PillowWriter(fps=1), dpi=105
    )
    plt.close(fig)
    return sample_animation


def save_training_comparison_animation(
    experiments: list[tuple[str, bool, dict[str, list]]],
    output_path: Path,
    *, fps: float = 1.5, dpi: int = 92,
    frame_indices: list[int] | None = None,
) -> animation.FuncAnimation:
    """Compare estimators in columns and Langevin off/on in rows.

    Optional frame indices shorten long animations while retaining full loss curves.
    """
    if len(experiments) != 4:
        raise ValueError("The comparison requires exactly four experiments.")
    by_setting = {(estimator, langevin): history
                  for estimator, langevin, history in experiments}
    order = [(estimator, langevin) for langevin in (False, True)
             for estimator in ("reparameterization", "log_derivative")]
    if set(by_setting) != set(order):
        raise ValueError("Provide each estimator with Langevin off and on exactly once.")
    experiments = [(estimator, langevin, by_setting[(estimator, langevin)])
                   for estimator, langevin in order]

    frame_counts = {len(history["step"]) for _, _, history in experiments}
    step_grids = {tuple(history["step"]) for _, _, history in experiments}
    if len(frame_counts) != 1 or len(step_grids) != 1:
        raise ValueError("All experiments must use the same logged training steps.")

    fig = plt.figure(figsize=(13.2, 8.2))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1.0, 1.0, 0.92), wspace=0.18, hspace=0.22
    )
    sample_axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]
    loss_axes = [fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])]
    sample_artists = []
    titles = []
    for axis, (gradient_estimator, use_langevin, history) in zip(
        sample_axes, experiments
    ):
        plot_reward_landscape(axis)
        samples = history["samples"][0]
        sample_artists.append(
            axis.scatter(
                samples[:, 0],
                samples[:, 1],
                s=10,
                color="#0067B1" if gradient_estimator == "reparameterization" else "#D76A12",
                alpha=0.58,
                zorder=4,
            )
        )
        estimator_label = (
            "Reparameterization"
            if gradient_estimator == "reparameterization"
            else "Log derivative"
        )
        langevin_label = "+ Langevin" if use_langevin else "without Langevin"
        titles.append(axis.set_title(f"{estimator_label} · {langevin_label}"))

    loss_markers = []
    current_step_lines = []
    for row, loss_axis in enumerate(loss_axes):
        row_experiments = experiments[2 * row : 2 * row + 2]
        for (gradient_estimator, _, history), color in zip(
            row_experiments, ("#0067B1", "#D76A12")
        ):
            label = ("Reparameterization" if gradient_estimator == "reparameterization"
                     else "Log derivative")
            values = history["evaluation_loss"]
            loss_axis.plot(
                history["step"], values, color=color, linewidth=1.9, label=label
            )
            loss_markers.append(
                loss_axis.scatter(
                    [history["step"][0]],
                    [values[0]],
                    s=30,
                    color=color,
                    zorder=3,
                )
            )
        current_step_lines.append(
            loss_axis.axvline(
                row_experiments[0][2]["step"][0],
                color="#1F2937",
                linestyle="--",
                linewidth=1.3,
            )
        )
        preconditioner_label = "without Langevin" if row == 0 else "+ Langevin"
        loss_axis.set(
            title=f"Path loss · {preconditioner_label}",
            xlabel="optimization step",
            ylabel="fixed-noise path objective",
            xlim=(
                row_experiments[0][2]["step"][0],
                row_experiments[0][2]["step"][-1],
            ),
        )
        set_loss_axis_scale(
            loss_axis, [history["evaluation_loss"] for _, _, history in row_experiments]
        )
        if row_experiments[0][2]["step"][-1] >= 10_000:
            from matplotlib.ticker import FuncFormatter, MultipleLocator
            loss_axis.xaxis.set_major_locator(MultipleLocator(5000))
            loss_axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0")
            )
        loss_axis.grid(alpha=0.2, which="both")
        loss_axis.legend(fontsize=8)

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.90)
    step_label = fig.suptitle("")

    def update(frame: int):
        for (_, _, history), sample_artist in zip(experiments, sample_artists):
            sample_artist.set_offsets(history["samples"][frame])
        reference_history = experiments[0][2]
        current_step = reference_history["step"][frame]
        for current_step_line in current_step_lines:
            current_step_line.set_xdata([current_step, current_step])
        for (_, _, history), loss_marker in zip(experiments, loss_markers):
            loss_marker.set_offsets(
                [[current_step, history["evaluation_loss"][frame]]]
            )
        step_label.set_text(
            f"GMM-{NUM_MODES} diffusion sampler training  |  "
            f"step {current_step}  |  "
            f"T={reference_history['temperature'][frame]:.2f}"
        )
        return [
            *sample_artists,
            *titles,
            *loss_markers,
            *current_step_lines,
            step_label,
        ]

    comparison_animation = animation.FuncAnimation(
        fig,
        update,
        frames=frame_counts.pop() if frame_indices is None else frame_indices,
        interval=1000 / fps,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_animation.save(
        output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi
    )
    plt.close(fig)
    return comparison_animation
