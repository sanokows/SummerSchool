"""Shared infrastructure for Lesson 4: diffusion samplers.

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
BETA_START = 0.02
BETA_END = 0.18
LANGEVIN_GRADIENT_CLIP = 1.0e2
LANGEVIN_DRIFT_CLIP = 1.0e4

MODE_LOCATIONS = torch.tensor(
    [[-3.0, -2.0], [-2.0, 2.5], [2.2, 2.7], [3.0, -1.8]]
)
MODE_WEIGHTS = torch.tensor([0.46, 0.29, 0.17, 0.08])
TARGET_STD = 0.5
GLOBAL_MODE_INDEX = int(MODE_WEIGHTS.argmax())
GLOBAL_MAX_LOCATION = MODE_LOCATIONS[GLOBAL_MODE_INDEX]


def target_reward(states: torch.Tensor) -> torch.Tensor:
    """Log-density reward of the weighted two-dimensional Gaussian mixture."""
    locations = MODE_LOCATIONS.to(device=states.device, dtype=states.dtype)
    weights = MODE_WEIGHTS.to(device=states.device, dtype=states.dtype)
    standardized = (states.unsqueeze(-2) - locations) / TARGET_STD
    component_log_prob = (
        -0.5 * standardized.square().sum(dim=-1)
        - STATE_DIMENSION * math.log(TARGET_STD * math.sqrt(2.0 * math.pi))
        + weights.log()
    )
    return torch.logsumexp(component_log_prob, dim=-1)


def target_log_density_gradient(
    states: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Analytic, detached gradient of log pi_T(x) = R(x) / temperature."""
    locations = MODE_LOCATIONS.to(device=states.device, dtype=states.dtype)
    weights = MODE_WEIGHTS.to(device=states.device, dtype=states.dtype)
    differences = locations - states.unsqueeze(-2)
    component_logits = (
        -0.5 * differences.square().sum(dim=-1) / TARGET_STD**2
        + weights.log()
    )
    responsibilities = component_logits.softmax(dim=-1)
    gradient = (
        responsibilities.unsqueeze(-1) * differences / TARGET_STD**2
    ).sum(dim=-2)
    return (gradient / temperature).detach()


PLOT_AXIS = torch.linspace(-5.5, 5.5, 250)
GRID_X, GRID_Y = torch.meshgrid(PLOT_AXIS, PLOT_AXIS, indexing="xy")
GRID_POINTS = torch.stack((GRID_X, GRID_Y), dim=-1)
REWARD_ON_GRID = target_reward(GRID_POINTS)
DISPLAY_REWARD = REWARD_ON_GRID.clamp(min=REWARD_ON_GRID.max() - 20.0)
REWARD_LEVELS = torch.linspace(
    DISPLAY_REWARD.min().item(), DISPLAY_REWARD.max().item(), 32
)


def plot_reward_landscape(axis: plt.Axes) -> None:
    """Draw the shared yellow-to-green reward landscape and target modes."""
    axis.contourf(
        GRID_X, GRID_Y, DISPLAY_REWARD, levels=REWARD_LEVELS, cmap="YlGn"
    )
    axis.contour(
        GRID_X,
        GRID_Y,
        DISPLAY_REWARD,
        levels=REWARD_LEVELS[3::4],
        colors="white",
        linewidths=0.7,
        alpha=0.85,
    )
    axis.scatter(
        MODE_LOCATIONS[:, 0],
        MODE_LOCATIONS[:, 1],
        s=700 * MODE_WEIGHTS,
        c=MODE_WEIGHTS,
        cmap="viridis",
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    for location, weight in zip(MODE_LOCATIONS, MODE_WEIGHTS):
        axis.annotate(
            f"target mass {100.0 * weight.item():.0f}%",
            location + 0.18,
            color="white",
            fontsize=8,
            bbox={
                "facecolor": "black",
                "alpha": 0.66,
                "edgecolor": "none",
                "pad": 1.4,
            },
        )
    axis.scatter(
        *GLOBAL_MAX_LOCATION,
        marker="*",
        s=280,
        color="gold",
        edgecolor="black",
        linewidth=1.2,
        zorder=5,
        label="global reward maximum",
    )
    axis.set(
        xlim=(-5.2, 5.2),
        ylim=(-5.2, 5.2),
        xlabel="state x₁",
        ylabel="state x₂",
        aspect="equal",
    )


class VPSchedule(nn.Module):
    """Monotone VP noise schedule with fixed, well-conditioned endpoints."""

    def __init__(
        self,
        num_steps: int = NUM_DIFFUSION_STEPS,
        beta_start: float = BETA_START,
        beta_end: float = BETA_END,
        learn_schedule: bool = False,
    ):
        super().__init__()
        if num_steps < 2:
            raise ValueError("A diffusion schedule needs at least two steps.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("Require 0 < beta_start < beta_end < 1.")

        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learn_schedule = learn_schedule

        if learn_schedule:
            # Softmax increments keep beta_1 < ... < beta_K while preserving
            # the endpoints and therefore the total amount of noising.
            self.increment_logits = nn.Parameter(torch.zeros(num_steps - 1))
        else:
            self.register_buffer(
                "fixed_betas", torch.linspace(beta_start, beta_end, num_steps)
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
        """Return alpha-bar_k = product_{j<=k}(1-beta_j)."""
        return torch.cumprod(1.0 - self(), dim=0)


class ScoreNetwork(nn.Module):
    """Time-conditioned score with optional DDS/PIS-GRAD preconditioning."""

    def __init__(
        self, width: int = 96, *, use_langevin_preconditioning: bool = False
    ):
        super().__init__()
        self.use_langevin_preconditioning = use_langevin_preconditioning
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
        time = states.new_full((*states.shape[:-1], 1), (step_index + 1) / num_steps)
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
            ).clamp(-LANGEVIN_GRADIENT_CLIP, LANGEVIN_GRADIENT_CLIP)
            drift_correction = drift_correction + self.langevin_gate(
                time_features
            ) * target_gradient
            drift_correction = drift_correction.clamp(
                -LANGEVIN_DRIFT_CLIP, LANGEVIN_DRIFT_CLIP
            )

        # score=-x leaves a standard normal invariant under the VP kernel.
        return -states + drift_correction


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


def standard_normal_log_prob(states: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        states.square() + math.log(2.0 * math.pi)
    ).sum(dim=-1)


class DiffusionSampler(nn.Module):
    """A learned reverse VP chain starting from a standard-normal prior."""

    def __init__(
        self,
        kernels: DiffusionKernels,
        *,
        learn_schedule: bool = False,
        use_langevin_preconditioning: bool = False,
        num_steps: int = NUM_DIFFUSION_STEPS,
    ):
        super().__init__()
        self.kernels = kernels
        self.score_network = ScoreNetwork(
            use_langevin_preconditioning=use_langevin_preconditioning
        )
        self.temperature = TEMPERATURE
        self.schedule = VPSchedule(
            num_steps=num_steps, learn_schedule=learn_schedule
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
        betas = self.schedule()
        if prior_noise is None:
            state = torch.randn(num_samples, STATE_DIMENSION, device=self.device)
        else:
            state = prior_noise.to(self.device)
            if state.shape != (num_samples, STATE_DIMENSION):
                raise ValueError("prior_noise has the wrong shape.")

        if step_noises is not None:
            expected_shape = (self.num_steps, num_samples, STATE_DIMENSION)
            if step_noises.shape != expected_shape:
                raise ValueError(f"step_noises must have shape {expected_shape}.")
            step_noises = step_noises.to(self.device)

        states: list[torch.Tensor | None] = [None] * (self.num_steps + 1)
        states[self.num_steps] = state
        log_q = standard_normal_log_prob(state)
        log_forward = state.new_zeros(num_samples)

        for step_index in reversed(range(self.num_steps)):
            beta = betas[step_index]
            noise = None if step_noises is None else step_noises[step_index]
            previous_state = self.kernels.reverse_sde_step(
                self.score_network,
                state,
                step_index,
                beta,
                self.num_steps,
                reparameterize=reparameterize,
                noise=noise,
            )
            log_q = log_q + self.kernels.reverse_kernel_log_prob(
                self.score_network,
                previous_state,
                state,
                step_index,
                beta,
                self.num_steps,
            )
            log_forward = log_forward + self.kernels.forward_kernel_log_prob(
                previous_state, state, beta
            )
            states[step_index] = previous_state
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


def check_kernel_functions(kernels: DiffusionKernels) -> None:
    """Structural and gradient checks for the four student functions."""
    torch.manual_seed(3)
    score_network = ScoreNetwork(width=32)
    beta = torch.tensor(0.08, requires_grad=True)
    clean_state = torch.randn(16, STATE_DIMENSION)

    noisy_state = kernels.forward_sde_step(
        clean_state, beta, reparameterize=True
    )
    recovered_state = kernels.reverse_sde_step(
        score_network,
        noisy_state,
        4,
        beta,
        NUM_DIFFUSION_STEPS,
        reparameterize=True,
    )
    forward_log_prob = kernels.forward_kernel_log_prob(
        clean_state, noisy_state, beta
    )
    reverse_log_prob = kernels.reverse_kernel_log_prob(
        score_network,
        recovered_state,
        noisy_state,
        4,
        beta,
        NUM_DIFFUSION_STEPS,
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
    assert beta.grad is not None and torch.isfinite(beta.grad)
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
    learning_rate: float = 2.0e-3,
    schedule_learning_rate: float = 3.0e-4,
    log_every: int = 25,
    initial_temperature: float = 2.0,
    temperature_anneal_steps: int = 700,
    num_animation_samples: int = 600,
) -> tuple[DiffusionSampler, dict[str, list]]:
    """Train a sampler and retain fixed-noise samples for a smooth GIF."""
    if gradient_estimator not in {"reparameterization", "log_derivative"}:
        raise ValueError("Unknown gradient estimator.")

    torch.manual_seed(seed)
    sampler = DiffusionSampler(
        kernels,
        learn_schedule=learn_schedule,
        use_langevin_preconditioning=use_langevin_preconditioning,
    )
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

    animation_generator = torch.Generator().manual_seed(seed + 10_000)
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
        "temperature": [],
    }
    latest_loss = float("nan")

    for step in range(steps + 1):
        annealing_progress = min(step / max(temperature_anneal_steps, 1), 1.0)
        sampler.temperature = initial_temperature + annealing_progress * (
            TEMPERATURE - initial_temperature
        )
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
                history["temperature"].append(sampler.temperature)

        if step == steps:
            break

        optimizer.zero_grad()
        loss = loss_function(sampler, batch_size)
        loss.backward()
        nn.utils.clip_grad_norm_(sampler.parameters(), max_norm=10.0)
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
    diffusion_steps = torch.arange(1, len(final_betas) + 1)
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


def save_training_animation(
    history: dict[str, list],
    output_path: Path,
    *,
    gradient_estimator: str,
    learn_schedule: bool,
    use_langevin_preconditioning: bool,
) -> animation.FuncAnimation:
    """Save the fixed-noise sample cloud over optimization iterations."""
    fig, axis = plt.subplots(figsize=(6.4, 5.8))
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

    def update(frame: int):
        sample_artist.set_offsets(history["samples"][frame])
        schedule_label = "learned schedule" if learn_schedule else "fixed schedule"
        preconditioner_label = (
            "Langevin preconditioned"
            if use_langevin_preconditioning
            else "unpreconditioned"
        )
        title.set_text(
            "Lesson 4 diffusion sampler\n"
            f"{gradient_estimator}, {schedule_label}, {preconditioner_label}\n"
            f"step {history['step'][frame]}, T={history['temperature'][frame]:.2f}"
        )
        return sample_artist, title

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
