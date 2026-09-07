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
BETA_START = 5.0e-4
BETA_END = 2.0e-3
PRIOR_STD = 30.0
PRIOR_VARIANCE = PRIOR_STD**2
LANGEVIN_GRADIENT_CLIP = 1.0e2
LANGEVIN_DRIFT_CLIP = 1.0e4

NUM_MODES = 40
MODE_BOUND = 40.0
TARGET_VARIANCE = 1.0
TARGET_STD = math.sqrt(TARGET_VARIANCE)
LOG_MIXTURE_WEIGHT = -math.log(NUM_MODES)
_mode_generator = torch.Generator().manual_seed(0)
MODE_LOCATIONS = (
    2.0
    * MODE_BOUND
    * torch.rand(NUM_MODES, STATE_DIMENSION, generator=_mode_generator)
    - MODE_BOUND
)


def target_reward(states: torch.Tensor) -> torch.Tensor:
    """Log density of the seed-0, equal-weight GMM-40 benchmark."""
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
    """Analytic, detached gradient of log pi_T(x) = R(x) / temperature."""
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
    return (gradient / temperature).detach()


PLOT_BOUND = 56.0
PLOT_AXIS = torch.linspace(-PLOT_BOUND, PLOT_BOUND, 360)
GRID_X, GRID_Y = torch.meshgrid(PLOT_AXIS, PLOT_AXIS, indexing="xy")
GRID_POINTS = torch.stack((GRID_X, GRID_Y), dim=-1)
REWARD_ON_GRID = target_reward(GRID_POINTS)
DISPLAY_REWARD = REWARD_ON_GRID.clamp(min=REWARD_ON_GRID.max() - 100.0)
REWARD_LEVELS = torch.linspace(
    DISPLAY_REWARD.min().item(), DISPLAY_REWARD.max().item(), 80
)


def plot_reward_landscape(axis: plt.Axes) -> None:
    """Draw the paper-style GMM-40 contours and component centers."""
    axis.contour(
        GRID_X,
        GRID_Y,
        DISPLAY_REWARD,
        levels=REWARD_LEVELS,
        cmap="viridis",
        linewidths=0.55,
        alpha=0.9,
    )
    axis.scatter(
        MODE_LOCATIONS[:, 0],
        MODE_LOCATIONS[:, 1],
        s=9,
        color="#0067B1",
        alpha=0.82,
        zorder=3,
        label="GMM component means",
    )
    axis.set(
        xlim=(-PLOT_BOUND, PLOT_BOUND),
        ylim=(-PLOT_BOUND, PLOT_BOUND),
        xticks=(-40, 0, 40),
        yticks=(-40, 0, 40),
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        aspect="equal",
    )
    axis.set_facecolor("white")


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

        # score=-x/sigma^2 leaves N(0, sigma^2 I) invariant under the
        # correspondingly scaled VP kernel used in this benchmark.
        return -states / PRIOR_VARIANCE + drift_correction


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


def prior_log_prob(states: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        states.square() / PRIOR_VARIANCE
        + math.log(2.0 * math.pi * PRIOR_VARIANCE)
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
            state = PRIOR_STD * torch.randn(
                num_samples, STATE_DIMENSION, device=self.device
            )
        else:
            state = PRIOR_STD * prior_noise.to(self.device)
            if state.shape != (num_samples, STATE_DIMENSION):
                raise ValueError("prior_noise has the wrong shape.")

        if step_noises is not None:
            expected_shape = (self.num_steps, num_samples, STATE_DIMENSION)
            if step_noises.shape != expected_shape:
                raise ValueError(f"step_noises must have shape {expected_shape}.")
            step_noises = step_noises.to(self.device)

        states: list[torch.Tensor | None] = [None] * (self.num_steps + 1)
        states[self.num_steps] = state
        log_q = prior_log_prob(state)
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
    learning_rate: float = 5.0e-4,
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
        "evaluation_loss": [],
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
                history["evaluation_loss"].append(
                    sampler.path_cost(fixed_path).mean().item()
                )
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
    loss_axis.grid(alpha=0.2)

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
            "Lesson 4 diffusion sampler\n"
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
) -> animation.FuncAnimation:
    """Save the 2x2 estimator-by-preconditioning training comparison."""
    if len(experiments) != 4:
        raise ValueError("The comparison requires exactly four experiments.")

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
                color="#0067B1",
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
        for (_, use_langevin, history), color in zip(
            row_experiments, ("#64748B", "#0067B1")
        ):
            label = "+ Langevin" if use_langevin else "without Langevin"
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
        estimator_label = "Reparameterization" if row == 0 else "Log derivative"
        loss_axis.set(
            title=f"Path loss · {estimator_label}",
            xlabel="optimization step",
            ylabel="fixed-noise path objective",
            xlim=(
                row_experiments[0][2]["step"][0],
                row_experiments[0][2]["step"][-1],
            ),
        )
        loss_axis.grid(alpha=0.2)
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
            "GMM-40 diffusion sampler training  |  "
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
        frames=frame_counts.pop(),
        interval=700,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_animation.save(
        output_path, writer=animation.PillowWriter(fps=1.5), dpi=92
    )
    plt.close(fig)
    return comparison_animation
