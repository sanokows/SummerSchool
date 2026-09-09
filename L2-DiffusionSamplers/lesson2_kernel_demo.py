"""Interactive 1D VP experiments driven by the student's transition functions.

The orange curves are exact forward marginals for comparison. Every histogram
and moving particle comes from calls to the supplied notebook kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DemoConfig:
    prior_std: float = 1.0
    num_steps: int = 600
    schedule: str = "linear"
    delta_start: float = 0.005
    delta_end: float = 0.035
    mode_location: float = 3.0
    component_std: float = 0.55

    def deltas(self) -> torch.Tensor:
        """K+1 coefficients, with the same adjacent indexing as the slides."""
        if self.num_steps < 2:
            raise ValueError("Use at least two diffusion steps.")
        if not all(math.isfinite(x) and x > 0 for x in (
            self.prior_std, self.mode_location, self.component_std,
            self.delta_start, self.delta_end,
        )):
            raise ValueError("Scales and noise coefficients must be positive and finite.")
        if max(self.delta_start, self.delta_end) >= 2:
            raise ValueError("Require delta < 2 for positive VP shrinkage.")
        time = torch.linspace(0, 1, self.num_steps + 1)
        if self.schedule == "constant":
            return torch.full_like(time, self.delta_end)
        if self.schedule == "linear":
            ramp = time
        elif self.schedule == "cosine":
            # A cosine ramp of delta itself, not the DDPM cumulative-alpha schedule.
            ramp = 0.5 * (1 - torch.cos(math.pi * time))
        else:
            raise ValueError(f"Unknown noise schedule: {self.schedule}")
        return self.delta_start + (self.delta_end - self.delta_start) * ramp


def marginal_parameters(config: DemoConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive component mean and common variance at k=0,...,K."""
    means = torch.empty(config.num_steps + 1)
    variances = torch.empty_like(means)
    means[0], variances[0] = config.mode_location, config.component_std**2
    for k, delta in enumerate(config.deltas()[:-1], 1):
        shrink = 1 - 0.5 * delta
        means[k] = shrink * means[k - 1]
        variances[k] = shrink.square() * variances[k - 1] + config.prior_std**2 * delta
    return means, variances


def mixture_score(states, means, variances):
    """Exact marginal score, used only to train/validate the bundled model."""
    return (means * torch.tanh(states * means / variances) - states) / variances


class DemoScoreNetwork(nn.Module):
    """Small time-conditioned MLP; no target density is evaluated at inference."""

    def __init__(self, width: int = 64, prior_std: float = 1.0):
        super().__init__()
        self.prior_std = prior_std
        self.residual = nn.Sequential(
            nn.Linear(6, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, states, step_index, num_steps):
        time = torch.as_tensor(step_index, device=states.device, dtype=states.dtype) / num_steps
        time = torch.broadcast_to(time, states.shape)
        times = torch.cat((time, torch.sin(math.pi * time), torch.cos(math.pi * time),
                           torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)), -1)
        position = states / self.prior_std
        positive = self.residual(torch.cat((position, times), -1))
        negative = self.residual(torch.cat((-position, times), -1))
        # The equally weighted symmetric target has an odd score. Enforcing
        # this symmetry keeps approximation error from favoring either mode.
        return -states / self.prior_std**2 + (positive - negative) / 2


@dataclass(frozen=True)
class PretrainedDemo:
    config: DemoConfig
    score: DemoScoreNetwork
    metadata: dict


def load_pretrained_demo(checkpoint_path: str | Path) -> PretrainedDemo:
    """Load weights and their fixed prior, target, and diffusion configuration."""
    path = Path(checkpoint_path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["format_version"] != 1:
        raise ValueError("Unsupported demo checkpoint format.")
    config = DemoConfig(**metadata["diffusion"])
    config.deltas()  # Validate the configuration before using its weights.
    model = DemoScoreNetwork(width=metadata["model_width"], prior_std=config.prior_std)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    return PretrainedDemo(config, model, metadata)


@dataclass
class DemoPath:
    config: DemoConfig
    states: torch.Tensor  # [K+1, particles, 1], indexed by mathematical k.
    reverse: bool = False


def _check_state(state, expected_shape):
    if not isinstance(state, torch.Tensor) or state.shape != expected_shape:
        raise ValueError(f"Your kernel must return a tensor of shape {expected_shape}.")
    if not torch.isfinite(state).all():
        raise ValueError("Your kernel produced non-finite states; check its mean and noise scale.")


@torch.no_grad()
def simulate_forward(forward_step, config: DemoConfig, *, num_samples=4096, seed=17) -> DemoPath:
    generator = torch.Generator().manual_seed(seed)
    signs = 2 * torch.randint(0, 2, (num_samples, 1), generator=generator) - 1
    state = signs * config.mode_location + config.component_std * torch.randn(
        num_samples, 1, generator=generator)
    states = [state.clone()]
    for delta in config.deltas()[:-1]:
        state = forward_step(
            state, delta, prior_std=config.prior_std, reparameterize=False,
            noise=torch.randn(state.shape, generator=generator),
        )
        _check_state(state, states[0].shape)
        states.append(state.clone())
    return DemoPath(config, torch.stack(states))


@torch.no_grad()
def simulate_reverse(reverse_step, pretrained: PretrainedDemo, *, num_samples=4096, seed=17) -> DemoPath:
    # All generative settings come from the checkpoint, never from the forward controls.
    config, score = pretrained.config, pretrained.score
    generator = torch.Generator().manual_seed(seed)
    state = config.prior_std * torch.randn(num_samples, 1, generator=generator)
    states = torch.empty(config.num_steps + 1, num_samples, 1)
    states[-1] = state
    deltas = config.deltas()
    for k in range(config.num_steps, 0, -1):
        state = reverse_step(
            score, state, k, deltas[k], config.num_steps,
            prior_std=config.prior_std, reparameterize=False,
            noise=torch.randn(state.shape, generator=generator),
        )
        _check_state(state, states[-1].shape)
        states[k - 1] = state
    return DemoPath(config, states, reverse=True)


class KernelDemo:
    """Play/pause and scrub actual kernel samples in a standard Jupyter widget."""

    def __init__(self, simulate, config, *, adjustable, frame_count=81, interval_ms=120):
        import ipywidgets as widgets
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self._simulate = simulate
        self._config = config
        self._frame_count = frame_count
        self._updating = False
        self.path = None
        self.controls = {}
        self.status = widgets.HTML()
        self.image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1050px"))
        self.play = widgets.Play(min=0, max=frame_count - 1, interval=interval_ms)
        self.frame = widgets.IntSlider(min=0, max=frame_count - 1, description="Frame",
                                      continuous_update=False, layout=widgets.Layout(width="75%"))
        self._play_link = widgets.jslink((self.play, "value"), (self.frame, "value"))
        self.frame.observe(self._draw, names="value")
        self.figure = Figure(figsize=(10.5, 4.9), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasAgg(self.figure)
        grid = self.figure.add_gridspec(2, 2, height_ratios=(5, 1), width_ratios=(3, 1))
        self.density_axis = self.figure.add_subplot(grid[0, 0])
        self.particle_axis = self.figure.add_subplot(grid[1, 0], sharex=self.density_axis)
        self.schedule_axis = self.figure.add_subplot(grid[:, 1])

        children = []
        if adjustable:
            slider_args = dict(continuous_update=False, style={"description_width": "95px"},
                               layout=widgets.Layout(width="310px"))
            self.controls = {
                "prior_std": widgets.FloatSlider(description="Prior std ν", value=config.prior_std,
                                                  min=0.5, max=3, step=0.1, **slider_args),
                "num_steps": widgets.IntSlider(description="Steps K", value=config.num_steps,
                                                min=20, max=1000, step=20, **slider_args),
                "schedule": widgets.Dropdown(description="Schedule", value=config.schedule,
                                              options=[("Constant", "constant"), ("Linear", "linear"),
                                                       ("Cosine ramp", "cosine")],
                                              style={"description_width": "95px"},
                                              layout=widgets.Layout(width="310px")),
                "delta_start": widgets.FloatSlider(description="Start δ₀", value=config.delta_start,
                                                     min=0.001, max=0.08, step=0.001,
                                                     readout_format=".3f", **slider_args),
                "delta_end": widgets.FloatSlider(description="End δK", value=config.delta_end,
                                                   min=0.001, max=0.08, step=0.001,
                                                   readout_format=".3f", **slider_args),
            }
            children.append(widgets.Box(list(self.controls.values()), layout=widgets.Layout(
                display="flex", flex_flow="row wrap")))
            for control in self.controls.values():
                control.observe(self._refresh, names="value")
        else:
            schedule_label = (f"constant δ={config.delta_end:g}" if config.schedule == "constant"
                              else f"{config.schedule} δ₀={config.delta_start:g} → δK={config.delta_end:g}")
            children.append(widgets.HTML(
                f"<b>Fixed checkpoint settings:</b> N(0, {config.prior_std:g}²), "
                f"K={config.num_steps}, {schedule_label}. "
                "The forward sliders do not change these settings."))
        children.extend([widgets.HBox([self.play, self.frame]), self.image, self.status])
        self.widget = widgets.VBox(children)
        self._refresh()

    def _refresh(self, change=None):
        from html import escape

        self.play.playing = False
        self._updating = True
        self.image.value = b""
        self.path = None
        self.status.value = "Running your kernel…"
        try:
            if self.controls:
                values = {name: widget.value for name, widget in self.controls.items()}
                self.controls["delta_start"].disabled = values["schedule"] == "constant"
                self._config = DemoConfig(**values, mode_location=self._config.mode_location,
                                          component_std=self._config.component_std)
            self.path = self._simulate(self._config)
            self._means, self._variances = marginal_parameters(self.path.config)
            # Quadratic spacing slows down the interesting small-k transitions.
            self._steps = np.unique(np.rint(
                self.path.config.num_steps * np.linspace(0, 1, self._frame_count)**2
            ).astype(int))
            if self.path.reverse:
                self._steps = self._steps[::-1].copy()
            self.frame.max = self.play.max = len(self._steps) - 1
            self.frame.value = self.play.value = 0
            self.frame.disabled = self.play.disabled = False
            self.status.value = (
                "<b>Blue histogram and particles:</b> your kernel. "
                "<b>Orange curve:</b> exact forward marginal for comparison. "
                "Use ▶ or drag the frame slider. "
                + ("Reverse sampling is approximate because the score is learned and steps are finite."
                   if self.path.reverse else
                   "Small steps approach the Gaussian prior approximately; too little noise can leave two peaks.")
            )
        except Exception as exc:
            self.frame.disabled = self.play.disabled = True
            self.status.value = (
                f"<b>{escape(type(exc).__name__)}:</b> {escape(str(exc))}. "
                "Implement or fix the kernel above, then rerun this visualization cell."
            )
        finally:
            self._updating = False
        self._draw()

    def _draw(self, change=None):
        if self._updating or self.path is None:
            return
        config = self.path.config
        k = int(self._steps[self.frame.value])
        samples = self.path.states[k, :, 0].numpy()
        bound = max(config.mode_location + 4 * config.component_std, 4.5 * config.prior_std)
        x = np.linspace(-bound, bound, 501)
        mean, variance = float(self._means[k]), float(self._variances[k])

        def normal_pdf(mean, variance):
            return np.exp(-0.5 * (x - mean)**2 / variance) / math.sqrt(2 * math.pi * variance)

        exact = (normal_pdf(mean, variance) + normal_pdf(-mean, variance)) / 2
        endpoint = ((normal_pdf(config.mode_location, config.component_std**2)
                     + normal_pdf(-config.mode_location, config.component_std**2)) / 2
                    if self.path.reverse else normal_pdf(0, config.prior_std**2))
        axis = self.density_axis
        axis.clear()
        bins = np.linspace(-bound, bound, 71)
        counts, _ = np.histogram(samples, bins=bins)
        heights = counts / (len(samples) * np.diff(bins))
        axis.stairs(heights, bins, fill=True, color="#1E90FF", alpha=0.35, label="Your kernel samples")
        axis.plot(x, exact, color="#B97819", linewidth=2.2, label=f"Reference p{k}")
        axis.plot(x, endpoint, color="#17243A", linestyle="--", linewidth=1.6,
                  label="Bimodal target" if self.path.reverse else "Gaussian prior")
        axis.set(xlim=(-bound, bound), ylabel="Density", title=(
            f"{'Reverse denoising' if self.path.reverse else 'Forward noising'}: k = {k} / {config.num_steps}"))
        axis.legend(loc="upper right", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelbottom=False)
        axis.set_ylim(0, max(exact.max(), endpoint.max(), heights.max()) * 1.28)
        dots = self.particle_axis
        dots.clear()
        count = min(160, len(samples))
        dots.scatter(samples[:count], np.random.default_rng(1).uniform(-0.7, 0.7, count),
                     s=9, alpha=0.5, color="#1E90FF")
        dots.set(xlim=(-bound, bound), ylim=(-1, 1), yticks=[], xlabel="x")
        dots.set_ylabel("Particles", fontsize=9)
        dots.spines[["top", "left", "right"]].set_visible(False)

        axis = self.schedule_axis
        axis.clear()
        deltas = config.deltas().numpy()
        axis.plot(np.arange(config.num_steps + 1), deltas, color="#17243A")
        axis.axvline(k, color="#1E90FF", linestyle="--")
        axis.set(xlabel="Diffusion index k", ylabel="Noise coefficient δₖ", title="Noise schedule",
                 ylim=(0, max(deltas) * 1.2))
        axis.spines[["top", "right"]].set_visible(False)
        outside = float(np.mean(np.abs(samples) > bound))
        self.figure.suptitle(
            f"Samples: mean {samples.mean():.2f} · std {samples.std():.2f} · "
            f"left / right {np.mean(samples < 0):.0%} / {np.mean(samples >= 0):.0%}"
            + (f" · outside plot {outside:.1%}" if outside else ""), fontsize=10,
        )
        output = BytesIO()
        self.canvas.print_png(output)
        self.image.value = output.getvalue()

    def _ipython_display_(self):
        from IPython.display import display
        display(self.widget)


def forward_demo(forward_step, *, config=DemoConfig(), num_samples=4096, seed=17,
                 frame_count=81, interval_ms=120):
    return KernelDemo(
        lambda settings: simulate_forward(forward_step, settings, num_samples=num_samples, seed=seed),
        config, adjustable=True, frame_count=frame_count, interval_ms=interval_ms,
    )


def reverse_demo(reverse_step, pretrained, *, num_samples=4096, seed=17,
                 frame_count=81, interval_ms=120):
    return KernelDemo(
        lambda _: simulate_reverse(reverse_step, pretrained, num_samples=num_samples, seed=seed),
        pretrained.config, adjustable=False, frame_count=frame_count, interval_ms=interval_ms,
    )
