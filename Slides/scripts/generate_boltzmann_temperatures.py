#!/usr/bin/env python3
"""Draw a double-well cost and its normalized Boltzmann densities."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INK = "#17243A"
SLATE = "#64748B"
TEMPERATURES = (0.3, 1.0, 3.0)
COLORS = ("#1E90FF", "#9E6424", "#487773")
STYLES = ("-", "--", ":")


def raw_cost(x: np.ndarray) -> np.ndarray:
    return (x**2 - 4) ** 2 / 8 + 0.35 * x


# Shift the global minimum to zero; this does not change Boltzmann densities.
STATIONARY_POINTS = np.sort(np.roots([0.5, 0, -2, 0.35]))
COST_OFFSET = raw_cost(STATIONARY_POINTS).min()


def cost(x: np.ndarray) -> np.ndarray:
    return raw_cost(x) - COST_OFFSET


def make_axes(ylabel: str, show_x: bool) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.24, top=0.91)
    ax.set(xlim=(-3.5, 3.5), xticks=[-3, -2, -1, 0, 1, 2, 3], ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(SLATE)
    ax.tick_params(colors=SLATE, length=3, labelbottom=show_x)
    if show_x:
        ax.set_xlabel(r"$x$")
    for minimum in STATIONARY_POINTS[[0, 2]]:
        ax.axvline(minimum, color=SLATE, alpha=0.3, linestyle="--", linewidth=1)
    return fig, ax


def save(fig: plt.Figure, name: str) -> None:
    path = ROOT / "assets" / name
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15,
                         "text.color": INK, "axes.labelcolor": INK,
                         "pdf.fonttype": 42})
    x = np.linspace(-3.5, 3.5, 2401)
    integration_x = np.linspace(-8, 8, 32001)
    integration_cost = cost(integration_x)

    fig, ax = make_axes(r"Cost $C(x)$", show_x=False)
    ax.plot(x, cost(x), color=INK, linewidth=2.7)
    ax.fill_between(x, cost(x), color=INK, alpha=0.06)
    ax.set(ylim=(-0.3, 10.8), yticks=[0, 5, 10])
    save(fig, "boltzmann_cost.pdf")

    fig, ax = make_axes(r"Density $p_{\mathcal{T}}(x)$", show_x=True)
    expected_costs = []
    for temperature, color, style in zip(TEMPERATURES, COLORS, STYLES):
        unnormalized = np.exp(-integration_cost / temperature)
        partition = np.trapezoid(unnormalized, integration_x)
        coarse_partition = np.trapezoid(unnormalized[::2], integration_x[::2])
        assert np.isclose(partition, coarse_partition, rtol=1e-9)
        density = unnormalized / partition
        assert max(density[0], density[-1]) < 1e-12
        expected_cost = np.trapezoid(density * integration_cost, integration_x)
        expected_costs.append(expected_cost)
        ax.plot(x, np.exp(-cost(x) / temperature) / partition,
                color=color, linestyle=style, linewidth=2.7,
                label=rf"$\mathcal{{T}}={temperature:g}$")
        print(f"T={temperature:g}: Z={partition:.6f}, E[C]={expected_cost:.6f}")
    assert np.all(np.diff(expected_costs) > 0)
    ax.set(ylim=(0, 1.65), yticks=[0, 0.8, 1.6])
    ax.legend(loc="upper right", frameon=False, handlelength=2.2,
              borderaxespad=0.1, labelspacing=0.35)
    save(fig, "boltzmann_temperatures.pdf")


if __name__ == "__main__":
    main()
