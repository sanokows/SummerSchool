"""Plot the exact Multimodal Agent reward used in the benchmark histograms.

Source: DMERL/src/env_utils/turning_double_well_env.py, reward_from_angle;
parameters: DMERL/config/env/mjx_double_well.yaml.
The motion snaps turns to +/-45 degrees; rewards use the original action.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Benchmark and plot settings.
MAX_TURN_DEGREES = 90.0   # Action +/-1 corresponds to this relative turn.
WELL_ANGLE_DEGREES = 45.0 # Both reward-maximizing relative turns.
WELL_HEIGHT = 0.85        # Reward at a = 0, between the two maxima.
GRID_POINTS = 801         # Includes both maxima and the endpoints exactly.
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def reward(actions):
    u = np.clip(np.asarray(actions), -1, 1) ** 2
    peak_u = (WELL_ANGLE_DEGREES / MAX_TURN_DEGREES) ** 2
    inner = WELL_HEIGHT + (1 - WELL_HEIGHT) * (1 - (1 - u / peak_u) ** 2)
    outer = 1 - ((u - peak_u) / (1 - peak_u)) ** 2
    return np.where(u <= peak_u, inner, outer)


def main():
    actions = np.linspace(-1, 1, GRID_POINTS)
    values = reward(actions)
    np.testing.assert_allclose(reward([-1, -0.5, 0, 0.5, 1]), [0, 1, WELL_HEIGHT, 1, 0])
    np.testing.assert_allclose(values, values[::-1], atol=1e-12)
    assert values.max() == 1.0

    fig, axis = plt.subplots(figsize=(5.4, 3.6))
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.94)
    axis.fill_between(actions, values, color="#D89A43", alpha=0.12)
    axis.plot(actions, values, color="black", linewidth=2.6)
    for action, label in [(-0.5, r"$-45^\circ$"), (0.5, r"$+45^\circ$")]:
        axis.vlines(action, 0, 1, color="#9E6424", linestyle="--", linewidth=1.1)
        axis.text(action, 1.035, label, ha="center", va="bottom", color="#9E6424", fontsize=12)
    axis.set(xlim=(-1, 1), ylim=(0, 1.18), xticks=[-1, -0.5, 0, 0.5, 1],
             yticks=[0, 0.5, 1], xlabel=r"Action $a$", ylabel=r"Reward $R_{\mathrm{env}}(s,a)$")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#64748B")
    axis.tick_params(colors="#64748B", labelsize=10)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.grid(axis="y", color="#64748B", alpha=0.12)
    fig.savefig(ASSETS / "multimodal_agent_reward.pdf")
    plt.close(fig)
    (ASSETS / "multimodal_agent_reward.json").write_text(json.dumps({
        "source": "DMERL/src/env_utils/turning_double_well_env.py:reward_from_angle",
        "config": "DMERL/config/env/mjx_double_well.yaml",
        "max_turn_degrees": MAX_TURN_DEGREES,
        "well_angle_degrees": WELL_ANGLE_DEGREES,
        "well_height": WELL_HEIGHT,
        "reward_maximizing_actions": [-0.5, 0.5],
        "reward_depends_on_heading": False,
    }, indent=2) + "\n")
    print("Saved the exact benchmark reward curve (maxima at a = +/-0.5).")


if __name__ == "__main__":
    main()
