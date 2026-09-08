# Summer School

The lessons follow the [presentation](Slides/main.pdf):

| Lesson | Topic | Exercise |
| --- | --- | --- |
| 1 | Variational inference | [Gaussian policy notebook](L1-VariationalInference/gaussian_policy_student.ipynb) |
| 2 | Diffusion samplers | [Diffusion sampler notebook](L2-DiffusionSamplers/diffusion_sampler_student.ipynb) |
| 3 | Reinforcement learning as variational inference | Covered in the slides |
| 4 | Diffusion-augmented MDPs | Covered in the slides |

Each student and solution notebook starts with a **Hyperparameters** panel.
Settings are grouped by target, model, optimization, and diagnostics, with a
comment explaining every parameter. Edit the panel, then **Run All** so the
target, model, training loop, and plots all use the new values.

Start with Lesson 1 and implement the reparameterization and log-derivative
losses. Each exercise has optional **Get a hint** sections. Change `TEMPERATURE`
in the panel to compare variational inference at positive temperature with
reward maximization at `T = 0`.

Lesson 2 supports learning the interior diffusion coefficients with
`LEARN_DIFFUSION_SCHEDULE=True`, while keeping their endpoints fixed.
`PRIOR_STD` configures the fixed Gaussian prior; learning the prior itself is
discussed in the slides but is not implemented in the exercise.

## Installation

### 1. Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Clone and install the environment

```bash
git clone https://github.com/sanokows/SummerSchool.git
cd SummerSchool
uv sync --locked
```

This installs Python 3.12 and all dependencies in `.venv`.

### 3. Start JupyterLab

```bash
uv run jupyter lab
```
