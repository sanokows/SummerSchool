# Summer School

The lessons follow the [presentation](Slides/main.pdf):

| Lesson | Topic | Exercise |
| --- | --- | --- |
| 1 | Variational inference | [Gaussian policy notebook](L1-VariationalInference/gaussian_policy_student.ipynb) |
| 2 | Diffusion samplers | [Diffusion sampler notebook](L2-DiffusionSamplers/diffusion_sampler_student.ipynb) |
| 3 | Reinforcement learning as variational inference | Covered in the slides |
| 4 | Diffusion-augmented MDPs | Covered in the slides |

Solution notebooks: [Lesson 1](L1-VariationalInference/gaussian_policy_solution.ipynb)
and [Lesson 2](L2-DiffusionSamplers/diffusion_sampler_solution.ipynb).

Each student and solution notebook starts with a **Hyperparameters** panel.
Settings are grouped by target, model, optimization, and diagnostics, with a
comment explaining every parameter. Edit the panel, then **Run All** so the
target, model, training loop, and plots all use the new values.

Start with Lesson 1 and implement the reparameterization and log-derivative
losses. Each exercise has optional **Get a hint** sections. Change `TEMPERATURE`
in the panel to compare variational inference at positive temperature with
reward maximization at `T = 0`.

Lesson 2 begins with two interactive kernel exercises:

1. Implement the forward step, then use sliders for the prior standard deviation,
   step count, and noise coefficients, with constant, linear, or cosine schedules.
   Play or scrub the animation to watch two Gaussian peaks merge. The histogram
   and particles come from your implementation; a reference density helps check it.
   Defaults use a standard-normal prior, 600 steps, a linear noise schedule
   from `delta_0 = 0.005` to `delta_K = 0.035`, and 4096 particles.
2. Implement the reverse step, then load the bundled pretrained neural score to
   recover the two peaks. Its prior, target, 600 steps, and matching linear schedule
   are fixed by the checkpoint, independently of the forward sliders.

Each visualization is directly below its task; the kernel log probabilities
are only needed afterward for GMM-40 training. Kernels receive `prior_std`
explicitly and work with both 1D demo states and 2D training states. If a task is
unfinished, the visualization shows a reminder; rerun it after implementing the
kernel. The controls use Jupyter widgets and require a running notebook kernel.
After updating an existing checkout, run `uv sync --locked` and restart JupyterLab
to install the widget dependency.

The 22 KB pretrained model and its configuration are in
`L2-DiffusionSamplers/assets/bimodal_score.{pt,json}`. No download or training is
needed to use it. Instructors can regenerate it with
`.venv/bin/python L2-DiffusionSamplers/train_demo_score.py`; its JSON records the
training settings and validation error. It learns the known forward-mixture
scores; the reverse demo evaluates the trained network.

The later Lesson 2 sampler supports learning the interior diffusion coefficients with
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
