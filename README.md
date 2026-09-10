# Summer School

## Install the environment and start coding

The exercises run on CPU; no GPU is required. You need Git and
[uv](https://docs.astral.sh/uv/getting-started/installation/).
uv manages Python and the project dependencies for you.

### 1. Install uv

Linux/macOS terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a new terminal after installation, then check that `uv --version` and
`git --version` work. If Git is missing, install it using the
[Git installation instructions](https://git-scm.com/downloads).

### 2. Download the course and install its dependencies

```bash
git clone --depth 1 https://github.com/sanokows/SummerSchool.git
cd SummerSchool
uv sync --locked
```

This creates `.venv`, installs Python 3.12 if needed, and installs the versions
recorded in `uv.lock`, including CPU PyTorch, JupyterLab, and the interactive
widgets. The first installation needs an internet connection.

### 3. Start JupyterLab

Run this from the `SummerSchool` folder:

```bash
uv run --locked jupyter lab
```

Open the browser link printed in the terminal if JupyterLab does not open
automatically. Leave this terminal running while you work. No manual
environment activation is needed.

### 4. Open the first exercise

In JupyterLab, open
[`L1-VariationalInference/gaussian_policy_student.ipynb`](L1-VariationalInference/gaussian_policy_student.ipynb).
If prompted, choose **Python 3 (ipykernel)**; launching JupyterLab with `uv run`
uses the project's environment. See the
[uv Jupyter guide](https://docs.astral.sh/uv/guides/integration/jupyter/) for details.

- Run cells from top to bottom with **Shift+Enter**.
- Implement the marked tasks; expand **Get a hint** when needed.
- Unfinished tasks may raise `NotImplementedError` or show a reminder. Implement
  the function, rerun its cell, then continue.
- Once the tasks are complete, use **Restart Kernel and Run All Cells** to
  check the whole notebook.

The Lesson 2 sliders and animations need a running notebook kernel; the GitHub
preview only displays saved notebook content.

## Lessons and solutions

Follow along with the [presentation](Slides/main.pdf). Its references are grouped
into VI and diffusion foundations, diffusion samplers, RL, and diffusion policies
and RL. The talk closes with a brief pitch for
[Guided Discovery of New Behaviors using Diffusion Policies](https://arxiv.org/abs/2606.08743)
(accepted to CoRL) as another route to diverse behaviors: rare-case sampling,
shooting-based trajectory repair, and policy fine-tuning. The StackCube and
PushT slides link to the public [DA-MDP PyTorch repository](https://github.com/Atarilab/DA_MDP_pytorch)
with code QR codes; that repository includes installation instructions,
multimodal GIFs, working configs, and downloadable checkpoints.

The StackCube example compares animated DA:REPPO and DA:PPO overlays. The
DA-MDP lesson expands an ordinary policy decision into denoising decisions,
and a notation table provides a reference before the closing slide.

| Lesson | Topic | Exercise | Solution |
| --- | --- | --- | --- |
| 1 | Variational inference | [Gaussian policy](L1-VariationalInference/gaussian_policy_student.ipynb) | [Lesson 1 solution](L1-VariationalInference/gaussian_policy_solution.ipynb) |
| 2 | Diffusion samplers | [Diffusion sampler](L2-DiffusionSamplers/diffusion_sampler_student.ipynb) | [Lesson 2 solution](L2-DiffusionSamplers/diffusion_sampler_solution.ipynb) |
| 3 | Reinforcement learning as variational inference | Slides | — |
| 4 | Diffusion-augmented MDPs | Slides | — |

### Training runs and expected time

The assignment slides show what to implement, initial samples before training,
and a short time estimate. After implementing the tasks,
execute the default comparison cell to run every configuration:

| Exercise | Runs | Default size per run | Laptop execution budget |
| --- | --- | --- | --- |
| Lesson 1: Gaussian VI on GMM-40 | Reparameterization and log derivative (2 runs) | 800 updates, 512 samples/update | 1–2 min total, including GIF export |
| Lesson 2: 1D kernel demos | Forward simulation and reverse simulation with a pretrained score; no training | 600 diffusion steps, 4096 particles | A few seconds per simulation |
| Lesson 2: diffusion samplers on GMM-40 | Both estimators, each with Langevin off and on (4 runs) | 800 updates, 256 paths/update, 24 diffusion steps/path | 3–5 min total, including GIF export |

These are execution budgets, excluding coding and environment installation.
On an Intel i9-14900HX CPU, the default Lesson 1 training took 3 seconds total
and GIF export took 8 seconds. Lesson 2 took 84 seconds for all four runs and
final evaluation, plus 13 seconds for GIF export, using its default 2 CPU
threads. Slower laptops may take longer; no GPU is required.

The 26,000-update diffusion animations in the slides are prepared long runs.
Students run the shorter 800-update comparison to observe learning; full
convergence is not expected from every configuration. Temperature experiments
and learning the diffusion schedule are optional extensions. Lessons 3 and 4
use slides and prepared RL results, with no additional student training runs.

### Hyperparameters

Each student and solution notebook starts with a **Hyperparameters** panel.
Settings are grouped by target, model, optimization, and diagnostics, with a
comment explaining every parameter. After changing the panel, restart the kernel
and run the notebook from the top so all cells use the new values.

### Lesson 1: variational inference

Implement the reparameterization loss by explicitly constructing
`mean + std * epsilon`, then implement the log-derivative loss. Train both
Gaussian policies and compare their learning curves and sample animations.
Change `TEMPERATURE` to compare variational inference at positive temperature
with reward maximization at `T = 0`.

### Lesson 2: diffusion samplers

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
kernel.

The 22 KB pretrained model and its configuration are in
`L2-DiffusionSamplers/assets/bimodal_score.{pt,json}`. No download or training is
needed to use it. The reverse demo evaluates this trained network; its JSON
records the fixed diffusion settings and validation error.

The later GMM-40 experiment compares reparameterization and log derivative,
each with and without Langevin preconditioning. `TEMPERATURE` stays fixed:
there is no temperature annealing. The default 800 training updates keep the
exercise short; the later estimator-comparison slide shows separate
26,000-update runs.

Set `LEARN_DIFFUSION_SCHEDULE=True` to learn the interior diffusion coefficients
while keeping their endpoints fixed.
`PRIOR_STD` configures the fixed Gaussian prior; learning the prior itself is
discussed in the slides but is not implemented in the exercise.

## Update an existing checkout

Save your notebooks and stop JupyterLab. From the `SummerSchool` folder:

```bash
git pull --ff-only
uv sync --locked
uv run --locked jupyter lab
```

Restarting JupyterLab also loads any updated widget dependencies.
