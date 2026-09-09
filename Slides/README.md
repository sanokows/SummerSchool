# Slides

This directory contains the Beamer source for **From Variational Inference to
Diffusion-Based Reinforcement Learning**.

The main deck's opening learning path marks variational inference and diffusion
samplers with a blue code symbol and “Coding exercise” label.

Equation colors are selective and follow the existing figures: blue highlights
the learned distribution/policy and entropy terms, amber the target, reward,
or forward reference, purple added latent variables, and gray fixed rollout
quantities, noise, or dynamics. Most operators, constants, indices, and simple
definitions stay neutral. Local comparisons reuse these colors (for example,
REPPO's existing blue policy update and amber KL correction). Code snippets
remain monochrome.

The separate nine-slide DA-MDP research introduction is `intro.pdf`, built from
`intro.tex`. Its approximately 5-minute speaking script is in
`intro_notes.md`; present it immediately before `main.pdf`. The introduction
has no Beamer reveal overlays, so each slide is exactly one PDF page. Its robot
videos are embedded as autoplaying, looping frame animations.

The opening slide shows animated humanoid and StackCube research previews. The
introduction also includes two silent real-world humanoid excerpts
from `IMG_4622.MOV`: 1:40--1:52 and 0:10--0:20. PDF-compatible JPEG frames are
in `assets/humanoid_motion_a/` and `assets/humanoid_motion_b/`, respectively.
The improved silent simulation video is `assets/humanoid_sim_gallery.mp4`: it
replays nine successful pickup motions sampled from the fine-tuned
`checkpoint_step_0000364.pt` policy together in one MuJoCo scene. Its exact
render settings are recorded in `assets/humanoid_sim_gallery.json`, and the
PDF-compatible animation frames are under `assets/humanoid_sim_gallery/`.
The humanoid comparison slide places data-only training on the left, this
fine-tuned gallery in the middle, and the two real-world excerpts playing
consecutively on the right. `assets/humanoid_sim_pretrained_gallery/` contains
five successful pretrained rollouts and four selected recorded failure cases,
rendered with the same camera. After each recording ends at its failure
threshold, a passive MuJoCo continuation lets the robot and box fall and
settle. Actuation is disabled for this illustrative continuation; it does not
represent further execution of the learned tracker. The selection emphasizes
failure examples, rather than estimating a success rate. The middle gallery
retains its original nine successful pickups.
The baseline video and provenance are stored alongside its frame directory
as `.mp4` and `.json` files. To regenerate them using the existing
`humanoid_diff` environment and archived trajectories, run
`scripts/render_humanoid_pretrained_gallery.py --humanoid-root /path/to/humanoid_diff`.

## Build

From this directory, run:

```bash
make
```

or invoke Tectonic directly:

```bash
tectonic --keep-logs --synctex main.tex
```

The resulting deck is `main.pdf`. The checked-in images under `assets/`
include the robotics-motivation, Lesson 1, and Lesson 2 PDF-compatible
animation frames, so the TeX source compiles without executing the notebooks
first. The StackCube overlay is also available as
`assets/stackcube_symmetric_overlay.gif`. Lesson 1 (variational inference)
uses `assets/l2_training.gif`, derived from
`L1-VariationalInference/maximum_entropy_reward_training.gif`; the Lesson 2
diffusion-sampler exercise GIF is `L2-DiffusionSamplers/diffusion_sampler_training.gif`.
The slides use the longer fixed-temperature comparison in
`assets/diffusion_estimator_training.gif`.
The Multimodal Agent benchmark includes a REPPO-only animation, an analytic
optimal-policy reference that samples both reward-maximizing turns, and a
synchronized REPPO-versus-DA:REPPO comparison. Their GIFs and PDF-compatible
frame sequences are stored under `assets/multimodal_agent_*`.
The benchmark is introduced before the REPPO results: a short state/action
description, the bimodal reward landscape on the left, and the existing
reward-optimal animation on the right. The reference chooses uniformly between
actions -0.5 and +0.5 (turns of -45 and +45 degrees) at each step; its colored
lines identify rollouts. The REPPO histogram and behavior comparison follow.
Motion snaps the commanded turn to +/-45 degrees and then advances one step;
the reward uses the original action. `assets/multimodal_agent_reward.pdf` plots
the exact benchmark reward, with maxima 1 at +/-0.5, central reward 0.85, and
endpoint reward 0. It is independent of the heading. Regenerate it from the
workspace root with
`.venv/bin/python Slides/scripts/generate_multimodal_agent_reward.py`;
the companion JSON records the source implementation and parameters.
Both histogram slides include the same visible state-heading legend, using the
original figure's exact colors: 0° cyan, 45° blue, 90° violet, 135° magenta,
180° red, 225° orange, 270° lime, and 315° green. Each histogram is the action
distribution conditioned on that state; the black line is the reward curve.
In the main deck, “What multimodality buys us” comes before the StackCube and
PushT demonstrations. Both demonstrations include a QR code linking to the
[DA-MDP paper](https://arxiv.org/abs/2512.02019) and the note “Updated version
in about 2 weeks.” This relative update announcement was added on September 9,
2026 and should be revised when the new paper version is available.
Immediately after StackCube, the main deck shows the paper's two successful
PushT rollouts side by side: clockwise (policy seed 20001) and counter-clockwise
(policy seed 20000), with the same initial block yaw of 192 degrees and
environment seed 10000. The synchronized original video is
`assets/pusht_multimodal.mp4`; provenance and rollout metadata are in
`assets/pusht_multimodal.json`. Its 117 PDF frames preserve the original 10 fps
and complete motions, holding the earlier success until both finish. The
diagnostic header/footer are cropped from the slide animation; direction
labels are supplied in TeX. Regenerate the frames from this directory using
FFmpeg (or the executable bundled with `imageio_ffmpeg`):

```bash
ffmpeg -i assets/pusht_multimodal.mp4 -vf 'crop=1024:512:0:78' \
  -q:v 2 -start_number 0 -y assets/pusht_multimodal/frame-%d.jpg
```

The research introduction's forward-noising and learned-reverse transport illustration includes the
GMM-40 cost landscape with samples overlaid on its low-cost basins. It is
stored as `assets/diffusion_transport.gif`; its PDF-compatible frames are in
`assets/diffusion_transport/` and can be regenerated with
`scripts/generate_diffusion_transport.py`.

The Lesson 2 starter-code assignment slide includes side-by-side animations of
forward noising and pretrained reverse sampling, below their respective kernel
TODOs. Both histograms are generated
by the solution notebook's sampling kernels with the notebook's linear-schedule
presets: standard-normal prior, 600 steps, `delta_0 = 0.005` to `delta_K = 0.035`,
4096 samples, and seed 17. Dashed curves show the destination densities.
The reverse animation loads the same bundled neural score as the exercise.
Regenerate the GIF, PDF frames, TeX include, and provenance JSON with
`.venv/bin/python Slides/scripts/generate_kernel_preview.py` from the workspace
root (requires the local instructor solution notebook). Outputs are
`assets/kernel_preview.gif`, `assets/kernel_preview/`, and the paired `.tex`
and `.json` files. The generator checks the forward Gaussian moments and
reverse mode locations, widths, and balance before rendering.

Lesson 2 keeps the Gaussian-versus-diffusion family comparison, then introduces
discrete forward noising, reverse sampling with the optimal score and then its
learned approximation, and learning from a cost. The DPI is introduced using
a latent variable `Z`, then applied to diffusion by setting `X = X^0` and
`Z = X^{1:K}`. The forward slide places the equations from
[Section 2.3 of the paper](https://arxiv.org/html/2512.02019v3#S2.SS3) beside
`assets/vp_diffusion_1d.gif`, with PDF animation frames in
`assets/vp_diffusion_1d/`. It evolves a symmetric mixture with means -3 and 3
and component standard deviation 0.55 toward a unit-Gaussian prior using
600 discrete steps with `delta = 0.02`. The density curves are exact mixture
marginals of that discrete process, and the dots follow its Markov updates.
Finite Euler steps have stationary variance `nu^2 / (1 - delta / 4)`, so the
displayed convergence to the prior is approximate. Regenerate the animation
from the workspace root with
`.venv/bin/python Slides/scripts/generate_vp_diffusion_1d.py`; the companion
JSON records its parameters and displayed step indices.

The student notebook, solution, and slide code use those same discrete Euler
updates: forward mean `(1 - delta_{k-1}/2) * x^{k-1}`, reverse mean
`(1 + delta_k/2) * x^k + nu^2 * delta_k * u_Psi(x^k, k)`. Both kernels use
standard deviation `nu * sqrt(delta)` with their respective coefficient.
The exercise sets `Delta = 1`, so `delta_k = beta_k`, and stores `K+1`
coefficients for `K` transitions. The sampler passes the appropriate coefficient
as `delta`; reverse network indices run from `K` down to `1`. Sampling and
log-probability evaluation use identical means and variances. There is no
separate exercise discretization or continuous-time SDE section.

The solution slide samples explicitly with `epsilon = torch.randn_like(mean)`
and `sample = mean + sigma * epsilon`, then applies `sample.detach()` when
`reparameterize=False`. Gaussian distribution objects are used only for the
log-probabilities. The notebook uses the same construction and also accepts
caller-supplied noise for reproducible animations and comparisons; that optional
noise-handling detail is omitted from the slide.

The notebook now visualizes each sampling kernel immediately after its task.
Its first demo uses the same 1D two-peak target as the forward-noising slide,
with sliders for prior scale, step count, and noise schedule. Its second demo
loads `L2-DiffusionSamplers/assets/bimodal_score.pt` and the paired JSON, fixing
the pretrained score's target, standard-normal prior, and 600-step linear schedule
(`delta_0 = 0.005`, `delta_K = 0.035`). Both notebook demos start with these
settings and 4096 particles. Histograms
and particles are generated by the student's kernels; exact forward marginals
are overlaid as references. All four kernel functions accept an explicit
`prior_std`, keeping these demos independent of each other and of GMM-40 training.

The sampler introduction mentions that both the prior and diffusion
coefficients can be learned. The exercise supports learning the monotone
interior coefficients; its Gaussian prior is fixed with a configurable scale.
The optional Langevin slide contains only the target-gradient identity and
the score parameterization using that gradient.

Both lessons' student and solution notebooks begin with a commented
hyperparameter panel. It controls the target geometry, model, training,
evaluation, and animation. After an edit, use **Run All**. Lesson 2 forwards
these values explicitly to its shared helpers, including the prior scale,
diffusion schedule, score width, and fixed target temperature. Lesson 2 keeps
`TEMPERATURE` constant from initialization through the final update; there
are no temperature-annealing parameters in the exercise or shared trainer.

Instructor checks cover the interactive controls, recovered bimodal samples,
the pretrained score, solution kernels, path indexing, gradients, and short training runs
for both estimators with fixed/learned schedules and Langevin preconditioning
on/off. They require the local `diffusion_sampler_solution.ipynb`, which is
intentionally excluded from Git; a student checkout reports an explicit skip.
Run them from the workspace root:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m unittest discover -s L2-DiffusionSamplers -v
```

Running the solution notebook executes the four default 800-update experiments
and regenerates `L2-DiffusionSamplers/diffusion_sampler_training.gif` at fixed
temperature. The slides instead use the completed 26,000-update runs at T=1:
columns compare reparameterization and log derivative, rows compare Langevin
off and on, and each right-hand plot compares both estimators' path losses.
The target minima have no dot markers. All four runs use seed 11, the same
training noise, 600 display particles, and 5,000 fixed evaluation paths.
Their learning rate is 0.0005 through update 20,000, decreases by the same
cosine schedule to 0.00002 at update 24,000, and stays there for the last
2,000 updates. This optimizer schedule is distinct from temperature annealing;
the target temperature remains 1 throughout.

The final losses are 13.894 (reparameterization) and 8.645 (log derivative)
without Langevin, and 4.799 and 4.854 with Langevin. These are one-seed results.
Plot data are checked in as `assets/diffusion_estimator_training.npz`; the
companion JSON records hyperparameters, source checkpoint hashes, and displayed
updates. Regenerate the GIF, PDF frames under `assets/l4_training/`, and TeX
animation include without retraining:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python Slides/scripts/generate_diffusion_estimator_comparison.py
```

Run this from the workspace root. To refresh the plotting data from the four
original fixed-temperature checkpoints, add
`--checkpoints artifacts/annealing_convergence`.

Lesson 1 starts with a cost, constructs its Boltzmann distribution, and motivates
approximating it with a tractable distribution. Beside the introductory bullets,
aligned plots show a double-well cost and normalized Boltzmann densities at
temperatures 0.3, 1, and 3. Regenerate `assets/boltzmann_cost.pdf` and
`assets/boltzmann_temperatures.pdf` from the workspace root with
`.venv/bin/python Slides/scripts/generate_boltzmann_temperatures.py`.
One compact KL slide then gives
the integral and expectation definitions, key properties, and a side-by-side
forward/reverse KL comparison. The vector plots `assets/kl_forward.pdf` and
`assets/kl_reverse.pdf` fit one Gaussian to the same two-component mixture.
Forward KL uses exact moment matching; reverse KL uses deterministic quadrature
and optimization from multiple starting points. Regenerate both from the
workspace root with `.venv/bin/python Slides/scripts/generate_kl_comparison.py`.

The reparameterization slide uses a one-dimensional Gaussian:
`X = mu + sigma * epsilon`, with standard-normal noise held fixed during
backpropagation. It gives the reward derivatives with respect to the mean and
standard deviation and the VI loss with analytic Gaussian entropy.

Lesson 3 opens with the reward-weighted trajectory target and its intractable
action marginal. A static branching tree distinguishes blue candidate-action
choices from gray stochastic environment transitions: each fixed action has
multiple possible next states, and the complete trajectories have separate
returns. The target includes both the dynamics factors and the exponential
return weight; the blue action edges do not represent target-policy
probabilities. This slide uses ordinary actions `a_t` and horizon `T`.

The next slide shows a full, static timeline of state--action pairs, transition
probabilities, environment rewards, and their cumulative return. It introduces
ordinary rewards only. Short bullets on the following slide define the
policy-induced density, normalized reward target, and both action marginals
above the VI objective. A single static derivation slide combines the DPI
inequality, joint-KL expansion, and maximum-entropy objective. The bound is
distinguished from the exact equivalence between minimizing the joint KL and
maximizing the return.

Immediately after that derivation, the complete timeline returns with soft
rewards: each pair adds the blue sampled entropy contribution
`h_t = -temperature * log q_theta(a_t | s_t)` to its environment reward. Their sum feeds
into the finite-horizon maximum-entropy return. A short identity distinguishes
the sampled contribution from conditional policy entropy, its expectation.
Both versions show the whole figure at once; discounting is introduced later.

Lesson 3 identifies the discounted infinite-horizon return as the standard
MaxEnt RL objective and cites Soft Actor-Critic. One compact slide gives the
local actor loss from the policy-gradient theorem, defines `Q^{theta_r}` and
`V^{theta_r}` through the soft Bellman equations to show their dependence on
the rollout parameters, and states the gradient identity without deriving
the surrogate. The
surrogate gradient equals the negative return gradient at the rollout
parameters, with exact values and expectations. Later updates on the same
batch are local approximations. The following slide keeps the TRPO motivation
compact: several updates with the rollout critic and state distribution fixed,
while staying close to the original rollout policy. One practical example adds
`lambda_KL * E_dr KL(q_theta_r || q_theta)` to the actor loss. The slide presents
this as a regularization example, without the worst-state bound, paired KL
constraints, or improvement-certificate details.

Lesson 4 begins by recalling the RL-as-VI objective. Four static slides connect
that familiar objective to DA-MDPs:

1. Recall the same action-marginal KL and replace the policy with a diffusion
   model. Its executed-action density is an integral over the denoising path,
   so the local log-policy term used in Lesson 3 is now intractable too.
2. Apply DPI again, showing both marginalization identities and the chain from
   action distributions to state-action trajectories to joint diffusion paths.
   `A` denotes executed actions, `S` the physical states, and purple `Z` the
   added diffusion variables. Forward augmentation preserves the target marginal.
3. Factorize the variational joint into the shared dynamics, prior, and blue
   reverse kernels. Factorize the target joint into its original trajectory
   target and orange forward reference kernels. Both sides retain `Z` explicitly.
4. Interpret the joint-KL objective as a DA-MDP. The improved `K=3` diagram comes
   after the derivation: a shaded band holds the physical state fixed, blue
   arrows denote local reverse-policy draws, and orange backward arrows denote
   forward reference kernels (not environment transitions). Each denoising
   transition receives a log-ratio bonus; only the final action earns the
   environment reward. Gray branches then show stochastic next states, fresh
   prior draws, and the reset to `k=3`.

The diagram slide gives the log-ratio bonus and the resulting discounted return.
The prior is fixed: its expected log-density contributes only a constant, as
does the original target normalizer. The earlier numbered augmentation and
full-path-objective slides are replaced by this sequence.

The DA-MDP critic and actor definitions share one static slide: soft Bellman
equations for `Q_DA^{theta_r}` and `V_DA^{theta_r}`, the phase-dependent discount,
the policy loss as an expected reverse/forward log ratio minus the frozen
critic, and gradient equivalence at the rollout parameters. This replaces the
separate critic and actor derivation slides. The old Step 4 and Step 5 slides
are replaced by one practical REPPO-to-DA:REPPO comparison slide. Aligned
columns pair the reparameterized samples, policy losses, and behavior-to-current
KL trust regions. The DA column uses the augmented critic and reverse/forward
log ratio. Its trust-region estimate is `K` times the local reverse-kernel KL
for a uniformly sampled diffusion phase, matching the chain-KL convention in
the paper's DA:REPPO appendix. A shared piecewise update uses the policy loss
inside the bound and the KL penalty outside. Two compact bullets retain the
general MaxEnt-RL-to-diffusion recipe (including DA: PPO) and memory efficiency.
The memory statement concerns activations for local actor updates and
minibatching diffusion steps; sampling and rollout storage still depend on
the number of diffusion steps.

The closing references include
[DIME (Celik et al., 2025)](https://arxiv.org/pdf/2502.02316) and
[TruDi (Le et al., 2026)](https://arxiv.org/abs/2606.15260), together with
Soft Actor-Critic and TRPO.
All references are visible together on one static slide, without reveal overlays.

The final thank-you slide invites collaboration on DA-MDPs and includes a
QR code for
[Sebastian Sanokowski's LinkedIn profile](https://www.linkedin.com/in/sebastian-sanokowski-0664451b7/).
The QR code is generated directly in TeX using the same `qrcode` package as
the research introduction.

The TUM and ATARI Lab marks used in the deck are stored under `assets/logos/`.
They were obtained from TUM's official logo portal and the official ATARI Lab
website, respectively.

Content slides are staged with Beamer overlays: advance once to reveal the next
bullet, equation, code alternative, or conclusion. The opening title and lesson
divider and closing frames remain static. The deck contains 51 teaching frames and 101 PDF
pages because each reveal state is stored as a PDF page. The embedded StackCube,
PushT, and training animations continue to play independently.

Lessons 1 and 2 use the reference GMM-40 geometry from Midgley et al.
(2023), with PyTorch seed 0, component means in `[-40, 40]^2`, and covariance
`I`. Their contour-and-sample styling follows arXiv:2502.06685. Lesson 2's
four-panel GIF compares reparameterization and log-derivative training, each
with and without Langevin preconditioning. Every landscape animation includes
the corresponding loss history and a vertical marker for the current frame.
Both lessons plot positive losses on logarithmic y axes. If an experiment's
objective reaches zero or becomes negative, the plotting code uses `symlog`
with a linear region between -1 and 1, preserving the original loss values.
The shared Lesson 2 helper applies the same convention to single-run animations.

## Build from the SummerSchool VS Code workspace

Open `/home/sebastian/code/SummerSchool` as the VS Code folder and install the
recommended **LaTeX Workshop** extension. The workspace settings select the
bundled Tectonic recipe automatically, so opening `Slides/main.tex` and saving
it rebuilds `Slides/main.pdf`.

You can also run **Terminal → Run Build Task** (or `Ctrl+Shift+B`) and select
`Build SummerSchool slides`. This task runs from the workspace root and does
not require `latexmk` or a full TeX Live installation.

The talk follows four lessons:

1. variational inference and its zero-temperature reward-maximization limit,
   using `L1-VariationalInference/`. Students implement both gradient estimators
   in this notebook, then vary `TEMPERATURE` (including `T = 0`);
2. variance-preserving diffusion samplers, the data-processing inequality,
   and a path-space KL upper bound in `L2-DiffusionSamplers/`;
3. reinforcement learning as variational inference over trajectories, starting
   with the reward-defined target and including the TRPO trust-region motivation
   for REPPO; and
4. diffusion-augmented MDPs following arXiv:2512.02019.
