# DA-MDP introduction notes

Target duration: approximately 5 minutes. The deck has no progressive reveal
overlays: each teaching slide is exactly one PDF page. Video sequences still
play as embedded frame animations in compatible PDF viewers.

## Slide 1 — Sebastian Sanokowski (0:00--0:20)

“My name is Sebastian Sanokowski. I am a postdoc at the Applied and
Theoretical Aspects of Robot Intelligence Lab, or ATARI. I study how neural
networks can learn through interaction, rather than only from fixed datasets.”

## Slide 2 — Robotics has many valid solutions (0:20--0:50)

“A robotics task rarely has only one correct solution. The same object can be
grasped from above or from the side. A trained policy should be able to retain
all valid solutions.”

## Slide 3 — What multimodality buys us (0:50--1:20)

“Keeping several solutions improves exploration because the agent can test
different strategies. It improves adaptability because an alternative remains
available when an obstacle blocks the usual motion. It can also improve the
world model: the model represents distinct possible futures instead of
pretending that the scene has only one continuation.”

## Slide 4 — Diffusion samplers (1:20--1:50)

“This is where diffusion samplers are useful. Like diffusion models, they turn
noise into a complex multimodal distribution. But they do not require a
dataset of correct solutions: a cost defines which samples are good, and the
sampler is optimized directly toward low-cost solutions.”

## Slide 5 — Diffusion-Augmented MDPs (1:50--2:30)

“To extend this idea to reinforcement learning, we make the denoising process
part of the MDP. Each denoising transition becomes one local RL step, so we
only need to store one step at a time. The construction can turn any
maximum-entropy RL method into a diffusion policy: PPO becomes DA:PPO, REPPO
becomes DA:REPPO, and WPO becomes DA:WPO. This is the main idea of our
Diffusion-Augmented MDPs paper.”

## Slide 6 — StackCube (2:30--2:55)

“Here DA:REPPO solves the symmetric StackCube task. The same policy represents
both valid stacking orders instead of selecting only one of them.”

## Slide 7 — Humanoid: data, RL fine-tuning, and real-world transfer (2:55--4:15)

“We first train a diffusion model on pickup motion data. Executing those
motions through the humanoid tracker still leaves failures: the left video
shows four selected failure cases alongside five successful pickups. We
fine-tune the diffusion policy with DA:PPO in simulation. The middle video
shows nine tasks sampled from the fine-tuned checkpoint. All pickups complete in this
gallery. A learned tracking controller then executes generated motions on the
real humanoid; the right panel plays two real-world clips consecutively.”

The left panel uses archived pretrained EMA rollouts (repeat 0), with four
deliberately selected failures: `sub10_largebox_086_b002` and
`sub7_largebox_047_b025` fail by falling; `sub16_largebox_046_b000` and
`sub3_largebox_030_b000` fail the object-height condition. The recorded motion
is preserved until termination. The subsequent fall is an illustrative
passive MuJoCo continuation with actuation disabled and velocity estimated
from the last two recorded poses; it is not a further learned-policy rollout.
The camera/grid match the middle panel, but the selected task entries differ.
The middle panel retains the original seed-91 gallery.
These are example rollouts, not an aggregate success-rate evaluation; 9/9
refers to completion, not the stricter final-goal-distance metric.

## Slide 8 — Tutorial and installation (4:15--4:50)

“The tutorial derives this story from the beginning: variational inference,
diffusion samplers, reinforcement learning as inference, and finally DA-MDPs.
There are executable exercises, so please clone the repository and run
`uv sync --locked` before the tutorial. The README contains the one-time `uv`
installation instructions.”

## Slide 9 — Thank you (4:50--5:00)

“Thank you for your attention. The repository contains the tutorial material
and installation instructions.”

After this slide, switch to `main.pdf`; its landscape title slide introduces
the detailed tutorial roadmap.
