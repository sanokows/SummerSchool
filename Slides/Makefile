.PHONY: all clean

VI_TRAINING_FRAMES := $(wildcard assets/l2_training/frame-*.png)
DIFFUSION_TRAINING_FRAMES := $(wildcard assets/l4_training/frame-*.png)
STACKCUBE_FRAMES := $(wildcard assets/stackcube_overlay/frame-*.png)
PUSHT_FRAMES := $(wildcard assets/pusht_multimodal/frame-*.jpg)
DIFFUSION_TRANSPORT_FRAMES := $(wildcard assets/diffusion_transport/frame-*.png)
VP_DIFFUSION_1D_FRAMES := $(wildcard assets/vp_diffusion_1d/frame-*.png)
MULTIMODAL_DA_FRAMES := $(wildcard assets/multimodal_agent_reppo_vs_da_reppo/frame-*.png)
HUMANOID_MOTION_A_FRAMES := $(wildcard assets/humanoid_motion_a/frame-*.jpg)
HUMANOID_MOTION_B_FRAMES := $(wildcard assets/humanoid_motion_b/frame-*.jpg)
HUMANOID_SIM_FRAMES := $(wildcard assets/humanoid_sim_comparison/frame-*.jpg)
HUMANOID_GALLERY_FRAMES := $(wildcard assets/humanoid_sim_gallery/frame-*.jpg)
HUMANOID_PRETRAINED_GALLERY_FRAMES := $(wildcard assets/humanoid_sim_pretrained_gallery/frame-*.jpg)

all: main.pdf intro.pdf

main.pdf: main.tex assets/logos/atari_lab.png assets/logos/tum_logo.png assets/l1_start.png assets/l2_end.png assets/kl_forward.pdf assets/kl_reverse.pdf assets/boltzmann_cost.pdf assets/boltzmann_temperatures.pdf $(STACKCUBE_FRAMES) $(PUSHT_FRAMES) $(VI_TRAINING_FRAMES) $(DIFFUSION_TRAINING_FRAMES) $(VP_DIFFUSION_1D_FRAMES) $(MULTIMODAL_DA_FRAMES)
	tectonic --keep-logs --synctex main.tex

intro.pdf: intro.tex assets/logos/atari_lab.png assets/logos/tum_logo.png $(STACKCUBE_FRAMES) $(DIFFUSION_TRANSPORT_FRAMES) $(HUMANOID_MOTION_A_FRAMES) $(HUMANOID_MOTION_B_FRAMES) $(HUMANOID_GALLERY_FRAMES) $(HUMANOID_PRETRAINED_GALLERY_FRAMES)
	tectonic --keep-logs --synctex intro.tex

clean:
	rm -f main.pdf main.log main.synctex.gz intro.pdf intro.log intro.synctex.gz
