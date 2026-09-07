.PHONY: all clean

L1_TRAINING_FRAMES := $(wildcard assets/l1_training/frame-*.png)
L4_TRAINING_FRAMES := $(wildcard assets/l4_training/frame-*.png)
STACKCUBE_FRAMES := $(wildcard assets/stackcube_overlay/frame-*.png)
DIFFUSION_TRANSPORT_FRAMES := $(wildcard assets/diffusion_transport/frame-*.png)
MULTIMODAL_DA_FRAMES := $(wildcard assets/multimodal_agent_reppo_vs_da_reppo/frame-*.png)
HUMANOID_MOTION_A_FRAMES := $(wildcard assets/humanoid_motion_a/frame-*.jpg)
HUMANOID_MOTION_B_FRAMES := $(wildcard assets/humanoid_motion_b/frame-*.jpg)
HUMANOID_SIM_FRAMES := $(wildcard assets/humanoid_sim_comparison/frame-*.jpg)
HUMANOID_GALLERY_FRAMES := $(wildcard assets/humanoid_sim_gallery/frame-*.jpg)
HUMANOID_PRETRAINED_GALLERY_FRAMES := $(wildcard assets/humanoid_sim_pretrained_gallery/frame-*.jpg)

all: main.pdf intro.pdf

main.pdf: main.tex assets/logos/atari_lab.png assets/logos/tum_logo.png assets/l1_start.png assets/l2_end.png $(STACKCUBE_FRAMES) $(L1_TRAINING_FRAMES) $(L4_TRAINING_FRAMES) $(DIFFUSION_TRANSPORT_FRAMES) $(MULTIMODAL_DA_FRAMES)
	tectonic --keep-logs --synctex main.tex

intro.pdf: intro.tex assets/logos/atari_lab.png assets/logos/tum_logo.png $(STACKCUBE_FRAMES) $(DIFFUSION_TRANSPORT_FRAMES) $(HUMANOID_MOTION_A_FRAMES) $(HUMANOID_MOTION_B_FRAMES) $(HUMANOID_GALLERY_FRAMES) $(HUMANOID_PRETRAINED_GALLERY_FRAMES)
	tectonic --keep-logs --synctex intro.tex

clean:
	rm -f main.pdf main.log main.synctex.gz intro.pdf intro.log intro.synctex.gz
