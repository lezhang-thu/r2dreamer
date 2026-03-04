#!/bin/bash

# ==== Settings ====
GPU_ID=0
DATE=$(date +%m%d) # auto complete
SEED_START=0
#SEED_END=400
SEED_END=0
SEED_STEP=100
METHOD=r2dreamer

# Fixed training settings for TransformerRSSM with large window.
WINDOW_SIZE=5120
IMAG_LAST=2
AC_REPEATS=32
WM_ACCUM_STEPS=2
AC_ACCUM_STEPS=1

# ==== Tasks ====
tasks=(
	#    "atari_alien"
	#    "atari_amidar"
	#    "atari_assault"
	#    "atari_asterix"
	#    "atari_bank_heist"
	#    "atari_battle_zone"
	#    "atari_boxing"
	#    "atari_breakout"
	#    "atari_chopper_command"
	#    "atari_crazy_climber"
	#    "atari_demon_attack"
	#    "atari_freeway"
	#    "atari_frostbite"
	#    "atari_gopher"
	#    "atari_hero"
	#    "atari_jamesbond"
	#    "atari_kangaroo"
	#    "atari_krull"
	#    "atari_kung_fu_master"
	#    "atari_ms_pacman"
	"atari_pong"
	#    "atari_private_eye"
	#    "atari_qbert"
	#    "atari_road_runner"
	#    "atari_seaquest"
	#    "atari_up_n_down"
)

# ==== Loop ====
for task in "${tasks[@]}"; do
	for seed in $(seq $SEED_START $SEED_STEP $SEED_END); do
		run_logdir=logdir/5120-${DATE}_${METHOD}_${task#atari_}_$seed
		echo "[RUN] task=$task seed=$seed window_size=$WINDOW_SIZE imag_last=$IMAG_LAST ac_repeats=$AC_REPEATS wm_accum_steps=$WM_ACCUM_STEPS ac_accum_steps=$AC_ACCUM_STEPS"
		CUDA_VISIBLE_DEVICES=$GPU_ID python train.py \
			env=atari100k \
			env.task=$task \
			logdir=$run_logdir \
			model.compile=True \
			device=cuda:0 \
			buffer.storage_device=cpu \
			model=size12M \
			model.rep_loss=${METHOD} \
			model.enforce_full_history_window=True \
			model.transformer.window_size=$WINDOW_SIZE \
			model.imag_last=$IMAG_LAST \
			model.ac_repeats=$AC_REPEATS \
			model.wm_accum_steps=$WM_ACCUM_STEPS \
			model.ac_accum_steps=$AC_ACCUM_STEPS \
			seed=$seed \
			batch_size=2 batch_length=5120
	done
done
