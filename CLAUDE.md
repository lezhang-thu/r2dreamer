# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

R2-Dreamer is a PyTorch implementation of redundancy-reduced world models for reinforcement learning (ICLR 2026). It provides an efficient DreamerV3 reproduction (~5x faster than reference implementations) with pluggable representation learning objectives: `r2dreamer` (default, Barlow Twins-style), `dreamer` (standard reconstruction), `infonce` (contrastive), and `dreamerpro` (prototype-based).

## Commands

### Training
```bash
python3 train.py logdir=./logdir/test                          # default (DMC Vision, 12M params)
python3 train.py model.rep_loss=r2dreamer                      # switch algorithm
python3 train.py env=atari100k env.task=atari_pong             # switch environment
python3 train.py model=size50M                                 # switch model size
```

### Code Quality
```bash
pre-commit run --all-files    # black, flake8, isort, pyupgrade, codespell
```

### Monitoring
```bash
tensorboard --logdir ./logdir
```

## Architecture

**Entry point:** `train.py` (Hydra config) → `trainer.py` (`OnlineTrainer` loop) → `dreamer.py` (agent)

**Core components:**
- `dreamer.py` — Agent class with world model, actor-critic, and loss computation (`_cal_grad` method). Uses `TransformerRSSM` dynamics and supports frozen network clones for R2-Dreamer rep loss.
- `rssm.py` — Transformer-based RSSM with KV-cache dynamics. States are `(stoch, deter)` tuples where stoch is S×K categorical, deter is D-dimensional.
- `networks.py` — MultiEncoder (CNN+MLP), MultiDecoder, MLPHead, BlockLinear layers.
- `distributions.py` — OneHot, TwoHot distributions; symlog/symexp transforms.
- `replay_y.py` — Episode-based cyclic replay buffer (RAM-only, chunked sampling).
- `envs/` — ParallelEnv wrapper + environment-specific modules (DMC, Atari, Crafter, Meta-World, Memory Maze).
- `optim/` — LaProp optimizer and Adaptive Gradient Clipping.

**Config cascade:** `configs/configs.yaml` → `configs/env/*.yaml` → `configs/model/*.yaml` → CLI overrides

## Key Patterns

- **Tensor shape conventions:** B=batch, T=sequence, E=embedding, F=feature(S*K+D), S=stoch groups, K=categories, D=deter dim. See `docs/tensor_shapes.md`.
- **Carry state:** `carry_train` is retained as an API-compatible placeholder in the trainer loop.
- **Data masking:** `t_mask` (B,T) boolean mask handles padding in variable-length sequences; all losses are masked before reduction.
- **Mixed precision:** BFloat16 compute with Float32 parameters. Optional `torch.compile` for further speedup.

## Code Style

- Line length: 120 characters (black + flake8)
- Import sorting: isort with black profile
- Python 3.10+ syntax (pyupgrade enforced)
