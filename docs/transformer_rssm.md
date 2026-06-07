# Transformer-based RSSM (TransformerRSSM)

This document describes `TransformerRSSM` in `rssm.py`, which is now the
default and only dynamics model used in `dreamer.py`.

## Motivation

`TransformerRSSM` updates latent dynamics from `(stoch, action)`, where
`stoch` is posterior-sampled from observation tokens.

Key change from earlier versions:
- Transition input is now `(stoch, action)`, not `(tokens, action)`.
- `_imag_mlp` alignment core is removed.
- Imagination uses transformer KV-cache transition + `_prior_head`.

## Architecture overview

```text
Training (observe path):

  tokens ─► post_head ─► z1 ─► Transformer ─► h1_prev
  tokens + context(h1_prev) ─► refine_post_head ─► z2 ─► Transformer ─► h2_prev
  tokens + context(h2_prev) ─► refine_post_head ─► z3 ─► Transformer ─► h3_prev

  prior_logit = prior_head(context(h2_prev))

  KL state at position t:    (z3_t, h2_prev_t)
  Feature state at position t: (z3_t, h3_prev_t)
```

All shifted deterministic contexts are zeroed on reset positions.

## Three operational modes

### 1. Training: segment sliding-window attention

`TransformerRSSM.observe(tokens, action, reset)`:

1. `proposal_logit = _post_head(tokens)` and sample `z1`.
2. Run the Transformer on `(z1, action)` and shift-right to get `h1_prev`.
3. `refine_logit = _refine_post_head(tokens, context(h1_prev))` and sample `z2`.
4. Run the Transformer on `(z2, action)` and shift-right to get `h2_prev`.
5. `post_logit = _refine_post_head(tokens, context(h2_prev))` and sample `z3`.
6. `prior_logit = _prior_head(context(h2_prev))`.
7. Run the final Transformer on `(z3, action)` and shift-right to get `h3_prev`.
8. Return detached final-pass trajectory KV tensors (`kv_k`, `kv_v`) with `memory_size`
   memory slots followed by current-segment keys for efficient
   imagination-start construction without replaying history.

The KL compares `post_logit` and `prior_logit`, both conditioned on `h2_prev`.
Reward, continuation, actor, critic, and projector features use `(z3, h3_prev)`.

### 2. Imagination: KV-cache rollout (windowed)

Imagination no longer uses `_imag_mlp`.

Given current latent `(stoch_t, h_prev_t)` and action `a_t`:
1. `update_carry(carry, stoch_t, a_t)` runs one transformer step with KV cache.
2. New deterministic context is `h_t = carry['h_prev']`.
3. `prior_head(h_t)` predicts `stoch_{t+1}`.

Carry keeps only `memory_size` previous steps, matching inference memory
behavior.
During training, starts are built from `observe()`-returned trajectory KV tensors
for `B*K` parallel starts with two sampling modes:
- if an episode has at least `K` valid steps, one contiguous block of `K`
  offsets is sampled;
- if an episode is shorter than `K`, `K` valid indices are sampled
  independently with replacement, so starts may repeat or be out of order.

`build_imag_starts()` handles both cases by gathering latent state and the
previous `memory_size` KV entries independently for each sampled start, so no
extra history replay is needed in `_cal_grad`.

### 3. Policy inference: two-phase KV-cache

- **Phase 1** (`get_feat_step`): posterior from current `tokens`, returns
  current `stoch` and `h_prev`.
- **Phase 2** (`update_carry`): update transformer context using
  `(stoch, action)` and KV cache.

## Integration in `dreamer.py`

`Dreamer` always instantiates `rssm.TransformerRSSM` and uses:
- Training observe path: `observe(tokens, action, reset)`
- Inference carry: `{kv_cache, pos, h_prev}`
- Imagination transition: `img_step_with_carry` + prior head

## Configuration

In `configs/model/_base_.yaml`:

```yaml
imag_last: 64

transformer:
  stoch: ${model.rssm.stoch}
  deter: ${model.deter}
  discrete: ${model.discrete}
  unimix_ratio: ${model.rssm.unimix_ratio}
  act: ${model.act}
  rope_base: 1000000.0
  rope_max_seq_len: 32768
  head_hidden: ${model.hidden}
  post_layers: 1
  prior_layers: 2
  n_heads: 8
  n_layers: 4
  d_ff: 4096
  memory_size: 64
  segment_length: ${batch_length}
```

Usage:

```bash
python3 train.py model.compile=False batch_length=500
```

## Practical notes

- Segment training still uses parallel causal attention.
- Training attention is windowed to `memory_size`, matching inference and
  imagination context limits.
- In imagination/inference, dynamics are rolled with a bounded KV window to
  control memory.
- `ReplayY` streams real episode chunks and may concatenate episode fragments
  inside a sampled segment; `is_first` and `position` define the boundaries.
- RoPE uses a modern cached implementation with dynamic cache growth if
  positions exceed `rope_max_seq_len`.
- `post_head` and `prior_head` are now multi-layer MLP heads (Linear + RMSNorm
  + activation blocks), not single linear projections.
