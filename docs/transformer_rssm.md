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

  tokens (B,T,E) ─► post_head ─► post_logit ─► sample stoch (B,T,S,K)
                                                 │
  action (B,T,A) ────────────────────────────────┤
                                                 ▼
                               cat(flat(stoch), action_norm)
                                                 ▼
                           inp_proj ─► causal Transformer ─► h (B,T,D)
                                                 │
                                 shift-right: h_prev = [0, h[:,:-1]]
                                                 │
                                                 ▼
                                          prior_head(h_prev)
                                                 ▼
                                            prior_logit

  State at position t: (stoch_t, h_prev_t)
  Feature vector:       cat(flat(stoch_t), h_prev_t)
```

`h_prev_t` is zeroed on reset positions.

## Three operational modes

### 1. Training: full-sequence causal attention

`TransformerRSSM.observe(tokens, action, reset)`:

1. `post_logit = _post_head(tokens)` and sample posterior `stoch`.
2. Build transformer inputs from `(stoch, action)`.
3. Run windowed-causal transformer (`_fwd`) over the full sequence, where each
   step attends to at most `window_size` recent steps (including itself).
4. Shift-right to get `h_prev`.
5. `prior_logit = _prior_head(h_prev)`.
6. Return detached trajectory KV tensors (`kv_k`, `kv_v`) with `window_size`
   prepended dummy zero slots for efficient imagination-start construction
   without replaying history.

Posterior is conditioned on `tokens` only. It does **not** take `h_prev`.

### 2. Imagination: KV-cache rollout (windowed)

Imagination no longer uses `_imag_mlp`.

Given current latent `(stoch_t, h_prev_t)` and action `a_t`:
1. `update_carry(carry, stoch_t, a_t)` runs one transformer step with KV cache.
2. New deterministic context is `h_t = carry['h_prev']`.
3. `prior_head(h_t)` predicts `stoch_{t+1}`.

Carry keeps only `window_size` past steps, matching inference memory behavior.
During training, starts are built from `observe()`-returned trajectory KV tensors
for contiguous `K` offsets (`B*K` parallel starts), so no extra history replay is
needed in `_cal_grad`. `window_size` dummy all-zero KV slots are prepended
before start-window extraction so imagination uses the same zero-cache style as
`initial()` in policy inference.

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
  window_size: 64
```

Usage:

```bash
python3 train.py model.compile=False batch_length=500
```

## Practical notes

- Full-sequence training still uses parallel causal attention.
- Training attention is windowed to `window_size`, matching inference and
  imagination context limits.
- In imagination/inference, dynamics are rolled with a bounded KV window to
  control memory.
- Training samples contiguous trajectory segments and never concatenates
  different trajectories inside one sampled sequence.
- RoPE uses a modern cached implementation with dynamic cache growth if
  positions exceed `rope_max_seq_len`.
- `post_head` and `prior_head` are now multi-layer MLP heads (Linear + RMSNorm
  + activation blocks), not single linear projections.
