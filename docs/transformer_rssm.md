# Transformer-based RSSM (TransformerRSSM)

This document describes the `TransformerRSSM` class in `rssm.py`, which replaces the GRU-based recurrence with causal Transformer attention. It is activated by setting `model.dyn_type=transformer`.

## Motivation

The standard RSSM uses a block-GRU for sequential state transitions, processing one timestep at a time during training. The Transformer variant processes the entire episode in parallel via causal self-attention, enabling better long-range credit assignment while maintaining a Markovian MLP for fast imagination rollouts.

## Architecture overview

```
Training (_observe_seq path):

  tokens (B,T,E) ──┐
                    ├─ cat ─► inp_proj ─► Causal Transformer ─► h (B,T,D)
  action (B,T,A) ──┘              │                                │
                                  │                    shift-right: h_prev = [0, h[:,:-1]]
                                  │                                │
                                  │          ┌─────────────────────┤
                                  │          │                     │
                              posterior    prior              alignment
                          cat(h_prev, tok)  h_prev       _imag_core(sg(h_prev),
                              │               │           sg(stoch), action)
                              ▼               ▼               ▼
                          post_logit     prior_logit    MSE(sg(h), pred)
                              │
                              ▼
                         stoch (sample)

  State at position t: (h_prev = h_{t-1}, stoch_t)
  Feature vector:       cat(flat(stoch_t), h_prev_t)
```

## Three operational modes

### 1. Training: full-sequence causal attention

`TransformerRSSM.observe(tokens, action, reset)` processes the entire episode at once:

1. Concatenate `(tokens_t, action_t)` and project to `d_model`
2. Run pre-norm causal Transformer with RoPE (`_fwd`)
3. Shift output right: `h_prev[t] = h[t-1]` (zero at `t=0` and resets)
4. Posterior: `_post_head(cat(h_prev, tokens))` &rarr; sample `stoch`
5. Prior: `_prior_head(h_prev)`
6. Alignment: `_imag_core(sg(h_prev), sg(stoch), action)` predicts `h`; MSE loss trains only the imagination MLP

**Action alignment**: each position pairs `(obs_t, a_t)` &mdash; the *current* action, not `a_{t-1}`. This differs from the GRU RSSM which uses shifted `prev_action`.

### 2. Imagination: Markovian MLP

`img_step(stoch, deter, action)` uses the `_imag_core` MLP:

```
cat(deter, flat(stoch), action_norm) ─► [Linear ─► RMSNorm ─► SiLU] x N ─► next_deter
next_deter ─► _prior_head ─► sample next_stoch
```

This has the same interface as the GRU RSSM's `img_step`, so `_imagine()` in `dreamer.py` works unchanged.

### 3. Policy inference: KV-cache

Two-phase per-step inference avoids recomputing the full sequence:

- **Phase 1** (`get_feat_step`): Compute posterior from cached `h_prev` + current `tokens`. No action needed yet.
- **Phase 2** (`update_carry`): After action selection, run one Transformer step with per-layer KV-cache (sliding window of size `W`). Updates `h_prev` for the next step.

The KV-cache stores RoPE-rotated keys at their absolute positions, preserving correct relative position encoding.

## Integration in `dreamer.py`

Controlled by `model.dyn_type` (`"rssm"` or `"transformer"`):

| Aspect | GRU RSSM | TransformerRSSM |
|--------|----------|-----------------|
| Training action | `prev_action` (shifted) | `action` (current) |
| `observe()` args | `(embed, action, initial, reset)` | `(tokens, action, reset)` |
| Carry state | `(stoch, deter, prev_action)` tuple | Not used (complete episodes) |
| Inference state | `{stoch, deter, prev_action}` | `{kv_cache, pos, h_prev}` |
| Imagination start | All `B*T` positions | `K` contiguous positions (random offset) |
| Extra loss | &mdash; | `align` (imag_core alignment) |

The `_imagine()` and `clone_and_freeze()` methods require no changes since `TransformerRSSM` implements the same `get_feat()` / `img_step()` interface.

## Configuration

In `configs/model/_base_.yaml`:

```yaml
dyn_type: "rssm"      # "rssm" or "transformer"
imag_last: 0          # 0 = all T positions; >0 = K contiguous positions

transformer:
  stoch: 32           # stochastic groups
  deter: 2048         # d_model (= hidden dim)
  discrete: 16        # categories per group
  n_heads: 8
  n_layers: 4
  d_ff: 4096          # FFN inner dimension
  imag_layers: 3      # MLP depth for imagination
  window_size: 128    # KV-cache sliding window

loss_scales:
  align: 1.0          # imag_core alignment loss weight
```

Usage:

```bash
python3 train.py model.dyn_type=transformer model.compile=False batch_length=500
```

## Key assumptions

- **Complete episodes**: `batch_length >= max_episode_length`. No chunked replay or carry state propagation needed.
- **Static shapes**: All tensor shapes are deterministic for `torch.compile` compatibility (when `imag_last=0`).
