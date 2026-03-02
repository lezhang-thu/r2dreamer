# Transformer-based RSSM (TransformerRSSM)

This document describes `TransformerRSSM` in `rssm.py`, enabled by
`model.dyn_type=transformer`.

## Motivation

The GRU RSSM transition is sequential and uses `(prev_stoch, prev_action)` per
step. The transformer variant now follows the same transition-style input
semantics: it updates latent dynamics from `(stoch, action)`, where `stoch` is
posterior-sampled from observation tokens.

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
3. Run causal transformer (`_fwd`) over the full sequence.
4. Shift-right to get `h_prev`.
5. `prior_logit = _prior_head(h_prev)`.
6. Return detached trajectory KV tensors (`kv_k`, `kv_v`) and `pos_before`
   for efficient imagination-start construction without replaying history.

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
needed in `_cal_grad`.

### 3. Policy inference: two-phase KV-cache

- **Phase 1** (`get_feat_step`): posterior from current `tokens`, returns
  current `stoch` and `h_prev`.
- **Phase 2** (`update_carry`): update transformer context using
  `(stoch, action)` and KV cache.

## Integration in `dreamer.py`

Controlled by `model.dyn_type` (`"rssm"` or `"transformer"`):

| Aspect | GRU RSSM | TransformerRSSM |
|--------|----------|-----------------|
| Training transition input | `(prev_stoch, prev_action)` via recurrent step | `(post_stoch, action)` via transformer |
| `observe()` args | `(embed, action, initial, reset)` | `(tokens, action, reset)` |
| Posterior input | embed | tokens |
| Prior input | deter | `h_prev` |
| Inference carry | `{stoch, deter, prev_action}` | `{kv_cache, pos, h_prev}` |
| Imagination transition | GRU `img_step` | KV-cache `img_step_with_carry` + prior head |
| Extra alignment loss | N/A | removed |

## Configuration

In `configs/model/_base_.yaml`:

```yaml
dyn_type: "transformer"
imag_last: 64

transformer:
  stoch: ${model.rssm.stoch}
  deter: ${model.deter}
  discrete: ${model.discrete}
  unimix_ratio: ${model.rssm.unimix_ratio}
  act: ${model.act}
  head_hidden: ${model.hidden}
  post_layers: 1
  prior_layers: 2
  n_heads: 8
  n_layers: 4
  d_ff: 4096
  window_size: 128
```

Usage:

```bash
python3 train.py model.dyn_type=transformer model.compile=False batch_length=500
```

## Practical notes

- Full-sequence training still uses parallel causal attention.
- In imagination/inference, dynamics are rolled with a bounded KV window to
  control memory.
- `post_head` and `prior_head` are now multi-layer MLP heads (Linear + RMSNorm
  + activation blocks), not single linear projections.
