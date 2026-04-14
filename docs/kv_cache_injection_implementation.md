# Expert KV-Cache Injection Implementation

This document describes the implementation of Idea #4 from `expert_trajectory_advanced_ideas.md`: injecting the expert trajectory's KV-cache directly into the transformer RSSM as a prefix.

## Overview

Instead of using an external memory-attention module that attends over expert `deter` states, we now **prepend the expert trajectory's KV-cache as a prefix** that the agent's transformer can attend to at every layer. This is analogous to prefix-tuning in LLMs — the expert trajectory becomes a "prompt" for the dynamics model.

## Key Changes

### 1. RSSM Layer (`rssm.py`)

**Modified methods:**

- `_fwd(x, return_kv=False, expert_kv_prefix=None)`: Accepts optional expert KV prefix. When provided, prepends expert keys/values to agent keys/values before attention. The attention mask allows all queries to attend to all expert positions plus causal agent positions.

- `observe(tokens, action, reset, sample=True, expert_kv_prefix=None)`: Passes expert KV prefix through to `_fwd`.

- `update_carry(carry, stoch, action, reset, expert_kv_prefix=None)`: During single-step inference/imagination, prepends expert KV to the sliding window cache before attention.

- `img_step_with_carry(stoch, carry, action, expert_kv_prefix=None)`: Passes expert KV prefix through to `update_carry`.

- `imagine_with_action(stoch, deter, actions, carry=None, expert_kv_prefix=None)`: Passes expert KV prefix through for multi-step imagination.

**Key implementation details:**

- Expert KV has shape `(L, T_exp, D)` where L=layers, T_exp=expert trajectory length, D=deter dimension.
- Expert KV is expanded to batch size and concatenated with agent KV: `K = cat([K_expert, K_agent], dim=seq)`.
- Expert KV already has RoPE applied during `refresh_memory_context()` (via the frozen RSSM's `observe()` → `_fwd()`), so RoPE is NOT re-applied when prepending.
- Attention mask is extended: `mask = cat([ones(T_query, T_exp), causal_mask(T_query, T_agent)], dim=1)`.

### 2. Dreamer Layer (`dreamer.py`)

**Modified methods:**

- `refresh_memory_context()`: Now stores expert KV-cache (`kv_k`, `kv_v`) in addition to `deter`.

- `_get_expert_kv_prefix(require_fresh=False)`: New helper method that extracts expert KV from memory context and strips the dummy W prefix slots (which are zero-padding from the `observe` method).

- `_world_model_forward(data)`: Gets expert KV prefix and passes it to `rssm.observe()` during training.

- `_imagine(start, imag_horizon, imag_carry=None)`: Gets expert KV prefix and passes it to `_frozen_rssm.img_step_with_carry()` during imagination rollouts.

- `act(obs, state, eval=False)`: Gets expert KV prefix and passes it to `_frozen_rssm.update_carry()` during policy inference.

**Removed components:**

- `memory_attention` module (external MHA) — no longer needed since attention happens inside the transformer.
- `_apply_memory_attention()` method — replaced by internal KV-cache injection.
- `rl_feat_size` reduced from `flat_stoch + 2*deter` to `flat_stoch + deter` (no more attended_deter).

### 3. Architecture Comparison

**Before (external memory attention):**

```
RSSM forward → deter
                ↓
         memory_attention(query=deter, kv=expert_deter)
                ↓
         attended_deter
                ↓
         rl_feat = [flat_stoch, deter, attended_deter]
                ↓
         actor/value
```

**After (KV-cache injection):**

```
RSSM forward with expert KV prefix prepended at every layer
    ↓
  deter (already contains expert information via cross-attention)
    ↓
  rl_feat = [flat_stoch, deter]
    ↓
  actor/value
```

## Why This Should Work Better

1. **Richer context**: The expert trajectory is visible at every transformer layer, not just as a post-hoc attention step. The dynamics model can learn to use expert information when predicting next states.

2. **Temporal structure preserved**: The expert KV-cache maintains the sequential structure of the expert trajectory. The transformer can attend to "what happened next" in the expert run, not just "similar states."

3. **Architecturally natural**: The transformer RSSM already has the right inductive bias for processing sequences with attention. Prepending expert KV is a minimal, principled extension.

4. **No action supervision**: Unlike behavioral cloning, this doesn't force the actor to copy expert actions. It biases the *dynamics model* toward the expert's state manifold, but the actor learns freely via RL.

## Configuration

No new hyperparameters are introduced. The existing `buffer.memory_sample_frac` controls how often expert segments appear in training batches.

## Testing

To verify the implementation works:

1. Check that training runs without errors.
2. Monitor world-model losses (dyn, rep) — they should be similar or better than baseline.
3. Monitor actor-critic losses — they should converge.
4. Check evaluation performance — the agent should learn faster or reach higher scores than the external memory-attention baseline.

## Future Extensions

- **Learnable expert KV projection**: Instead of using frozen expert KV directly, project it through a learnable layer to allow the model to "reinterpret" the expert trajectory.

- **Selective expert attention**: Add a gating mechanism that lets the model decide when to attend to the expert vs. rely on its own experience.

- **Multi-expert support**: Extend to multiple expert trajectories by concatenating their KV-caches.
