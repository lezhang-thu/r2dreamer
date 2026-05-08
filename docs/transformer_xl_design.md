# Transformer-XL Design for TransformerRSSM

This document summarizes the intended Transformer-XL-style memory design used
by the Atari entry point (`bash atari.sh`). The goal is to keep the policy
inference, world-model training, and imagination semantics aligned without
returning to full-episode `batch_length=5120` training.

## Core Configuration

Atari uses short trainable segments plus a detached memory cache:

```yaml
batch_length: 64

model:
  transformer:
    memory_size: 512
    segment_length: ${batch_length}
```

Inside `TransformerRSSM`, the literal KV-cache length is:

```text
window_size = memory_size + segment_length
```

With the current Atari defaults, the literal cache length is `512 + 64 = 576`.
The literal cache may contain unused zero slots, but replay samples do not
contain padding.

## Replay Semantics

`ReplayY` now samples streams of real episode tokens.

- Each sampled row returns exactly `batch_length` real steps.
- If an episode ends before the row is full, another episode is selected and
  the row continues immediately.
- `is_first=True` marks every new episode boundary, including boundaries inside
  a sampled segment.
- `position` stores the absolute position within the current episode and resets
  to `0` at every episode boundary.
- No replay-side padding is emitted.
- No replay-side memory prefix is emitted.
- `t_mask`, `seq_mask`, and `target_start` are not part of the sampled batch.

If an expert episode is available, `ReplayY` reserves exactly one sampled row
for the expert stream. The remaining rows come from regular replay episodes.
The legacy `expert_sample_frac` setting is kept only for configuration
compatibility.

## World-Model Training

`Dreamer` owns a persistent detached training carry, `self._train_carry`, with
the same structure as the inference carry:

```text
kv_cache: (B, n_layers, 2, memory_size + segment_length, deter)
pos:      (B,)
seg_pos:  (B,)
h_prev:   (B, deter)
```

For each update:

1. `ReplayY.sample()` returns one real segment per batch row.
2. `Dreamer._world_model_forward()` passes that segment plus the matching slice
   of `self._train_carry` into `rssm.observe(..., memory_carry=...)`.
3. `TransformerRSSM._fwd_segment_with_carry()` attends from current segment
   queries to detached memory keys plus causal current-segment keys.
4. Losses are computed only on real current-segment tokens.
5. `feat_dict["next_carry"]` is detached and stored back into
   `self._train_carry` after the optimizer step.

The training carry is not a replay prefix. It is a detached model-side cache
from the previous segment processed by the same stream row.

## Episode Boundary Masking

Since replay can concatenate episodes inside a segment, attention cannot rely
on a simple causal mask.

The segment forward builds attention masks from `is_first`:

- Tokens before an interior reset may attend to the previous memory cache.
- The reset token and all following tokens in that new episode cannot attend to
  memory from the previous episode.
- Current-segment attention is causal and restricted to tokens with the same
  episode count inside the sampled segment.
- The next carry keeps only the suffix belonging to the current episode by
  using the final `position` value as the usable memory length.

This makes complete-context semantics episode-local even when a batch row
contains multiple episode fragments.

## RoPE Position Semantics

The implementation keeps the existing RoPE encoding. It does not add
Transformer-XL relative positional encoding.

- Replay provides `position` for every real token.
- Training applies RoPE with those absolute per-episode positions.
- Acting keeps `pos` in the RSSM carry and increments it after each
  environment step.
- `_mask_carry()` resets `pos`, `seg_pos`, `kv_cache`, and `h_prev` on episode
  reset.
- Imagination starts inherit positions from the observed segment when available.

The invariant is that positions are episode-local absolute positions, not
global replay indices.

## Acting and Imagination

Acting and imagination use a fixed literal cache of
`memory_size + segment_length` slots, but the number of visible slots changes
with the current segment position:

```text
visible = min(pos + 1, memory_size + seg_pos + 1, window_size)
seg_pos = (seg_pos + 1) % segment_length
```

This gives Transformer-XL segment behavior:

- At the first token of a segment, the token can see memory plus itself.
- As the segment advances, it can see memory plus the current segment prefix.
- At the last token of a segment, it can see memory plus the whole segment.
- At the next segment boundary, visibility falls back to memory plus one token.

The literal cache length stays fixed, so the implementation can use one cache
shape everywhere. Unused slots are zero-filled and hidden by the attention mask.

`build_imag_starts()` constructs imagination carries from observed trajectory
KV tensors by gathering the fixed literal cache window immediately before each
sampled start. The sampled start's `seg_pos` is derived from
`start % segment_length`, so imagined rollouts follow the same segment-visible
window rule as policy inference.

## Optimizer Structure

The Transformer-XL design uses one optimizer over the world model and
actor-critic modules.

- `wm_accum_steps` controls world-model gradient accumulation over replay batch
  chunks.
- `ac_accum_steps` controls actor-critic gradient accumulation over imagination
  start chunks.
- `ac_updates_per_wm` is no longer part of the design.

Actor-critic training still starts from detached world-model outputs and frozen
RSSM/encoder copies for imagination, but it is performed as part of the same
overall update step.

## Main Invariants

- Replay batches contain real tokens only.
- Episode boundaries are explicit through `is_first`.
- `position` is the source of truth for RoPE positions during training.
- Fixed-size padded cache slots exist only inside RSSM carries, not in replay.
- Training, acting, and imagination all use the same literal cache shape.
- The actual visible attention context follows Transformer-XL segment semantics:
  memory plus the current segment prefix.
