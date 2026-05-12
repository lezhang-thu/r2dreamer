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

Inside `TransformerRSSM`, every carry stores only previous-token memory:

```text
cache_size = memory_size
```

With the current Atari defaults, the carry length is `512`. `segment_length`
controls how many real tokens are processed in one parallel training segment;
it does not expand the policy or imagination KV cache.

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

If an expert episode is available, `expert_sample_frac` is the probability used
when a row needs a new stream. For example, `expert_sample_frac: 0.5` means
that each newly initialized row, or each exhausted row being replaced, has a
50% chance to use the expert episode when regular replay is also available.
The source is not re-sampled on every `ReplayY.sample()` call. A row keeps
consuming its current episode until that episode is exhausted. If a row switches
between replay and expert after exhaustion, the new stream starts at episode
offset `0`, so the RSSM carry is reset by `is_first`.

## World-Model Training

`Dreamer` owns a persistent detached training carry, `self._train_carry`.
Unlike the inference/imagination carry, this training carry only stores the
Transformer-XL memory tokens, because world-model training receives the full
current segment in parallel:

```text
kv_cache: (B, n_layers, 2, memory_size, deter)
pos:      (B,)
h_prev:   (B, deter)
```

For each update:

1. `ReplayY.sample()` returns one real segment per batch row.
2. `Dreamer._world_model_forward()` passes that segment plus the matching slice
   of `self._train_carry` into `rssm.observe(..., memory_carry=...)`.
3. `TransformerRSSM.observe()` attends from current segment queries to detached
   memory keys plus causal current-segment keys, with a sliding window of
   `memory_size` previous tokens plus the current token.
4. Losses are computed only on real current-segment tokens.
5. `feat_dict["next_carry"]` is detached and stored back into
   `self._train_carry` after the optimizer step.

The training carry is not a replay prefix. It is a detached model-side cache
from the previous segment processed by the same stream row. `rssm.observe()`
requires this carry; the old no-carry full-sequence training path is not part
of the current implementation.

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
- `_mask_carry()` resets `pos`, `kv_cache`, and `h_prev` on episode
  reset.
- Imagination starts inherit positions from the observed segment when available.

The invariant is that positions are episode-local absolute positions, not
global replay indices.

## Acting and Imagination

Acting and imagination use the same `memory_size` previous-token cache as
training:

```text
visible_previous = min(pos, memory_size)
attention_keys = visible_previous cached keys + current key
```

This gives one aligned sliding-window behavior:

- A carry contains only previous tokens, never current-segment slack slots.
- Each update appends the current key/value and keeps the newest
  `memory_size` entries for the next step.
- No `seg_pos` is tracked, so there is no segment-boundary visibility reset in
  policy inference or imagination.

Unused cache slots before an episode has enough history are zero-filled and
hidden by the attention mask.

`build_imag_starts()` constructs imagination carries from observed trajectory
KV tensors by gathering the previous `memory_size` keys immediately before each
sampled start. Episode-local positions hide gathered keys that would cross an
episode boundary inside a streamed replay segment.

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
- Acting, imagination, and world-model training use the same carry shape.
- World-model training uses a compact `memory_size` cache and reconstructs
  imagination caches by gathering the previous `memory_size` keys from compact
  memory plus current-segment KV tensors.
- The actual visible attention context is a sliding window of at most
  `memory_size` previous tokens plus the current token.
