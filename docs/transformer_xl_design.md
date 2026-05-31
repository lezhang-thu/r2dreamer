# Transformer-XL Design for TransformerRSSM

This document summarizes the intended Transformer-XL-style memory design used
by the Atari entry point (`bash atari.sh`). The goal is to keep the policy
inference, world-model training, and imagination semantics aligned without
returning to full-episode `batch_length=5120` training.

## Core Configuration

```yaml
batch_length: 64

model:
  transformer:
    memory_size: 512
```

Inside `TransformerRSSM`, every carry stores only previous-token memory:

```text
cache_size = memory_size
```

## Replay Semantics

`Replay` now samples streams of real episode tokens.

- Each sampled row returns exactly `batch_length` real steps.
- If an episode ends before the row is full, another episode is selected and
  the row continues immediately.
- `is_first=True` marks every new episode boundary, including boundaries inside
  a sampled segment.
- `position` stores the logical position within the shifted stream and resets
  to `0` at every shifted episode boundary.

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

1. `Replay.sample()` returns one real segment per batch row.
2. `Dreamer._world_model_forward()` passes that segment plus the matching slice
   of `self._train_carry` into `rssm.observe(..., memory_carry=...)`.
3. `TransformerRSSM.observe()` attends from current segment queries to detached
   memory keys plus causal current-segment keys, with a sliding window of
   `memory_size` previous tokens plus the current token.
4. `feat_dict["next_carry"]` is detached and stored back into
   `self._train_carry` after the optimizer step.

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
- Training applies RoPE with those shifted-stream positions.
- Acting keeps `pos` in the RSSM carry and increments it after each
  environment step.
- `_mask_carry()` resets `pos`, `kv_cache`, and `h_prev` on episode
  reset.
- Imagination starts inherit positions from the observed segment when available.

The invariant is that positions are stream-local positions, not global replay
indices.

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

Unused cache slots before an episode has enough history are zero-filled and
hidden by the attention mask.

`build_imag_starts()` constructs imagination carries from observed trajectory
KV tensors by gathering the previous `memory_size` keys immediately before each
sampled start. Episode-local positions hide gathered keys that would cross an
episode boundary inside a streamed replay segment.
