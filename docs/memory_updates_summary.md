# Memory Updates Summary

This note summarizes the current usage of `memory`.

## 1. Dreamer no longer uses memory

`Dreamer` no longer receives or consumes a `memory` episode.

Actor and value use the standard latent RL feature again:

```text
rl_feat = concat(flat(stoch), deter)
```

There is no agent-side memory attention, no cached memory RSSM context, and no
trainer refresh hook for memory state.

## 2. Replay buffer still receives memory

`train.py` loads a replay-style expert episode from `ge.json` and passes it
into `ReplayY`.

`ReplayY` supports:

- an optional immutable memory episode
- a configurable sampling fraction `buffer.memory_sample_frac`

When memory exists, each sampled batch mixes:

- a fraction from the memory episode
- the rest from the ordinary replay buffer

If the live replay buffer is empty but memory exists, sampling falls back to
memory-only segments.

## 3. Default hyperparameter

The current default is:

```yaml
buffer.memory_sample_frac: 0.5
```

meaning half of each sampled replay batch comes from the expert memory episode
when both sources are available.
