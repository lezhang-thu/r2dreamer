# Memory Updates Summary

This note summarizes the main updates related to `memory`.

## 1. Agent-side memory reference

`train.py` loads a replay-style expert episode from `ge.json` and passes it
into the agent as `memory`.

Inside `Dreamer`:

- the memory episode is encoded with the frozen encoder and frozen
  `TransformerRSSM`
- its RSSM deterministic states are cached as detached reference
  representations
- the cache is refreshed after trainer update bursts so it stays aligned with
  the latest RSSM weights

## 2. Actor/value stay agent-state only

The actor and critic now use only the agent's own latent state:

```text
rl_feat = concat(flat(stoch), deter)
```

Reward and continue prediction remain unchanged and still use the original world
model feature:

```text
feat = concat(flat(stoch), deter)
```

So:

- reward/continue stay world-model based
- actor/value do not directly consume expert memory retrievals

## 3. Memory retrieval is used separately

Expert memory retrieval is still trained and used, but not as part of
`rl_feat`.

It is used for:

- expert-index alignment supervision
- expert-progress supervision
- potential-based reward shaping during imagination

## 4. No RSSM updates from actor-critic memory path

The memory RSSM representations are detached. During actor-critic training:

- `TransformerRSSM` is not updated through the memory-conditioning path
- the trainable part is the memory-attention module and its auxiliary heads

This keeps the world-model training path unchanged.

## 5. Inference uses the same agent-state policy input

During `act()`:

- the current latent state is inferred as before
- the actor uses `rl_feat = [flat(stoch), deter]`

So training-time and inference-time policy inputs are aligned.

## 6. Replay buffer now also receives memory

`train.py` now also passes `memory` into `ReplayY`.

`ReplayY` supports:

- an optional immutable memory episode
- a configurable sampling fraction `buffer.memory_sample_frac`

When memory exists, each sampled batch mixes:

- a fraction from the memory episode
- the rest from the ordinary replay buffer

If the live replay buffer is empty but memory exists, sampling falls back to
memory-only segments.

## 7. Default hyperparameter

The current default is:

```yaml
buffer.memory_sample_frac: 0.5
```

meaning half of each sampled replay batch comes from the expert memory episode
when both sources are available.
