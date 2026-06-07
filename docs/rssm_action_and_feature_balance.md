# RSSM Action and Feature Balance Notes

This note summarizes two Transformer RSSM input/feature balance concerns and
the corresponding implementation choices.

## 1. Action Conditioning in Transformer RSSM Tokens

The Transformer RSSM transition token is built from the stochastic state and
the current action using one concat projection:

```python
x = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))
```

For Atari, discrete actions are wrapped as one-hot vectors. The normalization:

```python
action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()
```

therefore leaves actions unchanged. The action part of `_inp_proj` is still not
a single scalar scale; for a one-hot action it is equivalent to selecting one
learned `deter`-dimensional action column inside the full concat projection.

The concern is relative branch strength. With `stoch=32`, the stochastic state
contains 32 active categorical entries, while the Atari action contributes one
active one-hot entry. This can make the action path weaker at initialization.
A separate plain linear action projection would not solve this, because affine
projections can be merged into the concat projection.

### Current Setting

The implementation intentionally uses the original concat projection for the
world-model token path, so action conditioning is handled by `_inp_proj` jointly
with `stoch_flat`. This keeps the action-input ablation simple and avoids adding
a dedicated nonlinear action branch.

## 2. Downstream Stochastic vs. Deterministic Feature Balance

RSSM features are used by actor, critic, reward, continuation, and the R2-Dreamer
projector. The raw feature is:

```python
feat = torch.cat([stoch_flat, deter], -1)
```

For `bash atari.sh`, `model=size-transformer` gives:

```text
deter = 512
stoch * discrete = 32 * 64 = 2048
raw_feat_size = 2560
```

Thus 80% of raw feature dimensions are stochastic categories and 20% are the
deterministic Transformer context. This differs from official DreamerV3 size200M,
which uses:

```text
deter = 8192
stoch * classes = 32 * 64 = 2048
```

There, the ratio is inverted: about 80% deterministic context and 20%
stochastic categories.

The raw dimension ratio is not the same as forward magnitude. `stoch_flat` is
sparse, with about 32 active one-hot entries, while `deter` is dense. Still, the
parameter allocation and downstream inductive bias are different: heads receive
many more raw stochastic columns than deterministic context columns.

Increasing Transformer `deter` to 2048 or 8192 is not practical here because it
would substantially increase attention, FFN, and KV-cache memory. Instead, the
downstream feature representation is adapted after the Transformer.

### Implemented Solution

Add an optional nonlinear deterministic context adapter. The deterministic
branch can be projected and reused by prior prediction and downstream heads.
The stochastic branch remains independently configurable; `null` leaves it raw.

```python
stoch = posterior_head(tokens)
h_prev = transformer_context(stoch, action)
deter_context = self._deter_feat(h_prev)
prior_logit = prior_head(deter_context)
feat = torch.cat([deter_context, stoch_flat], -1)
```

For projected branches, the transform is:

```python
Linear -> RMSNorm -> activation
```

This is not mergeable into only the first layer of the downstream heads because
the projected branch is shared by prior and downstream heads. It keeps the
Transformer state small while giving prediction heads a stronger deterministic
context route.

Base config leaves both branches raw:

```yaml
model:
  transformer:
    feat_deter_dim: null
    feat_stoch_dim: null
```

`model=size-transformer`, used by `bash atari.sh`, enables:

```yaml
model:
  transformer:
    feat_deter_dim: 8192
    feat_stoch_dim: null
```

So prior conditioning and downstream heads receive:

```text
deterministic context = 8192 projected deter
downstream feat_size = 8192 projected deter + 2048 raw stoch = 10240
```

This gives deterministic context a size200M-like width for prior/head
conditioning without increasing Transformer width, attention memory, or KV-cache
memory. It is still a context expansion of a 512-dimensional Transformer state,
not a true 8192-dimensional RSSM recurrent state.

## 3. Single-Pass Posterior

The current Transformer RSSM keeps posterior inference observation-only:

```text
z_t = posterior_head(obs_t)
```

The prior predicts this stochastic code from deterministic Transformer context:

```text
prior_head(context(h_prev_t)) -> z_t
```

The full training path is:

```text
z_t = posterior_head(obs_t)
h_prev_t = transformer_context(z, action)
prior_logit_t = prior_head(context(h_prev_t))
```

World-model heads use the same final state:

```text
reward/continue/actor/critic/projector use get_feat(z_t, h_prev_t)
imagination samples z from prior_head(context(h_prev)) and feeds z back into dynamics
```

The KL loss is now one-sided: `KL(stopgrad(posterior) || prior)`. This trains the
prior to predict observation-derived stochastic codes for imagination rollout
without the reverse representation KL.
