# RSSM Action and Feature Balance Notes

This note summarizes two Transformer RSSM input/feature balance concerns and
the corresponding implementation choices.

## 1. Action Conditioning in Transformer RSSM Tokens

The original Transformer RSSM transition token was built from the stochastic
state and the current action using one concat projection:

```python
x = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))
```

For Atari, discrete actions are wrapped as one-hot vectors. The normalization:

```python
action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()
```

therefore leaves actions unchanged. The action part of `_inp_proj` is not a
single scalar scale; for a one-hot action it is equivalent to selecting one
learned `deter`-dimensional action embedding column.

The concern is relative branch strength. With `stoch=32`, the stochastic state
contains 32 active categorical entries, while the Atari action contributes one
active one-hot entry. This can make the action path weaker at initialization.
A separate plain linear action projection would not solve this, because affine
projections can be merged into the concat projection.

### Implemented Solution

Use separate nonlinear stochastic and action input branches:

```python
stoch_token = self._stoch_in(stoch_flat)
action_token = self._action_in(action_norm)
x = stoch_token + action_token
```

Each branch is:

```python
Linear -> RMSNorm -> activation
```

This is intentionally not mergeable into a single affine concat projection
because each branch has its own normalization and activation. It gives actions a
dedicated nonlinear route into the Transformer token and also normalizes the
stochastic branch separately before the two branches are summed.

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

Add a nonlinear feature adapter inside `TransformerRSSM.get_feat()`:

```python
deter_feat = self._deter_feat(deter)
stoch_feat = self._stoch_feat(stoch_flat)
feat = torch.cat([deter_feat, stoch_feat], -1)
```

Each branch is:

```python
Linear -> RMSNorm -> activation
```

This is not mergeable into the first layer of the downstream heads because each
branch has separate normalization and nonlinearity. It keeps the Transformer
state small while giving downstream modules a more balanced representation.

Base config leaves the adapter disabled:

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
    feat_deter_dim: 1024
    feat_stoch_dim: 512
```

So downstream heads receive:

```text
adapted_feat_size = 1024 + 512 = 1536
```

This biases the downstream representation toward deterministic context without
increasing Transformer width or memory.
