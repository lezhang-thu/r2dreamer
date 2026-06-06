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

Add an optional deterministic feature source inside `TransformerRSSM.get_feat()`.
The deterministic branch can either be a learned projection of the final
512-dimensional residual state or the actual FFN hidden activation from the last
Transformer block.

```python
ffn_hidden = activation(last_block_ff1(ffn_norm(x)))
feat = torch.cat([ffn_hidden_prev, stoch_flat], -1)
```

For the learned-projection branch, the transform remains:

```python
Linear -> RMSNorm -> activation
```

For the FFN-hidden branch, the feature is not recovered from the final
512-dimensional output. It is captured directly inside the last Transformer
block and carried as `ffn_prev` alongside the 512-dimensional `h_prev`. It is
used only as a downstream readout; attention KV memory, recurrence, and the
prior still use the 512-dimensional `h_prev`.

Base config leaves both branches raw and uses `h_prev` as the deterministic
feature source:

```yaml
model:
  transformer:
    feat_deter_source: deter
    feat_deter_dim: null
    feat_stoch_dim: null
```

`model=size-transformer`, used by `bash atari.sh`, enables:

```yaml
model:
  transformer:
    d_ff: 8192
    feat_deter_source: ffn_hidden
    feat_deter_dim: null
    feat_stoch_dim: null
```

So downstream heads receive:

```text
adapted_feat_size = 8192 last-block FFN hidden + 2048 raw stoch = 10240
```

This exposes the actual large intermediate FFN representation to reward,
continue, actor, critic, and projector heads without increasing attention width
or KV-cache memory. It is still a downstream readout: the RSSM temporal state
passed between steps remains 512-dimensional.

## 3. Two-Pass Posterior Refinement

Official DreamerV3 conditions the posterior stochastic state on both the current
encoder token and deterministic context:

```text
q(stoch_t | obs_t, deter_t)
```

The original Transformer RSSM path inferred posterior stochastic state from the
encoder token alone:

```text
z1_t = q(obs_t)
```

This keeps segment training parallel, but makes the stochastic state more like a
per-frame code and less like a belief-state correction informed by temporal
context.

Exact DreamerV3-style posterior conditioning would make Transformer training
sequential because `stoch_t` would depend on `h_prev_t`, while `h_prev_t` depends
on previous posterior stochastic states. The implemented compromise keeps
parallel training with two full-segment Transformer passes:

```text
z1_t = post1(obs_t)
h1_prev_t = proposal_transformer_context(z1, action)
z2_t = post2(obs_t, h1_prev_t)
h2_prev_t = final_transformer_context(z2, action)
```

The first pass is only a proposal path. The final world-model state is
`(z2, h2_prev)`.

Consequently, prior and downstream heads use the refined state:

```text
prior_head(h2_prev_t) is trained to match z2_t
reward/continue/actor/critic/projector use get_feat(z2_t, h2_prev_t)
imagination samples z from prior_head(h_prev) and feeds z back into dynamics
```

This avoids the inconsistent variant where the prior learns to predict `z2` but
the transition model is trained on `z1`.
