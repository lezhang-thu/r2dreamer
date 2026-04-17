# Potential-Based Reward Shaping from Expert Memory

## The problem

The actor-critic learns entirely from imagined rollouts. In sparse-reward
environments, `imag_reward` is ~0 almost everywhere, so the actor gets no
useful gradient. The world model learns expert transition structure, but this
knowledge is detached from the actor (all imagination states are `.detach()`ed
before entering actor-critic).

## The fix

Add potential-based reward shaping (Ng et al., 1999) during imagination:

```
F(s, s') = γ_eff · Φ(s') - Φ(s)
```

where `Φ(s)` is the attention-weighted expert return-to-go retrieved at
imagined state `s`, and `γ_eff = imag_cont * disc` is the effective per-step
discount used by `_lambda_return`. The shaped reward becomes:

```
r_shaped = r_env + scale · (γ_eff · Φ(s') - Φ(s))
```

This improves the scalar advantage targets that weight the actor's policy
gradient. Note that `_imagine()` and `_lambda_return()` are both
`@torch.no_grad()`, so the shaping does not create a differentiable path from
Φ back through imagined states — it only changes the magnitude of the
REINFORCE-style advantage, biasing credit assignment toward expert-valued
regions.

### Why γ_eff, not disc alone

The return operator in `_lambda_return` discounts future values by
`imag_cont * disc`, not `disc` alone. `imag_cont` is the continuation
probability (< 1 near terminal states). For the shaping to telescope under the
return and preserve policy invariance, the coefficient on `Φ(s')` must match
the return's discount exactly. Using bare `disc` over-credits `Φ(s')` at
near-terminal states, breaking the telescoping sum.

## Why raw_rtg, not symlog(rtg)

Two reasons:

1. **Scale consistency.** `imag_reward` is in raw space (the reward head
   outputs `symexp(logits)` via `.mode()`). The shaping term must be in the
   same space. `raw_rtg` is raw; `rtg` is `symlog(raw_rtg)`. Adding symlog
   values to raw rewards is a scale mismatch.

2. **Preserving the theoretical guarantee.** Ng et al. proves that
   potential-based shaping does not change the optimal policy, but requires
   exactly `F = γ_eff·Φ(s') - Φ(s)` with the MDP's effective discount. The
   expert RTG satisfies `G_t = r_t + γ·G_{t+1}`, so `Φ = G` (raw) is a valid
   potential under discount `γ`. With `Φ = symlog(G)`, the nonlinearity breaks
   the relationship — `γ·symlog(G_{t+1}) ≠ symlog(γ·G_{t+1})` — so no single
   `γ` satisfies the theorem's form. The shaping becomes an arbitrary reward
   perturbation that may change the optimal policy.

## Config

```yaml
expert_shaping_scale: 0.1   # 0.0 disables shaping
```

## Implementation

In `_actor_critic_forward` of `dreamer.py`:

- Query expert memory with `imag_deter` (frozen) to retrieve `raw_rtg`
- Compute `gamma_eff = imag_cont[:, 1:] * disc`
- Compute `shaping = gamma_eff * phi[:, 1:] - phi[:, :-1]`
- Add to `imag_reward` at interior timesteps
- Log `metrics["shaping"]` for monitoring
