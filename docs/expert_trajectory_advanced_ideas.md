# Advanced Ideas: Empowering the Agent with a Single Expert Trajectory

This document captures ideas that go beyond simple behavioral cloning and beyond
the current memory-attention mechanism (which attends over expert RSSM states
with an external MHA module). Preliminary results indicate plain BC does not
work, and the current memory-attention approach also underperforms.

## Diagnosis: Why Memory-Attention Is Weak

The current approach (`dreamer.py:_apply_memory_attention`) uses vanilla
multi-head attention where query = current `deter`, key/value = expert `deter`
sequence. This gives the policy a **bag-of-states** view — it can retrieve
similar-looking expert states but has no notion of:

- **Temporal ordering** — which expert state comes *next*
- **What made the expert trajectory good** — reward signal is not in the
  attention
- **Progress** — how far along the expert path the agent currently is

The attended vector is just a soft average over expert deters, and the policy
has to figure out what to do with that context purely from RL reward, which is
a very weak signal especially early on.

---

## Idea 1: World-Model Grounding — Expert as Dynamics Consistency Target

Instead of using the expert trajectory to guide the *policy*, use it to improve
the *world model itself*.

### Mechanism

Run the RSSM in imagination mode using expert actions, and penalize divergence
from the expert's actual encoded observations at each step:

```
L_dynamics_consistency = Σ_t ||h_t^imagined - h_t^posterior||²
```

where `h_t^imagined` comes from rolling out the prior with expert actions, and
`h_t^posterior` comes from encoding the actual expert observations.

### Why this is different from BC

This doesn't touch the actor at all. It makes the world model more accurate in
the region the expert visits, so imagined rollouts starting near the expert path
are more reliable, so the actor-critic can learn better from imagination in that
region.

### Why this could work with one trajectory

The world model already trains on expert replay segments (50% sampling). This
loss adds a *sequential consistency* signal that the standard
observation-reconstruction + KL losses don't provide. It specifically teaches
the dynamics model that "if you're in this state and take this action, you
should end up *there*."

### Implementation notes

- Compute posterior states from expert observations (already done via
  `refresh_memory_context`)
- Roll out the prior from expert step 0 using expert actions
- MSE between imagined `deter` and posterior `deter` at each step
- Add as an auxiliary loss to the world-model loss with a tunable scale
- Detach the posterior target to avoid double-counting with the KL loss

---

## Idea 2: Temporal Contrastive Localization — "Where Am I on the Expert Path?"

Replace the vanilla MHA with a mechanism that explicitly localizes the agent on
the expert trajectory.

### Mechanism

Train a small head that, given the current latent state, predicts a distribution
over *positions* along the expert trajectory:

```
p(τ = t | z_current) = softmax(sim(z_current, z_expert_t) / temperature)
```

Then, instead of attending over the full expert trajectory, the policy receives:

- The predicted position `t*` (or soft position)
- The expert state at `t* + k` for some lookahead `k` (a **waypoint**)

This gives the policy a structured "you are here, go there" signal rather than
an amorphous attention blob.

### Training the localizer

It is self-supervised. When expert replay segments are in the batch, the
ground-truth position is known. When agent-generated segments are in the batch,
the soft position is computed (it may be diffuse/uncertain, which is fine).

### Why this could work

It converts the expert trajectory from a retrieval source into a structured
navigation aid. The policy sees a concrete goal state rather than a diffuse
context.

---

## Idea 3: Expert Trajectory as Reward Model Prior (Latent GAIL)

Instead of hand-designing a reward shaping function, train an auxiliary
trajectory discriminator that distinguishes expert transitions from agent
transitions, and use its output as a learned intrinsic reward.

### Mechanism

```
D(feat_t, action_t) → [0, 1]   (expert vs agent)
r_intrinsic_t = log D(feat_t, action_t)
```

The discriminator is a small MLP trained on the mixed replay batches (already
50/50 expert/agent data). The intrinsic reward is added to the extrinsic reward
during imagination rollouts.

### Why it is not BC

The discriminator provides a dense reward signal in latent space. The actor is
free to find *different* actions that look expert-like to the discriminator.
With only one trajectory the discriminator will be biased, but that is actually
fine — it provides a state-visitation prior rather than an action prior.

### Key detail

Train the discriminator on *latent features*, not raw observations. This means
it generalizes through the world model's learned representation rather than
pixel-level matching.

### Implementation notes

- Small MLP: `feat_size + act_dim → hidden → 1` with sigmoid
- Binary cross-entropy loss on expert vs agent segments
- Add `log D(feat, action)` as intrinsic reward during imagination
- Anneal the intrinsic reward weight over training to avoid reward hacking

---

## Idea 4: Expert KV-Cache Injection — Conditioning the Transformer Dynamics Directly

Instead of a separate memory-attention module, **prepend the expert
trajectory's KV-cache to the agent's own KV-cache** during both training and
imagination.

### Mechanism

During `rssm.observe()`, when processing agent trajectories, prepend the
expert's cached keys and values as a prefix that the agent's transformer can
attend to. This is analogous to how prefix-tuning works in LLMs — the expert
trajectory becomes a "prompt" for the dynamics model.

```python
# In _fwd(), before attention:
K_full = cat([K_expert_prefix, K_agent], dim=2)
V_full = cat([V_expert_prefix, V_agent], dim=2)
# Mask: agent tokens can attend to expert prefix + causal self
```

### Why this could work

The transformer RSSM already has the right inductive bias — it processes
sequences with attention. By letting it directly attend to expert dynamics
tokens, it can implicitly learn "what happened in the expert trajectory after a
state like this." This is much richer than the external MHA approach because it
happens *inside* the dynamics model at every layer.

### Why it will not collapse to BC

The dynamics model predicts next-state distributions, not actions. The expert
prefix biases the state transition model toward the expert's state manifold, but
the actor still learns freely.

### Implementation notes

- Encode the expert trajectory once with the frozen encoder + RSSM
- Cache expert KV tensors (already available from `observe()` return)
- In `_fwd()`, concatenate expert KV prefix before agent KV at each layer
- Adjust the attention mask so agent queries can attend to the expert prefix
  (but the expert prefix is not updated and has no causal dependency on the
  agent)
- During KV-cache inference (`update_carry`), similarly prepend the expert
  prefix to the cache
- The expert KV prefix should be **detached** — no gradients flow back to the
  expert encoding

---

## Idea 5: Hindsight Relabeling — Expert Trajectory as Alternative Futures

During imagination rollouts, occasionally branch from the agent's current
imagined state and splice in the expert trajectory's future states as if the
agent had reached them.

### Mechanism

When an imagination rollout reaches a state that is "close enough" to some
expert state at position `t`, replace the remaining imagined trajectory with the
expert's actual future from `t` onward. This creates **optimistic value
targets** for states near the expert path.

```python
# During imagination, at each step:
dist_to_expert = min_t ||deter_imagined - deter_expert_t||
if dist_to_expert < threshold:
    # Use expert's future returns as value bootstrap
    value_target = expert_return_to_go[t]
```

### Why this is powerful with one trajectory

It does not require the agent to follow the expert — it just says "if you
happen to reach a state near the expert path, we know the future is good." This
is a value prior, not an action prior.

### Implementation notes

- Precompute expert return-to-go: `G_t = Σ_{k=t}^T γ^{k-t} r_k`
- During `_imagine()`, compute cosine similarity between imagined deters and
  expert deters
- When similarity exceeds a threshold, use expert `G_t` as a bootstrap value
- Can be implemented as a soft interpolation: `v_boot = α * G_expert + (1-α) *
  v_imagined` where `α` is based on similarity

---

## Recommended Priority

1. **Idea 4 (KV-Cache Injection)** — Most architecturally natural for this
   codebase. Minimal new machinery; replaces the external memory-attention with
   native cross-attention inside the transformer. The RSSM already has KV-cache
   infrastructure.

2. **Idea 1 (Dynamics Consistency)** — Easiest to implement and diagnose. Just
   one additional loss term on the world model. Directly addresses "is the world
   model accurate in the expert's region of state space?"

3. **Idea 3 (Latent GAIL)** — Well-understood theoretically (GAIL literature)
   and the 50/50 replay split provides natural training data. Moderate
   implementation effort.

4. **Idea 2 (Temporal Contrastive Localization)** — Requires a new head and
   training loop, but gives the most structured signal. Good if ideas 1 and 4
   are insufficient.

5. **Idea 5 (Hindsight Relabeling)** — Most complex to implement correctly;
   threshold tuning and value interpolation need care. Try after simpler ideas.
