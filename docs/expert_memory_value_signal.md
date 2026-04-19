# Letting the Agent Know the Expert Trajectory Is Good

This note summarizes the key idea behind the current memory-attention design:
the agent should not only *see* the expert trajectory, it should also receive a
clear signal that states along that trajectory lead to high return.

## The core problem

Plain memory attention over expert latent states is too weak.

If the policy only attends over expert `deter` states, it gets a soft
"bag-of-states" summary:

- it may retrieve similar-looking expert states
- but it does not know which expert position it matched
- it does not know what happened next
- it does not know why that matched state was good

In particular, the reward signal is missing from the retrieval mechanism. The
agent is then forced to infer from sparse RL reward that "this attended memory
chunk is valuable", which is usually too weak.

## What the agent actually needs to know

The expert memory should represent an aligned sequence of expert tuples, not
just a sequence of latent states.

At expert step `t`, the useful memory entry is:

```text
m_t = (z_t, a_t, r_t, G_t, p_t, z_{t+k})
```

where:

- `z_t`: expert latent state at position `t`
- `a_t`: expert action at that position
- `r_t`: immediate reward
- `G_t`: discounted return-to-go from that position
- `p_t`: normalized progress along the trajectory
- `z_{t+k}`: a future waypoint latent

This tells the agent not only "a state like this existed", but also:

- what the expert did there
- how good that point was
- how far along the expert path it is
- where the expert trajectory goes next

## The key design principle

Use **one attention distribution over expert positions**, then read out all
expert fields from that same position.

Conceptually:

```text
alpha_t = attention(current_state, expert_position_t)
retrieved = sum_t alpha_t * (z_t, a_t, r_t, G_t, p_t, z_{t+k})
```

This is what makes the retrieved `(state, action, reward)` tuple meaningful.
The alignment comes from sharing the same attention weights across all expert
fields.

Without this, the model may receive expert state information and reward
information, but it still does not know that they correspond to the same expert
time step.

In practice, the retrieval also needs an explicit abstention mechanism. If the
current state is not meaningfully related to the expert trajectory, the model
should be able to keep a separate memory-use gate closed rather than being
forced to average over unrelated expert positions. This separates:

- which expert position matches best
- whether expert memory should be used at all

## Why return-to-go matters more than raw reward

For sparse-reward problems, immediate reward `r_t` is often too weak by itself.

What we really want to tell the agent is:

> if you reach a state like this, the future from here can be very good

That is a value signal, and `G_t` is the right signal for it.

So:

- `r_t` is local transition feedback
- `G_t` is the real "this part of the trajectory is good" signal

This is why expert return-to-go is more useful than simply concatenating raw
rewards into attention.

## What training signals are still needed

Attention alone is not enough. The model still needs explicit learning signals
that tell it how expert memory should be used.

The most important ones are:

### 1. Localization / alignment supervision

When the replay sample comes from the expert memory, the true expert index is
known. Use that to train the model to attend to the correct expert position.

This teaches:

- "this current latent corresponds to expert position `t`"
- "the retrieved tuple should come from that same position"

### 1b. Memory-use supervision and sparsity

The abstention gate should open on expert-memory states, but default toward
closed elsewhere unless using expert memory clearly helps.

This teaches:

- "expert states should use expert memory"
- "ordinary states should abstain unless there is strong evidence to retrieve"

### 2. Progress supervision

Train a small head to predict normalized trajectory progress `p_t`.

This gives the model an explicit notion of:

- where it is on the expert path
- whether it is moving forward

### 3. Potential-based shaping during imagination

Do not train the shared critic directly to predict expert return-to-go `G_t`.
That target belongs to the expert policy, not necessarily to the agent's
current policy.

Instead, use the retrieved expert return-to-go as a potential during imagined
rollouts:

```text
F(s, s') = γ_eff · Φ(s') - Φ(s)
```

with `Φ(s)` taken from retrieved expert `G_t`.

This teaches:

- "moving toward expert-valued regions should increase return"
- without forcing the critic to equal expert return on those states
- while keeping the critic target tied to the current policy's imagined return

## Why this is better than pure behavior cloning

The goal is not to say:

```text
copy the expert action exactly
```

The goal is closer to:

```text
recognize that this region of state space is good,
know where it is on that good trajectory,
and use expert action/waypoint information as guidance
```

This is more robust when:

- multiple actions can be good
- expert actions are brittle
- the agent reaches only approximate matches to expert states

## Current implementation direction on this branch

The current memory design follows this logic:

### Replay metadata

Replay marks expert samples explicitly with:

- `is_memory`
- `memory_index`

This makes expert-only supervision possible.

### Cached expert memory fields

The memory context stores:

- expert latent state
- expert stochastic feature
- expert action
- expert reward
- expert discounted return-to-go
- expert progress
- expert future waypoint
- a learned null memory slot for abstention

### Retrieved RL context

The actor and critic consume an RL feature that includes:

- current latent
- retrieved expert token
- retrieved waypoint
- retrieved expert action
- retrieved reward / return-to-go / progress

### Auxiliary losses

Two expert-only auxiliary losses are added:

- `memory_align`: attend to the correct expert index
- `memory_progress`: predict where we are on the expert path

In addition, actor-critic imagination uses potential-based reward shaping from
retrieved expert `raw_rtg` rather than a direct expert-value regression loss.

## Practical takeaway

If the question is:

```text
How do we let the agent know that the provided trajectory is good?
```

the answer is:

1. do not give only expert states
2. give aligned expert tuples
3. include return-to-go, not just reward
4. supervise which expert position is being matched
5. use return-to-go as a shaping potential, not as a direct critic target

That combination is what turns expert memory from a vague retrieval mechanism
into a usable progress-aware guide with a value-informed shaping signal.
