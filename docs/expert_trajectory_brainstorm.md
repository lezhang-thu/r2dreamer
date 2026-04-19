# Brainstorm: Other Ways to Use a Single High-Quality Expert Trajectory

This note summarizes possible ways to incorporate a single expert trajectory
beyond the current "memory as attention reference" mechanism.

## Framing

The main constraint is that there is only one high-quality expert episode.
Methods that assume broad expert state coverage can overfit easily. In general,
the more robust choices treat the expert trajectory as:

- a guide
- a prior
- a sparse scaffold

rather than a full supervision source.

## Good options

### 1. Behavior cloning loss on expert segments

Add an auxiliary actor loss on expert replay samples:

```text
L_bc = -log pi(a_exp | rl_feat_exp)
```

Why it can work:

- easy to add
- gives the actor a direct supervised signal
- especially useful early in training

Main risk:

- with only one trajectory, plain BC can overfit and hurt exploration off the
  expert path

Better version:

- use a schedule, strong early and anneal later
- only apply BC on expert replay samples

### 2. Q-filter / advantage-filter imitation

Only apply BC when the expert action looks better than the policy action under
the current critic.

Why this is better than plain BC:

- avoids forcing the policy to copy expert behavior when the policy has already
  found something better
- reduces damage from distribution mismatch

For this codebase:

- this is a strong fit
- actor/value machinery already exists, so it is natural to add

### 3. Prioritized replay over expert segments

Instead of fixing exactly 50% expert segments forever, treat expert segments as
high-priority replay and anneal their weight over time.

Why:

- early training benefits from frequent expert exposure
- later training should rely more on self-generated experience

A practical schedule:

- start near `0.5`
- decay to `0.1` or `0.0`
- optionally keep a small floor such as `0.05`

This is often simpler and more stable than hard-wiring a large constant expert
fraction forever.

### 4. Progress-based reward shaping from the expert trajectory

Use the expert trajectory as a latent path and reward the agent for moving
forward along that path.

Example:

- encode the expert trajectory into RSSM latent states
- for the current latent state, find the nearest expert step or an
  attention-weighted position
- reward forward movement toward later expert states

Why it is attractive:

- does not require exact action imitation
- encourages the same high-level progress rather than exact control
- can work better when expert actions are brittle

Risk:

- nearest-neighbor matching in latent space can be noisy
- shaping must be designed carefully to avoid shortcut solutions

This is one of the most principled alternatives if the goal is to use the
expert to show where to go, not exactly which action to take.

### 5. Subgoal conditioning using future expert states

Instead of attending to the whole expert memory at every step, choose a future
expert latent state as a subgoal and condition the actor on it.

Conceptually:

- pick expert latent at index `k`
- actor gets `(current_latent, goal_latent)`
- train the policy to move toward that latent

Why this is good:

- turns the expert trajectory into a sequence of waypoints
- is more structured than raw attention over the full memory
- is easier to interpret than "attend over everything"

This is particularly attractive if the expert trajectory has clear stages.

### 6. Episodic control / kNN value from expert states

Use the expert trajectory as episodic memory for value estimation.

Example:

- compare the current latent to expert latent states
- if it is similar to an expert state, bias the value upward using the expert
  return-to-go from that point

Why:

- a single trajectory is often more reliable as a value prior than as an
  action prior
- it tells the agent that states like this can lead to high return

This can be cleaner than direct action imitation when multiple good actions
exist from similar states.

## More structural ideas

### 7. Expert-conditioned policy prior via KL regularization

Train a small expert prior policy on the expert trajectory, then regularize the
main actor toward it only when the current state is close to expert states.

Loss idea:

```text
L = L_rl + lambda * KL(pi || pi_expert_prior)
```

Why this is attractive:

- separates expert knowledge from the main actor
- expert influence can be gated or annealed
- avoids forcing the main policy to use raw memory directly

This is a strong option if a cleaner modular design is preferred over direct
memory attention.

### 8. Return-to-go conditioning

Treat the expert trajectory as an example of high-return behavior and condition
the policy on desired return or progress-to-go.

This is more useful if the design is moving toward a
trajectory-conditioned/return-conditioned policy, but with only one expert
episode it is less compelling than the simpler methods above.

### 9. Planning bias rather than policy bias

If planning over imagined rollouts is added later, use the expert trajectory to
bias candidate action sequences or latent plans instead of directly changing the
actor.

Why:

- lets the expert act like a planner prior
- keeps the reactive policy cleaner

This is more complex and probably not the first thing to try.

## What to try first

Given the current code and architecture, the most practical next ideas are:

1. expert replay with annealed memory fraction
2. auxiliary BC loss on expert samples
3. Q-filtered BC
4. progress/value shaping from expert latent states

If only one practical next step is chosen, a strong candidate is:

- keep expert replay sampling
- add a small BC loss on expert samples only
- anneal its weight over training
- optionally gate it by critic advantage later

This is low risk and easy to diagnose.

## Why this can be better than only attention

The current memory-attention mechanism helps the policy "know about" the expert
trajectory, but it does not directly say:

- which expert action was good
- which states are valuable
- how strongly to trust the expert

A small imitation or value-prior loss gives a clearer learning signal. The
attention mechanism can still remain, but it becomes one part of the solution
rather than the whole solution.

## A practical combined design

A strong combined version would be:

- replay includes expert segments with an annealed fraction
- actor/value use `rl_feat = [stoch, deter]`, while memory is used for shaping
- add BC loss on expert replay samples only
- optionally weight BC by a schedule or critic-based filter
- optionally add a value prior from expert return-to-go

This uses the expert in three complementary ways:

- as training data
- as context
- as supervision
