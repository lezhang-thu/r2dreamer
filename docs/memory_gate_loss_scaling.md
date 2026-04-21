# Memory Gate Loss Scaling: Raising `memory_use` and `memory_sparse` to 1.0

## Monitoring plan

After switching to `memory_use: 1.0` and `memory_sparse: 1.0`, watch the
existing metrics in TensorBoard:

1. **`metrics/memory_use`** — gate value averaged over `is_memory=True`
   positions. Should rise to ≥ 0.7 within the first few thousand training
   steps. If it stalls near 0.5, the sparse-on-memory pressure during
   early-training low-confidence attention is too strong. Drop
   `memory_sparse` to `0.5`.

2. **`metrics/memory_confidence`** — mean `1 − H(weights) / log(n)` over all
   `t_mask` positions, where `H(weights)` is the Shannon entropy of the
   attention distribution. Should rise over training as attention aligns on
   memory-labeled data. If it stays near 0, attention is not sharpening —
   the problem is upstream (alignment or representation), not loss-scale
   tuning.

If both curves look healthy, the 1.0 / 1.0 configuration is valid for the
confidence-gated design.
