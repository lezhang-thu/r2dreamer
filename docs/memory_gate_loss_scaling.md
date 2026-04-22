# Memory Gate Loss Scaling

## Monitoring plan

With explicit source-label gate supervision, the default scales are:

```yaml
memory_use: 0.1
memory_sparse: 0.01
```

Watch the existing metrics in TensorBoard:

1. **`metrics/memory_use`** — gate value averaged over `is_memory=True`
   positions. Should rise to ≥ 0.7 within the first few thousand training
   steps. If it stalls near 0.5, increase `memory_use` before increasing
   sparsity.

2. **`metrics/memory_sparse`** — gate value averaged over `is_memory=False`
   positions. This is a weak source-label sparsity prior, not a semantic
   "not memory-like" classifier. Keep it much smaller than `memory_use` unless
   agent replay is known to be off the expert trajectory.
