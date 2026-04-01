# torch.compile Pitfalls and Solutions

## The Bitter Lesson: `mode="reduce-overhead"` Causes Periodic Hangs

### Problem Description

When using `torch.compile` with `mode="reduce-overhead"`, training exhibits periodic hangs:
- Training runs normally for some time
- Suddenly stops with **zero GPU utilization** for 5-30 minutes
- Eventually resumes, then hangs again at regular intervals
- GPU memory remains occupied but GPU-Util shows 0%

### Root Cause

`mode="reduce-overhead"` uses **CUDA graphs** for maximum performance:
1. CUDA graphs record GPU operations with **fixed memory addresses**
2. When tensor memory addresses change (due to fragmentation, garbage collection, or different allocation patterns), the graph becomes **invalid**
3. PyTorch must **re-record the entire CUDA graph**, which takes 5-30 minutes
4. During re-recording, GPU shows zero utilization (CPU is compiling)
5. This happens periodically as memory layout changes during training

### Why It Happens More with Long Sequences

With `batch_length=5120` (vs typical 64):
- Each batch transfers 60MB from CPU to GPU (vs 0.75MB)
- Larger memory allocations increase fragmentation
- More frequent memory layout changes trigger more re-recordings
- Variable episode lengths (padding) can cause allocation pattern changes

### Solution: Use `mode="default"`

**In `dreamer.py` line 149-150:**

```python
# BAD: Causes periodic hangs
self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

# GOOD: Stable performance without hangs
self._cal_grad = torch.compile(self._cal_grad, mode="default")
```

### Performance Trade-offs

| Mode | Per-step Speed | Stability | Use Case |
|------|---------------|-----------|----------|
| `"reduce-overhead"` | Fastest (when working) | ❌ Periodic 5-30min hangs | Short sequences, fixed memory patterns |
| `"default"` | 5-10% slower | ✅ No hangs | Long sequences, variable lengths, production |
| `"max-autotune"` | Varies | ✅ No hangs | Experimental, long compile time |

### How to Diagnose

**1. Monitor GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

**2. Enable verbose logging:**
```bash
export TORCH_LOGS="+dynamo,recompiles"
export TORCHDYNAMO_VERBOSE=1
python train.py ...
```

Look for messages like:
- `"[WARNING] Recompiling function..."`
- `"CUDA graph recording..."`

**3. Use the diagnostic script:**
```bash
python diagnose_hang.py  # monitors GPU util and detects hangs
```

### Other torch.compile Best Practices

**1. Pre-allocate buffers to avoid dynamic allocation:**
```python
# BAD: Creates tensors inside compiled function
def forward(self, x):
    mask = torch.tril(torch.ones(T, T, device=x.device))  # Dynamic allocation!
    return x * mask

# GOOD: Pre-allocate as buffer
def __init__(self):
    self.register_buffer('mask', torch.tril(torch.ones(T, T)))

def forward(self, x):
    return x * self.mask[:x.size(0), :x.size(0)]  # Slice pre-allocated buffer
```

**2. Avoid data-dependent control flow:**
```python
# BAD: Recompiles for different values
def forward(self, x):
    if x.sum() > 0:  # Data-dependent!
        return x * 2
    return x

# GOOD: Use tensor operations
def forward(self, x):
    return torch.where(x.sum() > 0, x * 2, x)
```

**3. Keep tensor shapes static:**
```python
# BAD: Variable shapes trigger recompilation
def forward(self, x):
    valid_len = int(mask.sum().item())  # Different each time!
    return x[:, :valid_len]  # Variable shape!

# GOOD: Keep shapes fixed, use masking
def forward(self, x, mask):
    return x * mask  # Same shape, different values
```

### Related Issues Fixed

1. **Attention mask allocation** - Pre-allocated `_attn_mask_cache` buffer (see `rssm.py` line ~340)
2. **Dummy KV tensors** - Pre-allocated `_dummy_k_cache` and `_dummy_v_cache` (see `rssm.py` line ~345)
3. **Window range tensor** - Pre-allocated `_window_range` buffer (see `rssm.py` line ~355)

These optimizations eliminate dynamic tensor creation inside compiled code, improving both performance and stability.

### References

- PyTorch torch.compile docs: https://pytorch.org/docs/stable/generated/torch.compile.html
- CUDA graphs: https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs
- Compilation modes: https://pytorch.org/get-started/pytorch-2.0/#user-experience

### Lessons Learned

1. **`mode="reduce-overhead"` is not always better** - The "overhead" it reduces comes at the cost of CUDA graph rigidity
2. **Long sequences amplify memory issues** - Larger allocations increase fragmentation and graph invalidation
3. **Zero GPU utilization ≠ idle** - Could be CPU-bound compilation/re-recording
4. **Pre-allocation is critical** - Avoid any dynamic tensor creation inside compiled functions
5. **When in doubt, use `mode="default"`** - It's the safe, stable choice for production

---

*Last updated: 2026-03-03*
*Issue discovered during TransformerRSSM training with batch_length=5120*
