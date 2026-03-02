import threading

import numpy as np


class ReplayY:

    def __init__(self, length, capacity=1000, seed=0):
        self.length = length
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        # Cyclic list of complete episodes; each slot is None or a dict of
        # arrays with shape (ep_len, *field_shape).
        self.eps = [None] * self.capacity
        self.write_pos = 0
        self.num_eps = 0  # episodes stored so far (capped at capacity)

        # Per-worker buffers that accumulate steps until episode ends.
        self.local = {}
        self.lock = threading.Lock()

    def __len__(self):
        # Count any stored episode (zero-padding handles episodes shorter than
        # self.length, so every non-None slot is valid).
        return sum(1 for ep in self.eps if ep is not None)

    def add(self, step, worker=0):
        step = {k: v for k, v in step.items() if not k.startswith('log_')}
        step = {k: np.asarray(v) for k, v in step.items()}
        if worker not in self.local:
            self.local[worker] = []
        self.local[worker].append(step)
        if step.get('is_last', False):
            self._push_episode(worker)

    def _push_episode(self, worker):
        episode = self.local.pop(worker, [])
        if not episode:
            return
        # Stack individual step dicts into a single episode dict of arrays.
        ep = {k: np.stack([s[k] for s in episode]) for k in episode[0]}
        ep_len = len(episode)
        with self.lock:
            self.eps[self.write_pos] = ep
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.num_eps = min(self.num_eps + 1, self.capacity)

    def sample(self, batch):
        L = self.length
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        if not valid:
            raise RuntimeError(
                "ReplayY.sample() called with no complete episodes.")

        sampled_eps = [
            valid[self.rng.integers(0, len(valid))] for _ in range(batch)
        ]
        ep_lens = [len(next(iter(ep.values()))) for ep in sampled_eps]
        max_len = max(ep_lens)
        if max_len > L:
            raise ValueError(
                f"ReplayY length={L} is smaller than sampled complete episode "
                f"length={max_len}. Increase batch_length to avoid truncation.")

        seqs = []
        for ep, ep_len in zip(sampled_eps, ep_lens):
            pad_len = L - ep_len
            item = {}
            for k, v in ep.items():
                arr = v
                if pad_len > 0:
                    pad = np.zeros((pad_len, *v.shape[1:]), dtype=v.dtype)
                    arr = np.concatenate([arr, pad], axis=0)
                item[k] = arr
            seqs.append(item)

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}

        # t_mask[i, t] = True iff step t is real (not padding).
        t_mask = np.zeros((batch, L), dtype=bool)
        for i, ep_len in enumerate(ep_lens):
            t_mask[i, :ep_len] = True
        data["t_mask"] = t_mask
        return data
