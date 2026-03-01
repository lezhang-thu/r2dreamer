import math
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

        # Stateful chunked-sampling state.  The first call to sample() draws a
        # fresh batch of episodes; the next ceil(M/length)-1 calls serve
        # successive non-overlapping windows of the *same* episodes, where M is
        # the maximum episode length in the batch.
        self._current_eps = None  # list of episode dicts (length == batch)
        self._ep_lens = None  # list of int, actual episode lengths
        self._chunk_idx = 0  # which chunk is being served right now
        self._n_chunks = 0  # total chunks for the current episode batch

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

        # ---- start of a new episode-batch cycle ----
        if self._chunk_idx == 0:
            with self.lock:
                valid = [ep for ep in self.eps if ep is not None]
            self._current_eps = [
                valid[self.rng.integers(0, len(valid))] for _ in range(batch)
            ]
            self._ep_lens = [
                len(next(iter(ep.values()))) for ep in self._current_eps
            ]
            M = max(self._ep_lens)
            self._n_chunks = math.ceil(M / L)

        # ---- build the chunk for the current index ----
        c = self._chunk_idx
        start = c * L
        end = start + L

        seqs = []
        for ep, ep_len in zip(self._current_eps, self._ep_lens):
            real_start = min(start, ep_len)
            real_end = min(end, ep_len)
            real_len = real_end - real_start
            pad_len = L - real_len

            chunk = {}
            for k, v in ep.items():
                if real_len > 0:
                    arr = v[real_start:real_end]
                    if pad_len > 0:
                        pad = np.zeros((pad_len, *v.shape[1:]), v.dtype)
                        arr = np.concatenate([arr, pad], axis=0)
                else:
                    arr = np.zeros((L, *v.shape[1:]), v.dtype)
                chunk[k] = arr
            seqs.append(chunk)

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}

        # t_mask[i, t] = True  <=>  transition t of episode i is real data,
        #                           not zero-padding.
        t_mask = np.zeros((batch, L), dtype=bool)
        for i, ep_len in enumerate(self._ep_lens):
            real_start = min(start, ep_len)
            real_end = min(end, ep_len)
            real_len = real_end - real_start
            t_mask[i, :real_len] = True
        data['t_mask'] = t_mask

        # Advance and wrap the chunk counter.
        self._chunk_idx = (self._chunk_idx + 1) % self._n_chunks

        return data
