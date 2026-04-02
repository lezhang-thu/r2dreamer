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

    def num_segments(self):
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        if not valid:
            return 0
        ep_lens = np.asarray([len(next(iter(ep.values()))) for ep in valid],
                             dtype=np.int64)
        return int(np.maximum(ep_lens - self.length + 1, 1).sum())

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
        ep_lens = np.asarray([len(next(iter(ep.values()))) for ep in valid],
                             dtype=np.int64)
        segment_counts = np.maximum(ep_lens - L + 1, 1)
        segment_offsets = np.cumsum(segment_counts)
        total_segments = int(segment_offsets[-1])
        sampled_segments = self.rng.integers(0, total_segments, size=batch)

        seqs = []
        valid_lens = []
        for seg_id in sampled_segments:
            ep_index = int(
                np.searchsorted(segment_offsets, seg_id, side="right"))
            prev_offset = 0 if ep_index == 0 else int(segment_offsets[ep_index -
                                                                      1])
            start = int(seg_id - prev_offset)
            ep = valid[ep_index]
            ep_len = int(ep_lens[ep_index])
            stop = min(start + L, ep_len)
            valid_len = stop - start

            item = {}
            for k, v in ep.items():
                arr = v[start:stop]
                if valid_len < L:
                    pad = np.zeros((L - valid_len, *v.shape[1:]), dtype=v.dtype)
                    arr = np.concatenate([arr, pad], axis=0)
                item[k] = arr
            seqs.append(item)
            valid_lens.append(valid_len)

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}

        # t_mask[i, t] = True iff step t is real (not padding).
        t_mask = np.zeros((batch, L), dtype=bool)
        for i, valid_len in enumerate(valid_lens):
            t_mask[i, :valid_len] = True
        data["t_mask"] = t_mask

        if "is_first" in data:
            data["is_first"][:, 0] = True
        return data
