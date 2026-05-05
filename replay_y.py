import threading

import numpy as np


class ReplayY:

    def __init__(self,
                 length,
                 capacity=1000,
                 seed=0,
                 memory=None,
                 memory_sample_frac=0.0,
                 prefix_length=0):
        self.length = int(length)
        self.prefix_length = int(prefix_length)
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}.")
        if self.prefix_length < 0:
            raise ValueError("prefix_length must be >= 0, got "
                             f"{self.prefix_length}.")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.memory_sample_frac = float(memory_sample_frac)
        if not 0.0 <= self.memory_sample_frac <= 1.0:
            raise ValueError("memory_sample_frac must be in [0, 1], got "
                             f"{self.memory_sample_frac}.")

        # Cyclic list of complete episodes; each slot is None or a dict of
        # arrays with shape (ep_len, *field_shape).
        self.eps = [None] * self.capacity
        self.write_pos = 0
        self.num_eps = 0  # episodes stored so far (capped at capacity)
        self.memory = self._sanitize_episode(
            memory) if memory is not None else None

        # Per-worker buffers that accumulate steps until episode ends.
        self.local = {}
        self.lock = threading.Lock()

    def __len__(self):
        # Count any stored episode (zero-padding handles episodes shorter than
        # self.length, so every non-None slot is valid).
        return sum(1 for ep in self.eps if ep is not None) + int(
            self.memory is not None)

    @staticmethod
    def _copy_episode(episode):
        if episode is None:
            return None
        return {k: np.array(v, copy=True) for k, v in episode.items()}

    def state_dict(self):
        """Return a serializable ReplayY state without expert memory data."""
        with self.lock:
            eps = [self._copy_episode(ep) for ep in self.eps]
        return {"eps": eps}

    def load_state_dict(self, state_dict):
        """Load completed replay episodes, leaving self.memory untouched."""
        if isinstance(state_dict, dict):
            if "eps" not in state_dict:
                raise KeyError("ReplayY state_dict must contain an 'eps' key.")
            eps = state_dict["eps"]
        else:
            # Accept raw eps lists for simple/manual checkpoints.
            eps = state_dict

        if not isinstance(eps, (list, tuple)):
            raise TypeError("ReplayY eps checkpoint must be a list or tuple.")
        if len(eps) > self.capacity:
            raise ValueError(
                "ReplayY eps checkpoint is larger than this buffer capacity: "
                f"{len(eps)} > {self.capacity}.")

        loaded_eps = []
        for ep in eps:
            loaded_eps.append(None if ep is
                              None else self._sanitize_episode(ep))
        loaded_eps.extend([None] * (self.capacity - len(loaded_eps)))

        with self.lock:
            self.eps = loaded_eps
            self.num_eps = sum(ep is not None for ep in self.eps)
            self.write_pos = next(
                (i for i, ep in enumerate(self.eps) if ep is None), 0)
            self.local = {}

    @staticmethod
    def _sanitize_episode(episode):
        if episode is None:
            return None
        episode = {
            k: np.asarray(v)
            for k, v in episode.items()
            if not k.startswith('log_')
        }
        if not episode:
            raise ValueError("ReplayY memory episode cannot be empty.")
        lengths = {len(v) for v in episode.values()}
        if len(lengths) != 1:
            raise ValueError("ReplayY memory episode fields must all share the "
                             f"same leading length, got {sorted(lengths)}.")
        return episode

    def _episode_segment_count(self, ep):
        ep_len = int(len(next(iter(ep.values()))))
        return max(ep_len - self.length + 1, 1)

    def num_segments(self):
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        total = sum(self._episode_segment_count(ep) for ep in valid)
        if self.memory is not None:
            total += self._episode_segment_count(self.memory)
        return int(total)

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

    def _sample_episode_segments(self, episodes, batch):
        if batch <= 0:
            return [], [], [], [], []
        if not episodes:
            raise RuntimeError("ReplayY.sample() requested segments from an "
                               "empty episode source.")
        L = self.length
        P = self.prefix_length
        total_len = P + L
        ep_lens = np.asarray([len(next(iter(ep.values()))) for ep in episodes],
                             dtype=np.int64)
        segment_counts = np.maximum(ep_lens - L + 1, 1)
        segment_offsets = np.cumsum(segment_counts)
        total_segments = int(segment_offsets[-1])
        sampled_segments = self.rng.integers(0, total_segments, size=batch)

        seqs = []
        seq_masks = []
        target_masks = []
        target_starts = []
        positions = []
        for seg_id in sampled_segments:
            ep_index = int(
                np.searchsorted(segment_offsets, seg_id, side="right"))
            prev_offset = 0 if ep_index == 0 else int(segment_offsets[ep_index -
                                                                      1])
            target_start = int(seg_id - prev_offset)
            ep = episodes[ep_index]
            ep_len = int(ep_lens[ep_index])
            seq_start = target_start - P
            seq_stop = target_start + L
            src_start = max(seq_start, 0)
            src_stop = min(seq_stop, ep_len)
            dst_start = src_start - seq_start
            real_len = max(src_stop - src_start, 0)
            target_valid_len = max(
                min(target_start + L, ep_len) - target_start, 0)

            item = {}
            for k, v in ep.items():
                arr = np.zeros((total_len, *v.shape[1:]), dtype=v.dtype)
                if real_len > 0:
                    arr[dst_start:dst_start + real_len] = v[src_start:src_stop]
                item[k] = arr
            seqs.append(item)
            seq_mask = np.zeros((total_len,), dtype=bool)
            if real_len > 0:
                seq_mask[dst_start:dst_start + real_len] = True
            target_mask = np.zeros((total_len,), dtype=bool)
            if target_valid_len > 0:
                target_mask[P:P + target_valid_len] = True
            position = np.zeros((total_len,), dtype=np.int64)
            if real_len > 0:
                position[dst_start:dst_start + real_len] = np.arange(
                    src_start, src_stop, dtype=np.int64)
            seq_masks.append(seq_mask)
            target_masks.append(target_mask)
            target_starts.append(np.full((total_len,), P, dtype=np.int64))
            positions.append(position)
        return seqs, seq_masks, target_masks, target_starts, positions

    def sample(self, batch):
        L = self.length
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        if not valid and self.memory is None:
            raise RuntimeError(
                "ReplayY.sample() called with no complete episodes.")

        memory_batch = 0
        if self.memory is not None:
            if valid:
                memory_batch = int(
                    np.floor(batch * self.memory_sample_frac + 0.5))
                memory_batch = int(np.clip(memory_batch, 0, batch))
            else:
                memory_batch = int(batch)
        replay_batch = int(batch) - memory_batch
        if replay_batch > 0 and not valid:
            replay_batch = 0
            memory_batch = int(batch)
        if memory_batch > 0 and self.memory is None:
            memory_batch = 0
            replay_batch = int(batch)

        seqs = []
        seq_masks = []
        target_masks = []
        target_starts = []
        positions = []
        if replay_batch > 0:
            replay_seqs, replay_seq_masks, replay_target_masks, replay_target_starts, replay_positions = self._sample_episode_segments(
                valid, replay_batch)
            seqs.extend(replay_seqs)
            seq_masks.extend(replay_seq_masks)
            target_masks.extend(replay_target_masks)
            target_starts.extend(replay_target_starts)
            positions.extend(replay_positions)
        if memory_batch > 0:
            memory_seqs, memory_seq_masks, memory_target_masks, memory_target_starts, memory_positions = self._sample_episode_segments(
                [self.memory], memory_batch)
            seqs.extend(memory_seqs)
            seq_masks.extend(memory_seq_masks)
            target_masks.extend(memory_target_masks)
            target_starts.extend(memory_target_starts)
            positions.extend(memory_positions)

        if len(seqs) != int(batch):
            raise RuntimeError("ReplayY.sample() produced an unexpected number "
                               f"of segments: {len(seqs)} != {int(batch)}.")
        if len(seqs) > 1:
            order = self.rng.permutation(len(seqs))
            seqs = [seqs[i] for i in order]
            seq_masks = [seq_masks[i] for i in order]
            target_masks = [target_masks[i] for i in order]
            target_starts = [target_starts[i] for i in order]
            positions = [positions[i] for i in order]

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}

        # seq_mask marks real timesteps, including prefix context. t_mask marks
        # the target portion that contributes to losses and imagination starts.
        data["seq_mask"] = np.stack(seq_masks)
        data["t_mask"] = np.stack(target_masks)
        data["target_start"] = np.stack(target_starts)
        data["position"] = np.stack(positions)
        return data
