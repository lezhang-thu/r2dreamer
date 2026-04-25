import threading

import numpy as np


class ReplayY:

    def __init__(self,
                 length,
                 capacity=1000,
                 seed=0,
                 memory=None,
                 memory_sample_frac=0.0,
                 reward_interval_sample_frac=0.8,
                 reward_stability_window=10,
                 reward_stability_frac=1.0):
        self.length = length
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.memory_sample_frac = float(memory_sample_frac)
        if not 0.0 <= self.memory_sample_frac <= 1.0:
            raise ValueError("memory_sample_frac must be in [0, 1], got "
                             f"{self.memory_sample_frac}.")
        self.reward_interval_sample_frac = float(reward_interval_sample_frac)
        if not 0.0 <= self.reward_interval_sample_frac <= 1.0:
            raise ValueError(
                "reward_interval_sample_frac must be in [0, 1], got "
                f"{self.reward_interval_sample_frac}.")
        self.reward_stability_window = int(reward_stability_window)
        if self.reward_stability_window < 1:
            raise ValueError("reward_stability_window must be >= 1, got "
                             f"{self.reward_stability_window}.")
        self.reward_stability_frac = float(reward_stability_frac)
        if not 0.0 < self.reward_stability_frac <= 1.0:
            raise ValueError("reward_stability_frac must be in (0, 1], got "
                             f"{self.reward_stability_frac}.")

        # Cyclic list of complete episodes; each slot is None or a dict of
        # arrays with shape (ep_len, *field_shape).
        self.eps = [None] * self.capacity
        self.write_pos = 0
        self.num_eps = 0  # episodes stored so far (capped at capacity)
        self.recent_reward_counts = []
        self.stable_reward_count = 0
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

    @staticmethod
    def _reward_indices(ep):
        reward = ep.get("reward")
        if reward is None:
            return np.asarray([], dtype=np.int64)
        reward = np.asarray(reward)
        flat = reward.reshape(len(reward), -1)
        return np.flatnonzero(np.any(flat != 0, axis=1)).astype(np.int64)

    @staticmethod
    def _stable_count(counts, window, frac):
        if len(counts) < window:
            return 0
        recent = np.asarray(counts[-window:], dtype=np.int64)
        if recent.size == 0:
            return 0

        required = int(np.ceil(window * frac))
        stable = 0
        for reward_count in range(1, int(np.max(recent)) + 1):
            if int(np.count_nonzero(recent >= reward_count)) >= required:
                stable = reward_count
            else:
                break
        return stable

    def _reward_interval_bounds(self, ep, interval_index):
        """Return inclusive state-index bounds for a reward-centered interval."""
        ep_len = int(len(next(iter(ep.values()))))
        if ep_len <= 0 or interval_index < 0:
            return None

        rewards = self._reward_indices(ep)
        reward_count = len(rewards)
        if reward_count == 0:
            return None

        if interval_index == 0:
            start = 0
            stop = rewards[1] if reward_count >= 2 else ep_len - 1
        elif interval_index < reward_count - 1:
            start = rewards[interval_index - 1]
            stop = rewards[interval_index + 1]
        elif interval_index == reward_count - 1:
            start = rewards[reward_count - 2] if reward_count >= 2 else 0
            stop = ep_len - 1
        elif interval_index == reward_count:
            # Live replay episodes at the current frontier have not reached
            # the next reward yet. Bias their observed tail after the last
            # reward so replay samples still emphasize frontier context.
            start = rewards[-1]
            stop = ep_len - 1
        else:
            return None

        start = int(np.clip(start, 0, ep_len - 1))
        stop = int(np.clip(stop, start, ep_len - 1))
        return start, stop

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
        reward_count = len(self._reward_indices(ep))
        with self.lock:
            self.eps[self.write_pos] = ep
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.num_eps = min(self.num_eps + 1, self.capacity)
            self.recent_reward_counts.append(reward_count)
            overflow = (len(self.recent_reward_counts) -
                        self.reward_stability_window)
            if overflow > 0:
                del self.recent_reward_counts[:overflow]
            self.stable_reward_count = self._stable_count(
                self.recent_reward_counts, self.reward_stability_window,
                self.reward_stability_frac)

    def _sample_from_ranges(self, ranges, batch):
        if batch <= 0:
            return np.asarray([], dtype=np.int64)

        counts = np.asarray([count for _, count in ranges], dtype=np.int64)
        offsets = np.cumsum(counts)
        total = int(offsets[-1])
        choices = self.rng.integers(0, total, size=batch)
        sampled = np.empty(batch, dtype=np.int64)
        for i, choice in enumerate(choices):
            range_index = int(np.searchsorted(offsets, choice, side="right"))
            prev_offset = 0 if range_index == 0 else int(
                offsets[range_index - 1])
            start, _ = ranges[range_index]
            sampled[i] = int(start + choice - prev_offset)
        return sampled

    def _preferred_segment_ranges(self, episodes, ep_lens, segment_offsets,
                                  interval_index):
        if interval_index is None:
            return []

        ranges = []
        L = self.length
        for ep_index, ep in enumerate(episodes):
            bounds = self._reward_interval_bounds(ep, interval_index)
            if bounds is None:
                continue

            interval_start, interval_stop = bounds
            segment_count = max(int(ep_lens[ep_index]) - L + 1, 1)
            max_start = segment_count - 1
            base_offset = 0 if ep_index == 0 else int(
                segment_offsets[ep_index - 1])

            # A fixed-length training sequence is useful for this interval if
            # the sequence overlaps the interval by at least one real step.
            start = max(0, interval_start - L + 1)
            stop = min(interval_stop, max_start)
            if start <= stop:
                ranges.append((base_offset + start, stop - start + 1))
        return ranges

    def _sample_episode_segments(self, episodes, batch, interval_index=None):
        if batch <= 0:
            return [], []
        if not episodes:
            raise RuntimeError("ReplayY.sample() requested segments from an "
                               "empty episode source.")
        L = self.length
        ep_lens = np.asarray([len(next(iter(ep.values()))) for ep in episodes],
                             dtype=np.int64)
        segment_counts = np.maximum(ep_lens - L + 1, 1)
        segment_offsets = np.cumsum(segment_counts)
        total_segments = int(segment_offsets[-1])

        preferred_ranges = self._preferred_segment_ranges(
            episodes, ep_lens, segment_offsets, interval_index)
        preferred_batch = 0
        if preferred_ranges:
            preferred_batch = int(
                np.floor(batch * self.reward_interval_sample_frac + 0.5))
            preferred_batch = int(np.clip(preferred_batch, 0, batch))
        uniform_batch = int(batch) - preferred_batch

        sampled_segments = []
        if preferred_batch > 0:
            sampled_segments.append(
                self._sample_from_ranges(preferred_ranges, preferred_batch))
        if uniform_batch > 0:
            sampled_segments.append(
                self.rng.integers(0, total_segments, size=uniform_batch))
        sampled_segments = np.concatenate(sampled_segments)
        if len(sampled_segments) > 1:
            self.rng.shuffle(sampled_segments)

        seqs = []
        valid_lens = []
        for seg_id in sampled_segments:
            ep_index = int(
                np.searchsorted(segment_offsets, seg_id, side="right"))
            prev_offset = 0 if ep_index == 0 else int(segment_offsets[ep_index -
                                                                      1])
            start = int(seg_id - prev_offset)
            ep = episodes[ep_index]
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
        return seqs, valid_lens

    def sample(self, batch):
        batch = int(batch)
        L = self.length
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
            target_interval_index = int(self.stable_reward_count)
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
        valid_lens = []
        if replay_batch > 0:
            replay_seqs, replay_valid_lens = self._sample_episode_segments(
                valid, replay_batch, target_interval_index)
            seqs.extend(replay_seqs)
            valid_lens.extend(replay_valid_lens)
        if memory_batch > 0:
            memory_seqs, memory_valid_lens = self._sample_episode_segments(
                [self.memory], memory_batch)
            seqs.extend(memory_seqs)
            valid_lens.extend(memory_valid_lens)

        if len(seqs) != int(batch):
            raise RuntimeError("ReplayY.sample() produced an unexpected number "
                               f"of segments: {len(seqs)} != {int(batch)}.")
        if len(seqs) > 1:
            order = self.rng.permutation(len(seqs))
            seqs = [seqs[i] for i in order]
            valid_lens = [valid_lens[i] for i in order]

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}

        # t_mask[i, t] = True iff step t is real (not padding).
        t_mask = np.zeros((batch, L), dtype=bool)
        for i, valid_len in enumerate(valid_lens):
            t_mask[i, :valid_len] = True
        data["t_mask"] = t_mask

        if "is_first" in data:
            data["is_first"][:, 0] = True
        return data
