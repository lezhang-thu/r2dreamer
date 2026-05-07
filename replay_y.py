import threading

import numpy as np


class ReplayY:

    def __init__(self,
                 length,
                 capacity=1000,
                 seed=0,
                 memory=None,
                 memory_sample_frac=0.0):
        self.length = int(length)
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}.")
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
        return 1

    def num_segments(self):
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        total = sum(self._episode_segment_count(ep) for ep in valid)
        if self.memory is not None:
            total += self._episode_segment_count(self.memory)
        return int(total)

    def can_sample(self, batch):
        """Return whether sample(batch) can produce dense replay rows."""
        batch = int(batch)
        if batch <= 0:
            return False
        with self.lock:
            has_replay = any(ep is not None for ep in self.eps)
        has_memory = self.memory is not None
        if not has_replay and not has_memory:
            return False
        # Expert memory is included exactly once. Any remaining rows must come
        # from non-memory replay episodes so memory is not duplicated.
        non_memory_rows = batch - int(has_memory)
        return non_memory_rows <= 0 or has_replay

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

    def _sample_episode(self, episodes):
        if not episodes:
            raise RuntimeError("ReplayY.sample() requested an episode from an "
                               "empty source.")
        return episodes[int(self.rng.integers(0, len(episodes)))]

    def _build_sequence(self, first_episode, append_episodes):
        """Concatenate full episodes and a final episode prefix to length L."""
        L = self.length
        if not append_episodes:
            append_episodes = [first_episode]
        pieces = []
        positions = []
        pos = 0
        ep = first_episode
        while pos < L:
            ep_len = int(len(next(iter(ep.values()))))
            if ep_len <= 0:
                raise RuntimeError("ReplayY encountered an empty episode.")
            take = min(ep_len, L - pos)
            piece = {k: np.array(v[:take], copy=True) for k, v in ep.items()}
            if "is_first" in piece:
                piece["is_first"][0] = True
            pieces.append(piece)
            positions.append(np.arange(take, dtype=np.int64))
            pos += take
            if pos < L:
                ep = self._sample_episode(append_episodes)

        item = {}
        for key in pieces[0]:
            item[key] = np.concatenate([piece[key] for piece in pieces],
                                       axis=0)
        position = np.concatenate(positions, axis=0)
        return item, position

    def _sample_episode_sequences(self, first_episodes, append_episodes):
        seqs = []
        positions = []
        for ep in first_episodes:
            seq, position = self._build_sequence(ep, append_episodes)
            seqs.append(seq)
            positions.append(position)
        return seqs, positions

    def sample(self, batch):
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        if not valid and self.memory is None:
            raise RuntimeError(
                "ReplayY.sample() called with no complete episodes.")

        batch = int(batch)
        memory_batch = int(self.memory is not None and batch > 0)
        replay_batch = batch - memory_batch
        if replay_batch > 0 and not valid:
            raise RuntimeError(
                "ReplayY.sample() needs replay episodes to fill the non-memory "
                "batch rows while keeping expert memory exactly once.")
        append_episodes = valid if valid else (
            [self.memory] if self.memory is not None else [])

        seqs = []
        positions = []
        if replay_batch > 0:
            replay_first = [
                self._sample_episode(valid) for _ in range(replay_batch)
            ]
            replay_seqs, replay_positions = self._sample_episode_sequences(
                replay_first, append_episodes)
            seqs.extend(replay_seqs)
            positions.extend(replay_positions)
        if memory_batch > 0:
            memory_seqs, memory_positions = self._sample_episode_sequences(
                [self.memory], append_episodes)
            seqs.extend(memory_seqs)
            positions.extend(memory_positions)

        if len(seqs) != int(batch):
            raise RuntimeError("ReplayY.sample() produced an unexpected number "
                               f"of segments: {len(seqs)} != {int(batch)}.")
        if len(seqs) > 1:
            order = self.rng.permutation(len(seqs))
            seqs = [seqs[i] for i in order]
            positions = [positions[i] for i in order]

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}
        data["position"] = np.stack(positions)
        return data
