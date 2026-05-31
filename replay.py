import threading

import numpy as np


class Replay:

    def __init__(self, length, capacity=1000, seed=0):
        self.length = int(length)
        if self.length < 1:
            raise ValueError(f"length must be >= 1, got {self.length}.")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)

        # Cyclic list of complete episodes; each slot is None or a dict of
        # arrays with shape (ep_len, *field_shape).
        self.eps = [None] * self.capacity
        self.write_pos = 0
        self.num_eps = 0  # episodes stored so far (capped at capacity)

        # Per-worker buffers accumulate steps until episode end.
        self.local = {}
        # Per-batch-row streaming cursors used for Transformer-XL segments.
        self.streams = []
        self.lock = threading.Lock()

    def __len__(self):
        return sum(1 for ep in self.eps if ep is not None)

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
            raise ValueError("Replay episode cannot be empty.")
        lengths = {len(v) for v in episode.values()}
        if len(lengths) != 1:
            raise ValueError("Replay episode fields must all share the "
                             f"same leading length, got {sorted(lengths)}.")
        if next(iter(lengths)) <= 0:
            raise ValueError("Replay episode length must be positive.")
        return episode

    @staticmethod
    def _episode_length(ep):
        return int(len(next(iter(ep.values()))))

    def can_sample(self, batch):
        """Return whether sample(batch) can produce segment streams."""
        batch = int(batch)
        if batch <= 0:
            return False
        with self.lock:
            has_replay = any(ep is not None for ep in self.eps)
        return has_replay

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
        ep = self._sanitize_episode(ep)
        with self.lock:
            self.eps[self.write_pos] = ep
            self.write_pos = (self.write_pos + 1) % self.capacity
            self.num_eps = min(self.num_eps + 1, self.capacity)

    def _sample_episode(self, episodes):
        if not episodes:
            raise RuntimeError("Replay.sample() requested an episode from an "
                               "empty source.")
        return episodes[int(self.rng.integers(0, len(episodes)))]

    def _new_stream(self, replay_episodes):
        episode = self._sample_episode(replay_episodes)
        offset_limit = min(self.length, self._episode_length(episode))
        offset = int(self.rng.integers(0, offset_limit))
        return {
            "episode": episode,
            "offset": offset,
            "position": 0,
        }

    def _new_sampled_stream(self, replay_episodes):
        return self._new_stream(replay_episodes)

    def _ensure_streams(self, batch, replay_episodes):
        if len(self.streams) > batch:
            self.streams = self.streams[:batch]
        while len(self.streams) < batch:
            self.streams.append(self._new_sampled_stream(replay_episodes))

    def _replace_exhausted_stream(self, stream, replay_episodes):
        stream.clear()
        stream.update(self._new_sampled_stream(replay_episodes))

    @staticmethod
    def _step_at(episode, offset):
        return {k: np.array(v[offset], copy=True) for k, v in episode.items()}

    def _sample_stream_segment(self, stream, replay_episodes):
        L = self.length
        items = []
        positions = []

        while len(items) < L:
            episode = stream["episode"]
            if stream["offset"] >= self._episode_length(episode):
                self._replace_exhausted_stream(stream, replay_episodes)
                episode = stream["episode"]

            offset = int(stream["offset"])
            position = int(stream["position"])
            step = self._step_at(episode, offset)
            if "is_first" in step:
                force_is_first = position == 0
                if force_is_first:
                    step["is_first"] = np.asarray(True,
                                                  dtype=step["is_first"].dtype)
            items.append(step)
            positions.append(position)

            stream["offset"] = offset + 1
            stream["position"] = position + 1
            is_last = bool(step.get("is_last", False))
            if is_last or stream["offset"] >= self._episode_length(episode):
                self._replace_exhausted_stream(stream, replay_episodes)

        seq = {
            k: np.stack([item[k] for item in items], axis=0) for k in items[0]
        }
        return seq, np.asarray(positions, dtype=np.int64)

    def sample(self, batch):
        with self.lock:
            valid = [ep for ep in self.eps if ep is not None]
        if not self.can_sample(batch):
            raise RuntimeError(
                "Replay.sample() called before enough complete episodes are "
                "available for the requested batch.")

        batch = int(batch)
        self._ensure_streams(batch, valid)

        seqs = []
        positions = []
        for stream in self.streams:
            seq, position = self._sample_stream_segment(stream, valid)
            seqs.append(seq)
            positions.append(position)

        data = {k: np.stack([s[k] for s in seqs]) for k in seqs[0]}
        data["position"] = np.stack(positions)
        return data
