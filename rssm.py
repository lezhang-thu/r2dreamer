import torch
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F

import distributions as dists
from _positional_embeddings import Qwen2RotaryPositionalEmbeddings
from networks import LambdaLayer
from tools import weight_init_


class TransformerRSSM(nn.Module):

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._feat_deter = int(getattr(config, "feat_deter", self._deter))
        self._discrete = int(config.discrete)
        self._unimix_ratio = float(config.unimix_ratio)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._n_heads = int(config.n_heads)
        self._n_layers = int(config.n_layers)
        self._d_ff = int(config.d_ff)
        self._memory_size = int(getattr(config, "memory_size", 0))
        configured_segment = getattr(config, "segment_length", None)
        if configured_segment is None:
            configured_segment = getattr(config, "window_size", None)
        if configured_segment is None:
            raise AttributeError(
                "TransformerRSSM config requires segment_length.")
        self._segment_length = int(configured_segment)
        if self._memory_size < 0:
            raise AssertionError("memory_size must be >= 0.")
        if self._segment_length < 1:
            raise AssertionError("segment_length must be >= 1.")
        # Carries store only previous-token memory. Segment length remains a
        # training sequence/config property, not part of cache length.
        self._cache_size = self._memory_size
        act_fn = getattr(torch.nn, config.act)

        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._feat_deter

        D = self._deter
        H = self._n_heads
        assert D % H == 0, f"deter ({D}) must be divisible by n_heads ({H})"
        self._d_head = D // H
        assert self._d_head % 2 == 0, (
            f"d_head ({self._d_head}) must be even for RoPE")

        self._rope_base = float(getattr(config, 'rope_base', 1_000_000.0))
        self._rope_max_seq_len = int(
            getattr(config, 'rope_max_seq_len',
                    max(self._cache_size, self._segment_length, 4096)))
        self._rope = Qwen2RotaryPositionalEmbeddings(
            dim=self._d_head,
            max_seq_len=self._rope_max_seq_len,
            base=self._rope_base,
        )

        # Activation for FFN sublayers
        self._act_fn = act_fn()

        # Input projection: (flat_stoch + action) -> deter
        self._inp_proj = nn.Linear(self.flat_stoch + act_dim, D)

        # Per-layer transformer components
        self._attn_norms = nn.ModuleList()
        self._q_projs = nn.ModuleList()
        self._k_projs = nn.ModuleList()
        self._v_projs = nn.ModuleList()
        self._o_projs = nn.ModuleList()
        self._ffn_norms = nn.ModuleList()
        self._ff1s = nn.ModuleList()
        self._ff2s = nn.ModuleList()

        for _ in range(self._n_layers):
            self._attn_norms.append(
                nn.RMSNorm(D, eps=1e-04, dtype=torch.float32))
            self._q_projs.append(nn.Linear(D, D, bias=False))
            self._k_projs.append(nn.Linear(D, D, bias=False))
            self._v_projs.append(nn.Linear(D, D, bias=False))
            self._o_projs.append(nn.Linear(D, D, bias=False))
            self._ffn_norms.append(nn.RMSNorm(D, eps=1e-04,
                                              dtype=torch.float32))
            self._ff1s.append(nn.Linear(D, self._d_ff))
            self._ff2s.append(nn.Linear(self._d_ff, D))

        # Output norm
        self._outnorm = nn.RMSNorm(D, eps=1e-04, dtype=torch.float32)
        if self._feat_deter == D:
            self._feat_proj = nn.Identity()
        else:
            self._feat_proj = nn.Linear(D, self._feat_deter, bias=True)

        # Richer posterior/prior heads, analogous to RSSM obs/img heads.
        self._head_hidden = int(getattr(config, 'head_hidden', D))
        self._post_layers = int(getattr(config, 'post_layers', 1))
        self._prior_layers = int(getattr(config, 'prior_layers', 2))

        self._post_head = nn.Sequential()
        inp_dim = embed_size
        for i in range(self._post_layers):
            self._post_head.add_module(
                f"post_{i}", nn.Linear(inp_dim, self._head_hidden, bias=True))
            self._post_head.add_module(
                f"post_n_{i}",
                nn.RMSNorm(self._head_hidden, eps=1e-04, dtype=torch.float32))
            self._post_head.add_module(f"post_a_{i}", act_fn())
            inp_dim = self._head_hidden
        self._post_head.add_module(
            "post_logit",
            nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._post_head.add_module(
            "post_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self.
                                            _discrete)),
        )

        self._prior_head = nn.Sequential()
        inp_dim = D
        for i in range(self._prior_layers):
            self._prior_head.add_module(
                f"prior_{i}", nn.Linear(inp_dim, self._head_hidden, bias=True))
            self._prior_head.add_module(
                f"prior_n_{i}",
                nn.RMSNorm(self._head_hidden, eps=1e-04, dtype=torch.float32))
            self._prior_head.add_module(f"prior_a_{i}", act_fn())
            inp_dim = self._head_hidden
        self._prior_head.add_module(
            "prior_logit",
            nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._prior_head.add_module(
            "prior_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self.
                                            _discrete)),
        )

        self.apply(weight_init_)

    # ------------------------------------------------------------------
    # Training path
    # ------------------------------------------------------------------

    def observe(self,
                tokens,
                action,
                reset,
                *,
                memory_carry,
                sample=True,
                positions=None):
        """Observe one real segment using detached Transformer-XL memory.

        Args:
            tokens: (B, T, E) encoder embeddings.
            action: (B, T, A) current actions a_t (not prev_action).
            reset: (B, T) boolean, True at episode start.
            sample: Whether to sample posterior stochastics. When False,
                uses the posterior mode, which is useful for deterministic
                cached reference trajectories.
            positions: Optional absolute episode positions, shape (B, T).
            memory_carry: Detached Transformer-XL carry for segment training
                without replay-side prefix tokens.
        Returns:
            entries: dict with 'deter' (B,T,D) and 'stoch' (B,T,S,K).
            feat: dict with deter, stoch, post_logit, prior_logit, and
                trajectory KV tensors for imagination starts.
        """
        B, T = tokens.shape[:2]
        if positions is None:
            positions = torch.arange(T, device=tokens.device,
                                     dtype=torch.long).unsqueeze(0).expand(
                                         B, -1)
        else:
            positions = positions.to(device=tokens.device, dtype=torch.long)

        carry = self._mask_carry(memory_carry, reset[:, 0])
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()
        post_logit = self._post_head(tokens)
        post_dist = self.get_dist(post_logit)
        if sample:
            stoch = post_dist.rsample()
        else:
            stoch = post_dist.base_dist.mode

        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        x = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))
        h, kv, next_carry = self._fwd_segment_with_carry(
            x, carry, positions, reset)

        h_prev = torch.cat([carry['h_prev'].unsqueeze(1), h[:, :-1]], dim=1)
        h_prev = h_prev * (1.0 - reset.unsqueeze(-1).float())
        prior_logit = self._prior_head(h_prev)

        kv_k = torch.cat([carry['kv_cache'][:, :, 0].detach(), kv['k']], dim=2)
        kv_v = torch.cat([carry['kv_cache'][:, :, 1].detach(), kv['v']], dim=2)
        entries = {'deter': h_prev, 'stoch': stoch}
        feat = {
            'deter': h_prev,
            'stoch': stoch,
            'post_logit': post_logit,
            'prior_logit': prior_logit,
            'kv_k': kv_k.detach(),
            'kv_v': kv_v.detach(),
            'next_carry': next_carry,
        }
        return entries, feat

    def _apply_rope(self, x, positions=None):
        """Apply RoPE to (B, H, T, D_head) tensor."""
        # Qwen2 RoPE expects (B, T, H, D_head).
        x_t = x.transpose(1, 2)
        if positions is None:
            x_t = self._rope(x_t, input_pos=None)
        else:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(x.shape[0], -1)
            x_t = self._rope(x_t,
                             input_pos=positions.to(device=x_t.device,
                                                    dtype=torch.long))
        return x_t.transpose(1, 2)

    def _fwd_segment_with_carry(self, x, carry, positions, reset):
        """Forward a segment against detached sliding-window memory."""
        B, T, D = x.shape
        H = self._n_heads
        D_head = self._d_head
        M = self._memory_size
        C = int(carry['kv_cache'].shape[3])
        if C != M:
            raise AssertionError("TransformerRSSM.observe expects a compact "
                                 f"memory carry of length {M}, got {C}.")
        positions = positions.to(device=x.device, dtype=torch.long)
        reset = reset.to(device=x.device, dtype=torch.bool)

        # Each query attends to at most M previous tokens plus itself across
        # detached memory and the current segment. Memory is visible only until
        # a reset appears inside the current segment.
        reset_count = reset.to(dtype=torch.long).cumsum(dim=1)
        n_mem = torch.clamp(carry['pos'].to(device=x.device), min=0, max=C)
        j_mem = torch.arange(C, device=x.device)
        mem_mask = j_mem.unsqueeze(0) >= (C - n_mem).unsqueeze(1)
        mem_time = (j_mem - C).view(1, 1, C)
        q_index = torch.arange(T, device=x.device).view(1, T, 1)
        mem_mask = mem_mask[:, None, :]
        mem_mask = mem_mask & (mem_time >= q_index - M)
        mem_mask = mem_mask & (reset_count == 0)[:, :, None]

        episode = reset_count
        k_index = torch.arange(T, device=x.device).view(1, 1, T)
        cur_mask = (k_index <= q_index) & (k_index >= q_index - M)
        cur_mask = cur_mask & (episode.unsqueeze(2) == episode.unsqueeze(1))
        attn_mask = torch.cat([mem_mask, cur_mask], dim=2).unsqueeze(1)

        k_layers = []
        v_layers = []
        new_cache_layers = []
        for i in range(self._n_layers):
            res = x
            x = self._attn_norms[i](x)
            Q = self._q_projs[i](x).reshape(B, T, H, D_head).transpose(1, 2)
            K_cur = self._k_projs[i](x).reshape(B, T, H, D_head).transpose(1, 2)
            V_cur = self._v_projs[i](x).reshape(B, T, H, D_head).transpose(1, 2)
            Q = self._apply_rope(Q, positions=positions)
            K_cur = self._apply_rope(K_cur, positions=positions)

            k_cur_flat = K_cur.transpose(1, 2).reshape(B, T, D)
            v_cur_flat = V_cur.transpose(1, 2).reshape(B, T, D)
            k_layers.append(k_cur_flat)
            v_layers.append(v_cur_flat)

            k_mem = carry['kv_cache'][:, i, 0].detach()
            v_mem = carry['kv_cache'][:, i, 1].detach()
            k_all = torch.cat([k_mem, k_cur_flat], dim=1)
            v_all = torch.cat([v_mem, v_cur_flat], dim=1)
            K_all = k_all.reshape(B, C + T, H, D_head).transpose(1, 2)
            V_all = v_all.reshape(B, C + T, H, D_head).transpose(1, 2)

            x = F.scaled_dot_product_attention(Q,
                                               K_all,
                                               V_all,
                                               attn_mask=attn_mask)
            x = x.transpose(1, 2).reshape(B, T, D)
            x = res + self._o_projs[i](x)

            res = x
            x = self._ffn_norms[i](x)
            x = self._ff2s[i](self._act_fn(self._ff1s[i](x)))
            x = res + x

            if M > 0:
                new_k = k_all[:, -M:]
                new_v = v_all[:, -M:]
            else:
                new_k = k_all[:, :0]
                new_v = v_all[:, :0]
            new_cache_layers.append((new_k, new_v))

        out = self._outnorm(x)
        ks = torch.stack([kv[0] for kv in new_cache_layers], dim=1)
        vs = torch.stack([kv[1] for kv in new_cache_layers], dim=1)
        next_kv_cache = torch.stack([ks, vs], dim=2).detach()
        next_pos = (positions[:, -1] + 1).to(torch.int32)
        next_carry = {
            'kv_cache': next_kv_cache,
            'pos': next_pos,
            'h_prev': out[:, -1].detach(),
        }
        return out, {
            'k': torch.stack(k_layers, dim=1),
            'v': torch.stack(v_layers, dim=1),
        }, next_carry

    # ------------------------------------------------------------------
    # Imagination (KV-cache transition)
    # ------------------------------------------------------------------

    def img_step_with_carry(self, stoch, carry, action):
        """Single prior step with KV-cache carry."""
        reset = torch.zeros(stoch.shape[0],
                            dtype=torch.bool,
                            device=stoch.device)
        carry = self.update_carry(carry, stoch, action, reset)
        deter = carry['h_prev']
        stoch, _ = self.prior(deter)
        return stoch, deter, carry

    @torch.no_grad()
    def build_imag_starts(self,
                          stoch_seq,
                          deter_seq,
                          kv_k,
                          kv_v,
                          starts,
                          positions=None):
        """Build imagination starts from trajectory KV tensors.

        Args:
            stoch_seq: (B, T, S, Kcat)
            deter_seq: (B, T, D) h_prev sequence
            kv_k: (B, L, M+T, D) cached keys with M memory slots followed by
                current-segment keys.
            kv_v: (B, L, M+T, D) cached values with M memory slots followed by
                current-segment values.
            starts: (B, K) integer start indices, all valid per episode
            positions: Optional (B, T) absolute episode positions for starts.
        Returns:
            start_stoch: (B*K, S, Kcat)
            start_deter: (B*K, D)
            carry: dict with kv_cache (B*K,L,2,M,D), pos, h_prev
        """
        B, T = stoch_seq.shape[:2]
        L = kv_k.shape[1]
        M = self._memory_size
        starts = starts.to(device=stoch_seq.device, dtype=torch.long)
        assert starts.ndim == 2 and starts.shape[0] == B
        assert kv_k.shape == (B, L, M + T, self._deter)
        assert kv_v.shape == (B, L, M + T, self._deter)
        if positions is not None:
            assert positions.shape == (B, T)
        assert torch.all(starts >= 0)
        assert torch.all(starts < T)

        K = starts.shape[1]
        D = self._deter
        batch_index = torch.arange(B, device=stoch_seq.device)[:, None]
        offsets = torch.arange(M, device=stoch_seq.device)

        start_stoch = stoch_seq[batch_index, starts]  # (B, K, S, Kcat)
        start_deter = deter_seq[batch_index, starts]  # (B, K, D)
        if positions is None:
            start_pos = starts
        else:
            start_pos = positions.to(device=stoch_seq.device,
                                     dtype=torch.long)[batch_index, starts]

        cache_list = []
        for k in range(K):
            s = starts[:, k]  # (B,)
            if M == 0:
                cache_list.append(kv_k.new_zeros(B, L, 2, 0, D))
                continue
            # Gather the previous M tokens immediately before start s. Times in
            # [-M, -1] address detached memory; times [0, s-1] address current
            # segment tokens. Positions, when available, hide keys from a
            # previous episode after a boundary inside a streamed segment.
            left_time = s[:, None] - M + offsets[None, :]
            gather_time = left_time + M
            valid_time = gather_time >= 0
            if positions is not None:
                left_pos = start_pos[:, k:k + 1] - M + offsets[None, :]
                valid_time = valid_time & (left_pos >= 0)
            gather_time = torch.clamp(gather_time, min=0, max=M + T - 1)
            gather_index = gather_time[:, None, :, None].expand(B, L, M, D)
            k_slice = kv_k.gather(2, gather_index)  # (B, L, M, D)
            v_slice = kv_v.gather(2, gather_index)  # (B, L, M, D)
            valid_time = valid_time[:, None, :, None].to(k_slice.dtype)
            k_slice = k_slice * valid_time
            v_slice = v_slice * valid_time
            cache_list.append(torch.stack([k_slice, v_slice], dim=2))

        kv_cache = torch.stack(cache_list, dim=1)  # (B, K, L, 2, M, D)
        start_stoch = start_stoch.reshape(B * K, *stoch_seq.shape[2:])
        start_deter = start_deter.reshape(B * K, deter_seq.shape[-1])
        carry = {
            'kv_cache': kv_cache.reshape(B * K, L, 2, M, D),
            'pos': start_pos.reshape(B * K).to(torch.int32),
            'h_prev': start_deter,
        }
        return start_stoch, start_deter, carry

    def imagine_with_action(self, stoch, deter, actions, carry=None):
        """Roll out prior dynamics given a sequence of actions.

        If carry is provided, rolling context is preserved with KV-cache.
        """
        L = actions.shape[1]
        stochs, deters = [], []
        if carry is None:
            carry = self.initial(stoch.shape[0])
            carry['h_prev'] = deter
        for i in range(L):
            stoch, deter, carry = self.img_step_with_carry(
                stoch, carry, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters, carry

    # ------------------------------------------------------------------
    # KV-cache policy inference
    # ------------------------------------------------------------------

    def _initial_carry(self, batch_size, cache_size):
        D = self._deter
        return {
            'kv_cache':
                torch.zeros(batch_size,
                            self._n_layers,
                            2,
                            cache_size,
                            D,
                            dtype=torch.float32,
                            device=self._device),
            'pos':
                torch.zeros(batch_size, dtype=torch.int32, device=self._device),
            'h_prev':
                torch.zeros(batch_size,
                            D,
                            dtype=torch.float32,
                            device=self._device),
        }

    def initial(self, batch_size):
        """Return initial carry dict for KV-cache inference/imagination."""
        return self._initial_carry(batch_size, self._cache_size)

    def initial_memory(self, batch_size):
        """Return initial carry for segment training."""
        return self._initial_carry(batch_size, self._memory_size)

    def get_feat_step(self, carry, tokens, reset):
        """Phase 1: posterior from tokens. No action needed.

        Args:
            carry: dict with kv_cache, pos, h_prev.
            tokens: (B, E) encoder embeddings for current step.
            reset: (B,) boolean, True at episode start.
        Returns:
            carry: updated carry (zeroed on reset).
            stoch: (B, S, K) sampled posterior stochastic state.
            h_prev: (B, D) transformer context from previous step.
        """
        carry = self._mask_carry(carry, reset)

        # Posterior from tokens only
        post_logit = self._post_head(tokens)  # (B, S, K)
        stoch = self.get_dist(post_logit).rsample()  # (B, S, K)

        return carry, stoch, carry['h_prev']

    def _mask_carry(self, carry, reset):
        """Zero carry state on episode reset."""
        reset_f = reset.float()
        h_prev = carry['h_prev'] * (1.0 - reset_f.unsqueeze(-1))
        kv_cache = carry['kv_cache'] * (1.0 - reset_f.reshape(-1, 1, 1, 1, 1))
        pos = carry['pos'] * (~reset).int()
        return {'kv_cache': kv_cache, 'pos': pos, 'h_prev': h_prev}

    def update_carry(self, carry, stoch, action, reset):
        """Phase 2: KV-cache Transformer step with (stoch, action) -> h_t.

        Args:
            carry: dict with kv_cache, pos, h_prev.
            stoch: (B, S, K) posterior/imagined stochastic state.
            action: (B, A) action taken at current step.
            reset: (B,) boolean.
        Returns:
            Updated carry dict with new h_prev = h_t.
        """
        carry = self._mask_carry(carry, reset)

        B = stoch.shape[0]
        D = self._deter
        H = self._n_heads
        D_head = self._d_head
        M = self._memory_size

        # Normalize action
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()

        # Input projection
        stoch_flat = stoch.reshape(B, self.flat_stoch)
        x_t = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))
        x_t = x_t.unsqueeze(1)  # (B, 1, D)

        kv_cache = carry['kv_cache']  # (B, L, 2, M, D)
        C = int(kv_cache.shape[3])
        if C != M:
            raise AssertionError("TransformerRSSM.update_carry expects a "
                                 f"memory cache of length {M}, got {C}.")
        pos = carry['pos']  # (B,)
        ts = pos.unsqueeze(1)  # (B, 1)

        # The carry contains only previous tokens. Add the current key/value
        # for this attention call, then store the shifted previous-token cache.
        n_valid = torch.clamp(pos, min=0, max=M)
        j = torch.arange(M, device=x_t.device)
        prev_mask = j.unsqueeze(0) >= (M - n_valid).unsqueeze(1)
        current_mask = torch.ones(B, 1, dtype=torch.bool, device=x_t.device)
        valid_mask = torch.cat([prev_mask, current_mask], dim=1)
        # (B, 1, 1, M+1) for scaled_dot_product_attention
        attn_mask = valid_mask.unsqueeze(1).unsqueeze(2)

        new_kv_cache_layers = []
        for i in range(self._n_layers):
            res = x_t
            x_t = self._attn_norms[i](x_t)

            # Project Q, K_new, V_new
            Q = self._q_projs[i](x_t).reshape(B, 1, H, D_head).transpose(
                1, 2)  # (B, H, 1, D_head)
            K_new = self._k_projs[i](x_t).reshape(B, 1, H,
                                                  D_head).transpose(1, 2)
            V_new = self._v_projs[i](x_t).reshape(B, 1, H,
                                                  D_head).transpose(1, 2)

            # Apply RoPE at current position
            Q = self._apply_rope(Q, ts)
            K_new = self._apply_rope(K_new, ts)

            # Flatten K_new, V_new to (B, 1, D) for cache storage
            k_new_flat = K_new.transpose(1, 2).reshape(B, 1, D)
            v_new_flat = V_new.transpose(1, 2).reshape(B, 1, D)

            k_all = torch.cat([kv_cache[:, i, 0], k_new_flat],
                              dim=1)  # (B, M+1, D)
            v_all = torch.cat([kv_cache[:, i, 1], v_new_flat], dim=1)

            if M > 0:
                k_cache = k_all[:, -M:]
                v_cache = v_all[:, -M:]
            else:
                k_cache = k_all[:, :0]
                v_cache = v_all[:, :0]
            new_kv_cache_layers.append((k_cache, v_cache))

            # Reshape K/V for attention: (B,M+1,D) -> (B,H,M+1,D_head)
            K_cached = k_all.reshape(B, M + 1, H, D_head).transpose(1, 2)
            V_cached = v_all.reshape(B, M + 1, H, D_head).transpose(1, 2)

            # Cross-attend Q to cached K/V
            x_t = F.scaled_dot_product_attention(Q,
                                                 K_cached,
                                                 V_cached,
                                                 attn_mask=attn_mask)
            x_t = x_t.transpose(1, 2).reshape(B, 1, D)
            x_t = res + self._o_projs[i](x_t)

            # FFN sublayer
            res = x_t
            x_t = self._ffn_norms[i](x_t)
            x_t = self._ff2s[i](self._act_fn(self._ff1s[i](x_t)))
            x_t = res + x_t

        x_t = self._outnorm(x_t)
        h_t = x_t[:, 0]  # (B, D)

        # Assemble updated KV cache
        ks = torch.stack([kv[0] for kv in new_kv_cache_layers],
                         dim=1)  # (B, L, M, D)
        vs = torch.stack([kv[1] for kv in new_kv_cache_layers], dim=1)
        new_kv_cache = torch.stack([ks, vs], dim=2)  # (B, L, 2, M, D)

        return {
            'kv_cache': new_kv_cache,
            'pos': pos + 1,
            'h_prev': h_t,
        }

    # ------------------------------------------------------------------
    # Shared methods
    # ------------------------------------------------------------------

    def prior(self, deter):
        """Compute prior distribution and sample stoch."""
        logit = self._prior_head(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        deter = self._feat_proj(deter)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(
            dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss
