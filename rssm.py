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
        self._window_size = self._memory_size + self._segment_length
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
            getattr(config, 'rope_max_seq_len', max(self._window_size, 4096)))
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
                sample=True,
                positions=None,
                valid=None,
                memory_carry=None):
        """Transformer training path.

        Args:
            tokens: (B, T, E) encoder embeddings.
            action: (B, T, A) current actions a_t (not prev_action).
            reset: (B, T) boolean, True at episode start.
            sample: Whether to sample posterior stochastics. When False,
                uses the posterior mode, which is useful for deterministic
                cached reference trajectories.
            positions: Optional absolute episode positions, shape (B, T).
            valid: Optional bool mask for real non-padding timesteps, shape
                (B, T). Invalid keys are hidden from attention.
            memory_carry: Optional detached Transformer-XL carry for segment
                training without replay-side prefix tokens.
        Returns:
            entries: dict with 'deter' (B,T,D) and 'stoch' (B,T,S,K).
            feat: dict with deter, stoch, post_logit, prior_logit, and
                trajectory KV tensors for imagination starts.
        """
        if memory_carry is not None:
            return self._observe_with_carry(tokens,
                                            action,
                                            reset,
                                            memory_carry,
                                            sample=sample,
                                            positions=positions,
                                            valid=valid)

        # Normalize action magnitude
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()

        # Posterior: conditioned on tokens only
        post_logit = self._post_head(tokens)  # (B, T, S, K)
        post_dist = self.get_dist(post_logit)
        if sample:
            stoch = post_dist.rsample()  # (B, T, S, K)
        else:
            stoch = post_dist.base_dist.mode

        # Input projection: cat(stoch, action) -> d_model
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        x = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))

        # Causal Transformer forward
        h, kv = self._fwd(x, return_kv=True, positions=positions,
                          valid=valid, reset=reset)  # (B, T, D)

        # Shift right: h_prev[t] = h[t-1], h_prev[0] = 0
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)

        # Zero h_prev at episode resets
        reset_mask = reset.unsqueeze(-1).float()  # (B, T, 1)
        h_prev = h_prev * (1.0 - reset_mask)
        if valid is not None:
            h_prev = h_prev * valid.unsqueeze(-1).to(h_prev.dtype)

        # Prior: conditioned on h_prev only
        prior_logit = self._prior_head(h_prev)  # (B, T, S, K)

        entries = {'deter': h_prev, 'stoch': stoch}
        B, L, T, D = kv['k'].shape
        dummy_k = torch.zeros(B,
                              L,
                              self._window_size,
                              D,
                              dtype=kv['k'].dtype,
                              device=kv['k'].device)
        dummy_v = torch.zeros(B,
                              L,
                              self._window_size,
                              D,
                              dtype=kv['v'].dtype,
                              device=kv['v'].device)
        feat = {
            'deter': h_prev,
            'stoch': stoch,
            'post_logit': post_logit,
            'prior_logit': prior_logit,
            # Detached to avoid expanding the autograd graph.
            # Prepend W dummy zero-KV slots to match initial() cache style.
            'kv_k': torch.cat([dummy_k, kv['k']],
                              dim=2).detach(),  # (B, L, T+W, D)
            'kv_v': torch.cat([dummy_v, kv['v']],
                              dim=2).detach(),  # (B, L, T+W, D)
        }
        return entries, feat

    def _observe_with_carry(self,
                            tokens,
                            action,
                            reset,
                            carry,
                            sample=True,
                            positions=None,
                            valid=None):
        """Observe one real segment using detached Transformer-XL memory."""
        B, T = tokens.shape[:2]
        if positions is None:
            positions = torch.arange(T, device=tokens.device,
                                     dtype=torch.long).unsqueeze(0).expand(B, -1)
        else:
            positions = positions.to(device=tokens.device, dtype=torch.long)
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=tokens.device)
        else:
            valid = valid.to(device=tokens.device, dtype=torch.bool)

        carry = self._mask_carry(carry, reset[:, 0])
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
            x, carry, positions, reset, valid)

        h_prev = torch.cat([carry['h_prev'].unsqueeze(1), h[:, :-1]], dim=1)
        h_prev = h_prev * (1.0 - reset.unsqueeze(-1).float())
        h_prev = h_prev * valid.unsqueeze(-1).to(h_prev.dtype)
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

    def _fwd(self, x, return_kv=False, positions=None, valid=None, reset=None):
        """Pre-norm causal Transformer with RoPE.

        Args:
            x: (B, T, D) input sequence.
            positions: Optional absolute episode positions, shape (B, T).
            valid: Optional bool mask for real non-padding timesteps, shape
                (B, T). Invalid keys are hidden from attention.
            reset: Optional bool mask for episode starts, shape (B, T).
        Returns:
            (B, T, D) transformed sequence.
        """
        B, T, D = x.shape
        H = self._n_heads
        D_head = self._d_head
        W = self._window_size
        # Windowed causal mask by episode position, not just sequence index.
        # This lets a training row contain multiple complete episodes while
        # preventing attention across episode boundaries.
        if positions is None:
            pos = torch.arange(T, device=x.device,
                               dtype=torch.long).unsqueeze(0).expand(B, -1)
        else:
            pos = positions.to(device=x.device, dtype=torch.long)
        if reset is None:
            episode = torch.zeros(B, T, dtype=torch.long, device=x.device)
        else:
            episode = reset.to(device=x.device, dtype=torch.long).cumsum(dim=1)
        q_pos = pos.unsqueeze(2)
        k_pos = pos.unsqueeze(1)
        same_episode = episode.unsqueeze(2) == episode.unsqueeze(1)
        attn_mask = same_episode & (k_pos <= q_pos) & ((q_pos - k_pos) < W)
        attn_mask = attn_mask.unsqueeze(1)  # (B, 1, T, T)
        valid_f = None
        if valid is not None:
            valid = valid.to(dtype=torch.bool, device=x.device)
            valid_f = valid.unsqueeze(-1).to(dtype=x.dtype)
            x = x * valid_f
            attn_mask = attn_mask & valid.unsqueeze(1).unsqueeze(2)
            invalid_query = (~valid).unsqueeze(1).unsqueeze(-1)
            diag = torch.eye(T, dtype=torch.bool,
                             device=x.device).unsqueeze(0).unsqueeze(0)
            attn_mask = attn_mask | (invalid_query & diag)
        if return_kv:
            k_layers = []
            v_layers = []

        for i in range(self._n_layers):
            # Self-attention sublayer (pre-norm)
            res = x
            x = self._attn_norms[i](x)
            Q = self._q_projs[i](x).reshape(B, T, H, D_head).transpose(
                1, 2)  # (B, H, T, D_head)
            K = self._k_projs[i](x).reshape(B, T, H, D_head).transpose(1, 2)
            V = self._v_projs[i](x).reshape(B, T, H, D_head).transpose(1, 2)
            Q = self._apply_rope(Q, positions=positions)
            K = self._apply_rope(K, positions=positions)
            if return_kv:
                # Match cache storage layout used by update_carry.
                k_layers.append(K.transpose(1, 2).reshape(B, T, D))
                v_layers.append(V.transpose(1, 2).reshape(B, T, D))
            x = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_mask)  # (B, H, T, D_head)
            x = x.transpose(1, 2).reshape(B, T, D)
            x = res + self._o_projs[i](x)

            # FFN sublayer (pre-norm)
            res = x
            x = self._ffn_norms[i](x)
            x = self._ff2s[i](self._act_fn(self._ff1s[i](x)))
            x = res + x
            if valid_f is not None:
                x = x * valid_f

        out = self._outnorm(x)
        if valid_f is not None:
            out = out * valid_f
        if return_kv:
            return out, {
                'k': torch.stack(k_layers, dim=1),  # (B, L, T, D)
                'v': torch.stack(v_layers, dim=1),  # (B, L, T, D)
            }
        return out

    def _fwd_segment_with_carry(self, x, carry, positions, reset, valid):
        """Forward a segment against detached fixed-length TXL cache."""
        B, T, D = x.shape
        H = self._n_heads
        D_head = self._d_head
        W = self._window_size
        M = self._memory_size
        positions = positions.to(device=x.device, dtype=torch.long)
        reset = reset.to(device=x.device, dtype=torch.bool)
        valid = valid.to(device=x.device, dtype=torch.bool)

        # Memory keys represent the previous episode context only until a reset
        # appears inside the current segment.
        reset_count = reset.to(dtype=torch.long).cumsum(dim=1)
        n_mem = torch.clamp(carry['pos'].to(device=x.device), min=0, max=M)
        j_mem = torch.arange(W, device=x.device)
        mem_mask = j_mem.unsqueeze(0) >= (W - n_mem).unsqueeze(1)
        mem_mask = mem_mask[:, None, :] & (reset_count == 0)[:, :, None]

        episode = reset_count
        q_index = torch.arange(T, device=x.device).view(1, T, 1)
        k_index = torch.arange(T, device=x.device).view(1, 1, T)
        cur_mask = (k_index <= q_index) & (
            episode.unsqueeze(2) == episode.unsqueeze(1))
        cur_mask = cur_mask & valid.unsqueeze(1)
        attn_mask = torch.cat([mem_mask, cur_mask], dim=2).unsqueeze(1)
        valid_f = valid.unsqueeze(-1).to(x.dtype)
        x = x * valid_f

        k_layers = []
        v_layers = []
        new_cache_layers = []
        for i in range(self._n_layers):
            res = x
            x = self._attn_norms[i](x)
            Q = self._q_projs[i](x).reshape(B, T, H, D_head).transpose(1, 2)
            K_cur = self._k_projs[i](x).reshape(B, T, H,
                                                D_head).transpose(1, 2)
            V_cur = self._v_projs[i](x).reshape(B, T, H,
                                                D_head).transpose(1, 2)
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
            K_all = k_all.reshape(B, W + T, H, D_head).transpose(1, 2)
            V_all = v_all.reshape(B, W + T, H, D_head).transpose(1, 2)

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
            x = x * valid_f

            new_k = k_all[:, -W:]
            new_v = v_all[:, -W:]
            new_cache_layers.append((new_k, new_v))

        out = self._outnorm(x) * valid_f
        ks = torch.stack([kv[0] for kv in new_cache_layers], dim=1)
        vs = torch.stack([kv[1] for kv in new_cache_layers], dim=1)
        next_kv_cache = torch.stack([ks, vs], dim=2).detach()
        next_pos = (positions[:, -1] + 1).to(torch.int32)
        next_carry = {
            'kv_cache': next_kv_cache,
            'pos': next_pos,
            'seg_pos': torch.zeros(B, dtype=torch.int32, device=x.device),
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
            kv_k: (B, L, T+W, D) cached keys with W prepended dummy slots
            kv_v: (B, L, T+W, D) cached values with W prepended dummy slots
            starts: (B, K) integer start indices, all valid per episode
            positions: Optional (B, T) absolute episode positions for starts.
        Returns:
            start_stoch: (B*K, S, Kcat)
            start_deter: (B*K, D)
            carry: dict with kv_cache (B*K,L,2,W,D), pos, seg_pos, h_prev
        """
        B, T = stoch_seq.shape[:2]
        L = kv_k.shape[1]
        W = self._window_size
        starts = starts.to(device=stoch_seq.device, dtype=torch.long)
        assert starts.ndim == 2 and starts.shape[0] == B
        assert kv_k.shape == (B, L, T + W, self._deter)
        assert kv_v.shape == (B, L, T + W, self._deter)
        if positions is not None:
            assert positions.shape == (B, T)
        assert torch.all(starts >= 0)
        assert torch.all(starts < T)

        K = starts.shape[1]
        D = self._deter
        batch_index = torch.arange(B, device=stoch_seq.device)[:, None]
        offsets = torch.arange(W, device=stoch_seq.device)

        start_stoch = stoch_seq[batch_index, starts]  # (B, K, S, Kcat)
        start_deter = deter_seq[batch_index, starts]  # (B, K, D)
        if positions is None:
            start_pos = starts
        else:
            start_pos = positions.to(device=stoch_seq.device,
                                     dtype=torch.long)[batch_index, starts]
        start_seg_pos = starts % max(1, self._segment_length)

        cache_list = []
        for k in range(K):
            s = starts[:, k]  # (B,)
            # Gather the W keys/values immediately before start s. kv_k/kv_v
            # are stored with W dummy slots prepended, so adding W to
            # [s-W, ..., s-1] simplifies to [s, ..., s+W-1].
            left_time = s[:, None] - W + offsets[None, :]  # (B, W)
            gather_time = left_time + W
            gather_index = gather_time[:, None, :, None].expand(B, L, W, D)
            k_slice = kv_k.gather(2, gather_index)  # (B, L, W, D)
            v_slice = kv_v.gather(2, gather_index)  # (B, L, W, D)
            cache_list.append(torch.stack([k_slice, v_slice], dim=2))

        kv_cache = torch.stack(cache_list, dim=1)  # (B, K, L, 2, W, D)
        start_stoch = start_stoch.reshape(B * K, *stoch_seq.shape[2:])
        start_deter = start_deter.reshape(B * K, deter_seq.shape[-1])
        carry = {
            'kv_cache': kv_cache.reshape(B * K, L, 2, W, D),
            'pos': start_pos.reshape(B * K).to(torch.int32),
            'seg_pos': start_seg_pos.reshape(B * K).to(torch.int32),
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

    def initial(self, batch_size):
        """Return initial carry dict for KV-cache inference."""
        D = self._deter
        W = self._window_size
        return {
            'kv_cache':
                torch.zeros(batch_size,
                            self._n_layers,
                            2,
                            W,
                            D,
                            dtype=torch.float32,
                            device=self._device),
            'pos':
                torch.zeros(batch_size, dtype=torch.int32, device=self._device),
            'seg_pos':
                torch.zeros(batch_size, dtype=torch.int32, device=self._device),
            'h_prev':
                torch.zeros(batch_size,
                            D,
                            dtype=torch.float32,
                            device=self._device),
        }

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
        seg_pos = carry.get('seg_pos', torch.zeros_like(pos)) * (~reset).int()
        return {
            'kv_cache': kv_cache,
            'pos': pos,
            'seg_pos': seg_pos,
            'h_prev': h_prev
        }

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
        W = self._window_size

        # Normalize action
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()

        # Input projection
        stoch_flat = stoch.reshape(B, self.flat_stoch)
        x_t = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))
        x_t = x_t.unsqueeze(1)  # (B, 1, D)

        kv_cache = carry['kv_cache']  # (B, L, 2, W, D)
        pos = carry['pos']  # (B,)
        seg_pos = carry.get('seg_pos', torch.zeros_like(pos))
        ts = pos.unsqueeze(1)  # (B, 1)

        # Literal cache length is memory_size + segment_length. At the start
        # of a segment, only memory_size + current token is visible; visibility
        # grows until the segment ends, then falls back on the next segment.
        segment_visible = self._memory_size + seg_pos + 1
        n_valid = torch.minimum(torch.clamp(pos + 1, max=W),
                                torch.clamp(segment_visible, max=W))
        j = torch.arange(W, device=x_t.device)
        # (B, W) bool — True = can attend
        valid_mask = (j.unsqueeze(0) >= (W - n_valid).unsqueeze(1))
        # (B, 1, 1, W) for scaled_dot_product_attention
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

            # Shift cache left by 1, append new K/V
            k_cache = torch.cat([kv_cache[:, i, 0, 1:], k_new_flat],
                                dim=1)  # (B, W, D)
            v_cache = torch.cat([kv_cache[:, i, 1, 1:], v_new_flat], dim=1)
            new_kv_cache_layers.append((k_cache, v_cache))

            # Reshape cached K/V for attention: (B, W, D) -> (B, H, W, D_head)
            K_cached = k_cache.reshape(B, W, H, D_head).transpose(1, 2)
            V_cached = v_cache.reshape(B, W, H, D_head).transpose(1, 2)

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
                         dim=1)  # (B, L, W, D)
        vs = torch.stack([kv[1] for kv in new_kv_cache_layers], dim=1)
        new_kv_cache = torch.stack([ks, vs], dim=2)  # (B, L, 2, W, D)

        return {
            'kv_cache': new_kv_cache,
            'pos': pos + 1,
            'seg_pos': (seg_pos + 1) % self._segment_length,
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
