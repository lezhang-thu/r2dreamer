import torch
from torch import distributions as torchd
from torch import nn
from torch.nn import functional as F

import distributions as dists
from _positional_embeddings import Qwen2RotaryPositionalEmbeddings
from networks import BlockLinear, LambdaLayer
from tools import rpad, weight_init_


class Deter(nn.Module):

    def __init__(self,
                 deter,
                 stoch,
                 act_dim,
                 hidden,
                 blocks,
                 dynlayers,
                 act="SiLU"):
        super().__init__()
        self.blocks = int(blocks)
        self.dynlayers = int(dynlayers)
        act = getattr(torch.nn, act)
        self._dyn_in0 = nn.Sequential(
            nn.Linear(deter, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act())
        self._dyn_in1 = nn.Sequential(
            nn.Linear(stoch, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act())
        self._dyn_in2 = nn.Sequential(
            nn.Linear(act_dim, hidden, bias=True),
            nn.RMSNorm(hidden, eps=1e-04, dtype=torch.float32), act())
        self._dyn_hid = nn.Sequential()
        in_ch = (3 * hidden + deter // self.blocks) * self.blocks
        for i in range(self.dynlayers):
            self._dyn_hid.add_module(f"dyn_hid_{i}",
                                     BlockLinear(in_ch, deter, self.blocks))
            self._dyn_hid.add_module(
                f"norm_{i}", nn.RMSNorm(deter, eps=1e-04, dtype=torch.float32))
            self._dyn_hid.add_module(f"act_{i}", act())
            in_ch = deter
        self._dyn_gru = BlockLinear(in_ch, 3 * deter, self.blocks)
        self.flat2group = lambda x: x.reshape(*x.shape[:-1], self.blocks, -1)
        self.group2flat = lambda x: x.reshape(*x.shape[:-2], -1)

    def forward(self, stoch, deter, action):
        """Deterministic state transition (block-GRU style)."""
        # (B, S, K), (B, D), (B, A)
        B = action.shape[0]

        # Flatten stochastic state and normalize action magnitude.
        # (B, S*K)
        stoch = stoch.reshape(B, -1)
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # (B, U)
        x0 = self._dyn_in0(deter)
        x1 = self._dyn_in1(stoch)
        x2 = self._dyn_in2(action)

        # Concatenate projected inputs and broadcast over blocks.
        # (B, 3*U)
        x = torch.cat([x0, x1, x2], -1)
        # (B, G, 3*U)
        x = x.unsqueeze(-2).expand(-1, self.blocks, -1)

        # Combine per-block deterministic state with per-block inputs.
        # (B, G, D/G + 3*U) -> (B, D + 3*U*G)
        x = self.group2flat(torch.cat([self.flat2group(deter), x], -1))

        # (B, D)
        x = self._dyn_hid(x)
        # (B, 3*D)
        x = self._dyn_gru(x)

        # Split GRU-style gates block-wise.
        # (B, G, 3*D/G)
        gates = torch.chunk(self.flat2group(x), 3, dim=-1)

        # (B, D)
        reset, cand, update = (self.group2flat(x) for x in gates)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        # (B, D)
        return update * cand + (1 - update) * deter


class RSSM(nn.Module):

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self._dyn_layers = int(config.dyn_layers)
        self._blocks = int(config.blocks)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = Deter(
            self._deter,
            self.flat_stoch,
            act_dim,
            self._hidden,
            blocks=self._blocks,
            dynlayers=self._dyn_layers,
            act=config.act,
        )

        self._obs_net = nn.Sequential()
        #inp_dim = self._deter + embed_size
        inp_dim = embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(
                f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(
                f"obs_net_n_{i}",
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module(
            "obs_net_logit",
            nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self.
                                            _discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(
                f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(
                f"img_net_n_{i}",
                nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module(
            "img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self.
                                            _discrete)),
        )
        self.apply(weight_init_)

    def initial(self, batch_size):
        """Return an initial latent state."""
        # (B, D), (B, S, K)
        deter = torch.zeros(batch_size,
                            self._deter,
                            dtype=torch.float32,
                            device=self._device)
        stoch = torch.zeros(batch_size,
                            self._stoch,
                            self._discrete,
                            dtype=torch.float32,
                            device=self._device)
        return stoch, deter

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations."""
        # (B, T, E), (B, T, A), ((B, S, K), (B, D)) (B, T)
        L = action.shape[1]
        stoch, deter = initial
        stochs, deters, logits = [], [], []
        for i in range(L):
            # (B, S, K), (B, D), (B, S, K)
            stoch, deter, logit = self.obs_step(stoch, deter, action[:, i],
                                                embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset):
        """Single posterior step."""
        # (B, S, K), (B, D), (B, A), (B, E), (B,)
        stoch = torch.where(rpad(reset,
                                 stoch.dim() - int(reset.dim())),
                            torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset,
                                 deter.dim() - int(reset.dim())),
                            torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset,
                 prev_action.dim() - int(reset.dim())),
            torch.zeros_like(prev_action), prev_action)

        # Deterministic transition then posterior logits conditioned on embed.
        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, D + E)
        #x = torch.cat([deter, embed], dim=-1)
        x = embed
        # (B, S, K)
        logit = self._obs_net(x)

        # Sample discrete stochastic state via straight-through Gumbel-Softmax.
        # (B, S, K)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit

    def img_step(self, stoch, deter, prev_action):
        """Single prior step (no observation)."""

        # (B, D)
        deter = self._deter_net(stoch, deter, prev_action)
        # (B, S, K)
        stoch, _ = self.prior(deter)
        return stoch, deter

    def prior(self, deter):
        """Compute prior distribution parameters and sample stoch."""

        # (B, S, K)
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions):
        """Roll out prior dynamics given a sequence of actions."""
        # (B, S, K), (B, D), (B, T, A)
        L = actions.shape[1]
        stochs, deters = [], []
        for i in range(L):
            stoch, deter = self.img_step(stoch, deter, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
        # (B, T, S, K), (B, T, D)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        # (B, S, K), (B, D)
        # (B, S*K)
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
        # (B, S*K + D)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(
            dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        kld = dists.kl
        rep_loss = kld(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = kld(post_logit.detach(), prior_logit).sum(-1)
        # Clipped gradients are not backpropagated using torch.clip.
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)

        return dyn_loss, rep_loss


class TransformerRSSM(nn.Module):

    def __init__(self, config, embed_size, act_dim):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._discrete = int(config.discrete)
        self._unimix_ratio = float(config.unimix_ratio)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._n_heads = int(config.n_heads)
        self._n_layers = int(config.n_layers)
        self._d_ff = int(config.d_ff)
        self._window_size = int(config.window_size)
        act_fn = getattr(torch.nn, config.act)

        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter

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

    def observe(self, tokens, action, reset):
        """Windowed-causal full-sequence training path.

        Args:
            tokens: (B, T, E) encoder embeddings.
            action: (B, T, A) current actions a_t (not prev_action).
            reset: (B, T) boolean, True at episode start.
        Returns:
            entries: dict with 'deter' (B,T,D) and 'stoch' (B,T,S,K).
            feat: dict with deter, stoch, post_logit, prior_logit, and
                trajectory KV tensors for imagination starts.
        """
        # Normalize action magnitude
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()

        # Posterior: conditioned on tokens only
        post_logit = self._post_head(tokens)  # (B, T, S, K)
        stoch = self.get_dist(post_logit).rsample()  # (B, T, S, K)

        # Input projection: cat(stoch, action) -> d_model
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        x = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))

        # Causal Transformer forward
        h, kv = self._fwd(x, return_kv=True)  # (B, T, D)

        # Shift right: h_prev[t] = h[t-1], h_prev[0] = 0
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)

        # Zero h_prev at episode resets
        reset_mask = reset.unsqueeze(-1).float()  # (B, T, 1)
        h_prev = h_prev * (1.0 - reset_mask)

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

    def _apply_rope(self, x, positions=None):
        """Apply RoPE to (B, H, T, D_head) tensor."""
        # Qwen2 RoPE expects (B, T, H, D_head).
        x_t = x.transpose(1, 2)
        if positions is None:
            x_t = self._rope(x_t, input_pos=None)
        else:
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(x.shape[0], -1)
            x_t = self._rope(x_t, input_pos=positions.to(torch.long))
        return x_t.transpose(1, 2)

    def _fwd(self, x, return_kv=False):
        """Pre-norm causal Transformer with RoPE.

        Args:
            x: (B, T, D) input sequence.
        Returns:
            (B, T, D) transformed sequence.
        """
        B, T, D = x.shape
        H = self._n_heads
        D_head = self._d_head
        W = self._window_size
        # Windowed causal mask: each query can attend to at most W recent keys
        # (including itself), matching inference/imagination transition scope.
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        if W < T:
            local = torch.triu(causal, diagonal=-(W - 1))
        else:
            local = causal
        attn_mask = local.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
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
            Q = self._apply_rope(Q, positions=None)
            K = self._apply_rope(K, positions=None)
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

        out = self._outnorm(x)
        if return_kv:
            return out, {
                'k': torch.stack(k_layers, dim=1),  # (B, L, T, D)
                'v': torch.stack(v_layers, dim=1),  # (B, L, T, D)
            }
        return out

    # ------------------------------------------------------------------
    # Imagination (KV-cache transition)
    # ------------------------------------------------------------------

    def img_step(self, stoch, deter, action):
        """Fallback single-step prior using zeroed KV carry."""
        carry = self.initial(stoch.shape[0])
        carry['h_prev'] = deter
        stoch, deter, _ = self.img_step_with_carry(stoch, carry, action)
        return stoch, deter

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
    def build_imag_starts(self, stoch_seq, deter_seq, kv_k, kv_v, start,
                          length):
        """Build B*K imagination starts from trajectory KV tensors.

        Args:
            stoch_seq: (B, T, S, Kcat)
            deter_seq: (B, T, D) h_prev sequence
            kv_k: (B, L, T+W, D) cached keys with W prepended dummy slots
            kv_v: (B, L, T+W, D) cached values with W prepended dummy slots
            start: int start index s0
            length: int number of contiguous starts K
        Returns:
            start_stoch: (B*K, S, Kcat)
            start_deter: (B*K, D)
            carry: dict with kv_cache (B*K,L,2,W,D), pos (B*K), h_prev (B*K,D)
        """
        B, T = stoch_seq.shape[:2]
        L = kv_k.shape[1]
        W = self._window_size
        start = int(start)
        length = int(length)
        assert 0 <= start < T
        assert length >= 1 and start + length <= T
        assert kv_k.shape == (B, L, T + W, self._deter)
        assert kv_v.shape == (B, L, T + W, self._deter)

        stoch_list, deter_list, cache_list, pos_list = [], [], [], []

        for s in range(start, start + length):
            # Right-aligned window over history [s-W, ..., s-1].
            # With W leading dummy slots, this is simply [s, ..., s+W-1].
            k_slice = kv_k[:, :, s:s + W, :]  # (B, L, W, D)
            v_slice = kv_v[:, :, s:s + W, :]  # (B, L, W, D)
            kv_cache = torch.stack([k_slice, v_slice], dim=2)  # (B,L,2,W,D)

            stoch_list.append(stoch_seq[:, s])
            deter_list.append(deter_seq[:, s])
            cache_list.append(kv_cache)
            pos_list.append(
                torch.full((B,), s, dtype=torch.int32, device=stoch_seq.device))

        start_stoch = torch.cat(stoch_list, dim=0)  # (B*K, S, Kcat)
        start_deter = torch.cat(deter_list, dim=0)  # (B*K, D)
        carry = {
            'kv_cache': torch.cat(cache_list, dim=0),  # (B*K,L,2,W,D)
            'pos': torch.cat(pos_list, dim=0).to(torch.int32),  # (B*K,)
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
        W = self._window_size

        # Normalize action
        action_norm = action / torch.clip(torch.abs(action), min=1.0).detach()

        # Input projection
        stoch_flat = stoch.reshape(B, self.flat_stoch)
        x_t = self._inp_proj(torch.cat([stoch_flat, action_norm], -1))
        x_t = x_t.unsqueeze(1)  # (B, 1, D)

        kv_cache = carry['kv_cache']  # (B, L, 2, W, D)
        pos = carry['pos']  # (B,)
        ts = pos.unsqueeze(1)  # (B, 1)

        # Valid mask: min(pos+1, W) rightmost entries are valid
        n_valid = torch.clamp(pos + 1, max=W)
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
            'h_prev': h_t,
        }

    # ------------------------------------------------------------------
    # Shared methods (same interface as GRU RSSM)
    # ------------------------------------------------------------------

    def prior(self, deter):
        """Compute prior distribution and sample stoch."""
        logit = self._prior_head(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter."""
        stoch = stoch.reshape(*stoch.shape[:-2], self._stoch * self._discrete)
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
