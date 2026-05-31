# Transformer-based RSSM (TransformerRSSM)

## Architecture overview

```text
Training (observe path):

  tokens (B,T,E) ─► post_head ─► post_logit ─► sample stoch (B,T,S,K)
                                                 │
  action (B,T,A) ────────────────────────────────┤
                                                 ▼
                               cat(flat(stoch), action_norm)
                                                 ▼
                           inp_proj ─► causal Transformer ─► h (B,T,D)
                                                 │
                                 shift-right: h_prev = [0, h[:,:-1]]
                                                 │
                                                 ▼
                                          prior_head(h_prev)
                                                 ▼
                                            prior_logit

  State at position t: (stoch_t, h_prev_t)
  Feature vector:       cat(flat(stoch_t), h_prev_t)
```

`h_prev_t` is zeroed on reset positions.

Posterior is conditioned on `tokens` only. It does **not** take `h_prev`.
