import unittest
from types import SimpleNamespace

import torch

from rssm import TransformerRSSM


def _config():
    return SimpleNamespace(
        stoch=2,
        deter=8,
        discrete=3,
        unimix_ratio=0.01,
        device="cpu",
        n_heads=2,
        n_layers=2,
        d_ff=16,
        window_size=4,
        act="SiLU",
    )


class RSSMChunkedObserveTest(unittest.TestCase):

    def test_observe_with_carry_matches_full_observe_for_deterministic_path(
            self):
        torch.manual_seed(0)
        model = TransformerRSSM(_config(), embed_size=5, act_dim=2)
        B, T = 2, 7
        tokens = torch.randn(B, T, 5)
        action = torch.randn(B, T, 2)
        reset = torch.zeros(B, T, dtype=torch.bool)
        reset[:, 0] = True

        _, full = model.observe(tokens, action, reset, sample=False)

        carry = model.initial(B)
        chunks = []
        for start in range(0, T, 3):
            end = min(start + 3, T)
            _, feat, carry = model.observe_with_carry(
                tokens[:, start:end],
                action[:, start:end],
                reset[:, start:end],
                carry,
                sample=False,
            )
            chunks.append(feat["deter"])
        chunked_deter = torch.cat(chunks, dim=1)

        torch.testing.assert_close(chunked_deter,
                                   full["deter"],
                                   atol=1e-5,
                                   rtol=1e-5)

    def test_update_carry_sequence_matches_repeated_update_with_resets(self):
        torch.manual_seed(1)
        model = TransformerRSSM(_config(), embed_size=5, act_dim=2)
        B, T = 2, 6
        stoch_index = torch.randint(0, 3, (B, T, 2))
        stoch = torch.nn.functional.one_hot(stoch_index, num_classes=3).float()
        action = torch.randn(B, T, 2)
        reset = torch.zeros(B, T, dtype=torch.bool)
        reset[:, 0] = True
        reset[1, 3] = True

        step_carry = model.initial(B)
        step_h = []
        for t in range(T):
            step_carry = model.update_carry(step_carry, stoch[:, t],
                                            action[:, t], reset[:, t])
            step_h.append(step_carry["h_prev"])
        step_h = torch.stack(step_h, dim=1)

        seq_h, seq_carry = model.update_carry_sequence(model.initial(B), stoch,
                                                       action, reset)

        torch.testing.assert_close(seq_h, step_h, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(seq_carry["h_prev"],
                                   step_carry["h_prev"],
                                   atol=1e-5,
                                   rtol=1e-5)
        torch.testing.assert_close(seq_carry["pos"], step_carry["pos"])
        torch.testing.assert_close(seq_carry["kv_cache"],
                                   step_carry["kv_cache"],
                                   atol=1e-5,
                                   rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
