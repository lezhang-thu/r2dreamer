import unittest

import torch

from dreamer import Dreamer


class _ModeDist:

    def __init__(self, value):
        self._value = value

    def mode(self):
        return self._value


class _MeanDist:

    def __init__(self, mean):
        self.mean = mean


class _PolicyDist:

    def __init__(self, feat):
        self._feat = feat

    def log_prob(self, action):
        return torch.zeros(action.shape[:-1],
                           dtype=action.dtype,
                           device=action.device)

    def entropy(self):
        return torch.zeros(self._feat.shape[:-1],
                           dtype=self._feat.dtype,
                           device=self._feat.device)


class _ValueDist:

    def log_prob(self, target):
        return torch.zeros(target.shape[:-1],
                           dtype=target.dtype,
                           device=target.device)


class _RewardHead:

    def __init__(self, reward):
        self._reward = reward

    def __call__(self, feat):
        del feat
        return _ModeDist(self._reward)


class _ContHead:

    def __init__(self, cont):
        self._cont = cont

    def __call__(self, feat):
        del feat
        return _MeanDist(self._cont)


class _FrozenValueHead:

    def __init__(self, value):
        self._value = value

    def __call__(self, feat):
        del feat
        return _ModeDist(self._value)


class _ActorHead:

    def __call__(self, feat):
        return _PolicyDist(feat)


class _ValueHead:

    def __call__(self, feat):
        del feat
        return _ValueDist()


class _ReturnEMADummy:

    def __init__(self):
        self.ema_vals = (
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )

    def __call__(self, ret):
        del ret
        return (
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )


class _ShapingDummy:
    _actor_critic_forward = Dreamer._actor_critic_forward
    _masked_mean = staticmethod(Dreamer._masked_mean)

    def __init__(self):
        self.imag_horizon = 2
        self.horizon = 10
        self.lamb = 0.95
        self.act_entropy = 0.0
        self.expert_shaping_scale = 0.5
        self.memory = {"reward": torch.tensor([0.0], dtype=torch.float32)}
        self.return_ema = _ReturnEMADummy()

        self._imag_feat = torch.zeros(1, 3, 5, dtype=torch.float32)
        self._imag_action = torch.zeros(1, 3, 2, dtype=torch.float32)
        self._imag_deter = torch.zeros(1, 3, 4, dtype=torch.float32)
        self._imag_reward = torch.tensor([[[5.0], [1.0], [2.0]]],
                                         dtype=torch.float32)
        self._imag_cont = torch.tensor([[[1.0], [0.4], [0.25]]],
                                       dtype=torch.float32)
        self._imag_value = torch.zeros(1, 3, 1, dtype=torch.float32)
        self._phi = torch.tensor([[[10.0], [20.0], [40.0]]],
                                 dtype=torch.float32)

        self._frozen_reward = _RewardHead(self._imag_reward)
        self._frozen_cont = _ContHead(self._imag_cont)
        self._frozen_value = _FrozenValueHead(self._imag_value)
        self._frozen_slow_value = _FrozenValueHead(self._imag_value)
        self.actor = _ActorHead()
        self.value = _ValueHead()

        self.captured_reward = None

    def _imagine(self, start, imag_horizon, imag_carry):
        del start, imag_carry
        assert imag_horizon == self.imag_horizon + 1
        return self._imag_feat, self._imag_action, self._imag_deter

    def _read_memory(self, imag_deter, frozen=True):
        del imag_deter, frozen
        return {
            "raw_rtg": self._phi,
        }

    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        del last, term, value, boot, disc, lamb
        self.captured_reward = reward.detach().clone()
        return torch.zeros(reward.shape[0],
                           reward.shape[1] - 1,
                           1,
                           dtype=reward.dtype,
                           device=reward.device)


class PotentialBasedShapingTest(unittest.TestCase):

    def test_actor_critic_uses_effective_discount_when_imag_cont_is_fractional(
            self):
        agent = _ShapingDummy()

        start_stoch = torch.zeros(1, 1, 1, dtype=torch.float32)
        start_deter = torch.zeros(1, 4, dtype=torch.float32)
        imag_carry = {"kv_cache": None, "pos": None, "h_prev": None}
        imag_mask = torch.ones(1, 2, 1, dtype=torch.float32)

        _, metrics = agent._actor_critic_forward(start_stoch, start_deter,
                                                 imag_carry, imag_mask)

        disc = 1.0 - 1.0 / agent.horizon
        gamma_eff = agent._imag_cont[:, 1:] * disc
        expected_shaping = gamma_eff * agent._phi[:, 1:] - agent._phi[:, :-1]
        expected_reward = torch.cat(
            [
                agent._imag_reward[:, :1],
                agent._imag_reward[:, 1:] +
                agent.expert_shaping_scale * expected_shaping,
            ],
            dim=1,
        )

        torch.testing.assert_close(agent.captured_reward, expected_reward)
        torch.testing.assert_close(metrics["shaping"], expected_shaping.mean())
        torch.testing.assert_close(metrics["rew"], expected_reward.mean())


if __name__ == "__main__":
    unittest.main()
