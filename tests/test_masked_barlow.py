import unittest

import torch

from dreamer import Dreamer


class MaskedBarlowTest(unittest.TestCase):

    def test_masked_barlow_ignores_padded_rows(self):
        lambd = 0.05
        valid_x1 = torch.tensor([[1.0, 2.0], [3.0, 0.0], [2.0, 1.0]])
        valid_x2 = torch.tensor([[0.5, 1.5], [2.5, -0.5], [1.5, 0.5]])
        valid_mask = torch.ones(valid_x1.shape[0], 1)

        padded_x1 = torch.cat(
            [valid_x1, torch.tensor([[50.0, -30.0], [-40.0, 25.0]])], dim=0)
        padded_x2 = torch.cat(
            [valid_x2, torch.tensor([[80.0, 70.0], [-60.0, -55.0]])], dim=0)
        padded_mask = torch.tensor([[1.0], [1.0], [1.0], [0.0], [0.0]])

        valid_loss = Dreamer._masked_barlow_loss(valid_x1, valid_x2, valid_mask,
                                                 lambd)
        padded_loss = Dreamer._masked_barlow_loss(padded_x1, padded_x2,
                                                  padded_mask, lambd)

        torch.testing.assert_close(padded_loss, valid_loss)


if __name__ == "__main__":
    unittest.main()
