import unittest
from types import SimpleNamespace

import torch

from networks import ConvEncoder, MultiEncoder


def _cnn_config(depth=8, mults=(2, 3, 4, 4)):
    return SimpleNamespace(
        act="SiLU",
        norm=True,
        kernel_size=3,
        stem_stride=2,
        minres=4,
        depth=depth,
        mults=list(mults),
        blocks=[1, 1, 1, 1],
    )


def _encoder_config():
    return SimpleNamespace(
        cnn_keys="image",
        mlp_keys="$^",
        cnn=_cnn_config(),
    )


class ConvEncoderTest(unittest.TestCase):

    def test_multi_encoder_uses_conv_encoder_for_image_inputs(self):
        encoder = MultiEncoder(_encoder_config(), {"image": (64, 64, 3)})

        self.assertEqual(len(encoder.encoders), 1)
        self.assertIsInstance(encoder.encoders[0], ConvEncoder)

    def test_conv_encoder_preserves_expected_output_shape(self):
        encoder = ConvEncoder(_cnn_config(), (64, 64, 3))
        obs = torch.randn(2, 5, 64, 64, 3)

        out = encoder(obs)

        self.assertEqual(out.shape, (2, 5, encoder.out_dim))
        self.assertEqual(encoder.out_dim, 32 * 4 * 4)

    def test_conv_encoder_matches_depth_64_atari_output_dim(self):
        encoder = ConvEncoder(_cnn_config(depth=64), (64, 64, 3))

        self.assertEqual(encoder.out_dim, 256 * 4 * 4)

    def test_conv_encoder_supports_multiple_blocks_per_stage(self):
        config = _cnn_config(depth=16)
        config.blocks = [1, 1, 2, 2]
        encoder = ConvEncoder(config, (64, 64, 3))
        obs = torch.randn(2, 3, 64, 64, 3)

        out = encoder(obs)

        self.assertEqual(out.shape, (2, 3, encoder.out_dim))
        self.assertEqual(encoder.out_dim, 64 * 4 * 4)

    def test_conv_encoder_backward_pass_produces_gradients(self):
        encoder = ConvEncoder(_cnn_config(), (64, 64, 3))
        obs = torch.randn(2, 3, 64, 64, 3)

        loss = encoder(obs).mean()
        loss.backward()

        self.assertTrue(
            any(param.grad is not None for param in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
