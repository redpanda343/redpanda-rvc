import unittest

import torch
import torch.nn.functional as F

from rvc.train.speaker_embeddings import match_speaker_embedding_scale


class SpeakerEmbeddingScaleTests(unittest.TestCase):
    def test_matches_pretrained_mean_norm_and_preserves_directions(self):
        torch.manual_seed(1234)
        fresh = torch.randn(41, 256)
        pretrained = torch.randn(109, 256) * 0.25

        result = match_speaker_embedding_scale(fresh, pretrained)

        self.assertEqual(result.shape, fresh.shape)
        self.assertEqual(result.dtype, fresh.dtype)
        self.assertTrue(
            torch.allclose(
                result.norm(dim=1).mean(),
                pretrained.norm(dim=1).mean(),
            )
        )
        self.assertTrue(
            torch.allclose(
                F.normalize(result, dim=1),
                F.normalize(fresh, dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertEqual(torch.unique(result, dim=0).shape[0], fresh.shape[0])

    def test_is_deterministic_for_the_same_fresh_table(self):
        fresh = torch.randn(5, 256)
        pretrained = torch.randn(109, 256) * 0.25

        first = match_speaker_embedding_scale(fresh, pretrained)
        second = match_speaker_embedding_scale(fresh, pretrained)

        self.assertTrue(torch.equal(first, second))

    def test_supports_different_speaker_counts_and_float16(self):
        fresh = torch.randn(2, 256, dtype=torch.float16)
        pretrained = torch.randn(109, 256, dtype=torch.float16) * 0.25

        result = match_speaker_embedding_scale(fresh, pretrained)

        self.assertEqual(result.shape, (2, 256))
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(
            torch.allclose(
                result.float().norm(dim=1).mean(),
                pretrained.float().norm(dim=1).mean(),
                atol=1e-3,
                rtol=1e-3,
            )
        )

    def test_rejects_invalid_embedding_tables(self):
        cases = [
            (torch.empty(0, 256), torch.randn(109, 256), ValueError),
            (torch.zeros(2, 256), torch.randn(109, 256), ValueError),
            (torch.randn(2, 128), torch.randn(109, 256), ValueError),
            (
                torch.ones(2, 256, dtype=torch.int64),
                torch.randn(109, 256),
                TypeError,
            ),
        ]

        for fresh, pretrained, error in cases:
            with self.subTest(error=error):
                with self.assertRaises(error):
                    match_speaker_embedding_scale(fresh, pretrained)


if __name__ == "__main__":
    unittest.main()
