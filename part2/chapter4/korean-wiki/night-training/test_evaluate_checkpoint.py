"""Coverage, aggregation and fixed-target identity checks without a model."""
import unittest

from evaluate_checkpoint import comparison_rows, fixed_windows, metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_total_token_counts_and_nll(self):
        result = metrics([5, 10., 3, 2, 6., 1])
        self.assertEqual(result["loss"], 2.)
        self.assertEqual(result["content_loss"], 3.)
        self.assertEqual(result["accuracy"], .6)
        self.assertEqual(result["content_accuracy"], .5)

    def test_invalid_counts_losses_and_correct_counts_are_rejected(self):
        for totals in ([0, 0., 0, 1, 1., 0], [1.5, 1., 0, 1, 1., 0],
                       [1, float("nan"), 0, 1, 1., 0], [1, -1., 0, 1, 1., 0],
                       [1, 1., 2, 1, 1., 0], [1, 1., 0, 0, 0., 0]):
            with self.subTest(totals=totals), self.assertRaises(ValueError):
                metrics(totals)

    def test_wrong_fixed_validation_target_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "selection differs"):
            fixed_windows(20000, 42, "wrong-target-hash")

    def test_comparison_reads_real_historical_epochs_and_rejects_mixed_targets(self):
        score = {"targets": 16384, "loss": 3.}
        old = [{"epochs": epoch, "validation": score} for epoch in (2, 3, 4)]
        result = comparison_rows(old, 4.074373, score)
        self.assertEqual([row["epochs"] for row in result], [2, 3, 4, 4.074373])
        with self.assertRaisesRegex(ValueError, "different fixed"):
            comparison_rows(old, 4.074373, {"targets": 32768})


if __name__ == "__main__":
    unittest.main()
