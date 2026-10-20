"""Independent small-corpus checks for exact vocabulary/retention analysis.

The oracle below uses actual top-K sets and paragraph lists. It intentionally
does not use the production rank histograms or frequency-cutoff shortcuts.
"""

from __future__ import annotations

from collections import Counter
from itertools import islice
from pathlib import Path
import random
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


FIXTURE_PARAGRAPHS = [
    [["a/NNG", "b/NNG", "./SF"], ["a/NNG", "rare/NNP", "./SF"]],
    [["a/NNG", "a/NNG", "./SF"], ["c/NNG", "./SF"]],
]


def independently_expand(token: str) -> tuple[str, ...]:
    """Lecture numeric behavior, written without the production helper."""
    form, separator, tag = token.rpartition("/")
    if not separator or not form or tag not in {"SN", "W_SERIAL"}:
        return (token,)
    form = "".join(chr(ord(char) - 0xFEE0) if "０" <= char <= "９" else char
                   for char in form)
    return tuple(f"{char}/{'SN' if char.isdigit() else 'SW'}" for char in form)


def expanded_paragraphs(paragraphs: list) -> list:
    return [[[piece for token in sentence for piece in independently_expand(token)]
             for sentence in paragraph] for paragraph in paragraphs]


def count_paragraphs(paragraphs: list) -> Counter:
    return Counter(token for paragraph in paragraphs for sentence in paragraph for token in sentence)


def brute_force_metrics(paragraphs: list, counts: Counter, k: int) -> dict:
    """Exact top-(K-4) membership, independent of histogram/CDF algorithms."""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    allowed = {token for token, _ in ranked[:max(0, k - 4)]}
    sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
    kept_per_paragraph = [[all(token in allowed for token in sentence)
                           for sentence in paragraph] for paragraph in paragraphs]
    kept_sentences = [sentence for sentence in sentences
                      if all(token in allowed for token in sentence)]
    total_tokens = sum(counts.values())
    covered_tokens = sum(count for token, count in counts.items() if token in allowed)
    kept_tokens = sum(len(sentence) for sentence in kept_sentences)
    return {
        "sentences": len(sentences), "paragraphs": len(paragraphs),
        "tokens": total_tokens, "covered_tokens": covered_tokens,
        "sentences_kept": len(kept_sentences),
        "paragraphs_any_kept": sum(any(flags) for flags in kept_per_paragraph),
        "paragraphs_all_kept": sum(all(flags) for flags in kept_per_paragraph),
        "retained_sentence_tokens": kept_tokens,
        "token_coverage_percent": covered_tokens / total_tokens * 100 if total_tokens else 0,
        "sentence_retention_percent": len(kept_sentences) / len(sentences) * 100 if sentences else 0,
    }


def real_sample(max_source_lines: int = 160, max_sentences: int = 100) -> list:
    """Read a bounded immutable Step-5 sample; this never scans the corpus."""
    source = ROOT / "tokens" / "part_00.txt"
    if not source.is_file():
        raise unittest.SkipTest("The first completed Kiwi shard is not available yet")
    paragraphs, current, sentence_count = [], [], 0
    with source.open(encoding="utf-8") as stream:
        for raw in islice(stream, max_source_lines):
            line = raw.strip()
            if not line:
                if current:
                    paragraphs.append(current)
                    current = []
                continue
            if sentence_count == max_sentences:
                break
            current.append(line.split(" "))
            sentence_count += 1
    if current:
        paragraphs.append(current)
    if not paragraphs:
        raise unittest.SkipTest("No tokenized sample sentences are available")
    return expanded_paragraphs(paragraphs)


class VocabularyCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import analyze_vocabulary_coverage as coverage
        cls.coverage = coverage

    def analyze(self, paragraphs: list, counts: Counter | None = None, splitter=None):
        counts = count_paragraphs(paragraphs) if counts is None else counts
        ranks = self.coverage.build_rank_map(counts)
        ranked = self.coverage.rank_counts(counts)
        hist = self.coverage.analyze_paragraphs(paragraphs, ranks, splitter=splitter)
        return hist, ranked

    def assert_metrics_match(self, actual: dict, expected: dict):
        mappings = {
            "covered_tokens": "covered_tokens",
            "kept_sentences": "sentences_kept",
            "lecture_kept_paragraphs": "paragraphs_any_kept",
            "strict_kept_paragraphs": "paragraphs_all_kept",
            "retained_training_tokens": "retained_sentence_tokens",
            "token_coverage_percent": "token_coverage_percent",
            "sentence_keep_percent": "sentence_retention_percent",
        }
        for actual_key, expected_key in mappings.items():
            with self.subTest(metric=actual_key):
                self.assertAlmostEqual(actual[actual_key], expected[expected_key], places=9)
        self.assertEqual(actual["removed_sentences"],
                         expected["sentences"] - expected["sentences_kept"])
        self.assertEqual(actual["lecture_removed_paragraphs"],
                         expected["paragraphs"] - expected["paragraphs_any_kept"])
        self.assertEqual(actual["strict_removed_paragraphs"],
                         expected["paragraphs"] - expected["paragraphs_all_kept"])
        self.assertEqual(actual["lost_training_tokens"],
                         expected["tokens"] - expected["retained_sentence_tokens"])

    def test_rank_is_one_based_frequency_descending_lexical_ties(self):
        counts = Counter({"차/NNG": 2, "a/SL": 2, "A/SL": 2, "<sp>/WS": 4})
        expected = [("<sp>/WS", 4), ("A/SL", 2), ("a/SL", 2), ("차/NNG", 2)]
        self.assertEqual(self.coverage.rank_counts(counts), expected)
        self.assertEqual(self.coverage.build_rank_map(counts),
                         {token: rank for rank, (token, _) in enumerate(expected, 1)})

    def test_k_includes_four_reserved_slots(self):
        hist, ranked = self.analyze(FIXTURE_PARAGRAPHS)
        for k in range(4, len(ranked) + 5):
            point = self.coverage.metrics_at_k(hist, ranked, k)
            expected = brute_force_metrics(FIXTURE_PARAGRAPHS,
                                            count_paragraphs(FIXTURE_PARAGRAPHS), k)
            self.assert_metrics_match(point, expected)
        self.assertEqual(self.coverage.metrics_at_k(hist, ranked, 4)["kept_sentences"], 0)
        self.assertEqual(self.coverage.metrics_at_k(hist, ranked, 9)["kept_sentences"], 4)

    def test_equal_frequency_boundary_does_not_expand_beyond_exact_top_k(self):
        hist, ranked = self.analyze(FIXTURE_PARAGRAPHS)
        counts = count_paragraphs(FIXTURE_PARAGRAPHS)
        point = self.coverage.metrics_at_k(hist, ranked, 7)
        cutoff = ranked[7 - 4 - 1][1]
        approximate_allowed = {token for token, count in counts.items() if count >= cutoff}
        approximate_kept = sum(all(token in approximate_allowed for token in sentence)
                               for paragraph in FIXTURE_PARAGRAPHS for sentence in paragraph)
        self.assertEqual(approximate_kept, 4)
        self.assertEqual(point["kept_sentences"], 2)
        self.assertLess(point["kept_sentences"], approximate_kept)

    def test_any_surviving_sentence_and_whole_paragraph_retention_are_distinct(self):
        hist, ranked = self.analyze(FIXTURE_PARAGRAPHS)
        point = self.coverage.metrics_at_k(hist, ranked, 7)
        self.assertEqual(point["kept_sentences"], 2)
        self.assertEqual(point["lecture_kept_paragraphs"], 2)
        self.assertEqual(point["strict_kept_paragraphs"], 0)
        self.assertEqual(point["lecture_paragraph_keep_percent"], 100)
        self.assertEqual(point["strict_paragraph_keep_percent"], 0)
        self.assertEqual(point["strict_retained_training_tokens"], 0)
        self.assertEqual(point["strict_lost_training_tokens"], 11)

    def test_token_prefix_coverage_differs_from_tokens_in_retained_sentences(self):
        hist, ranked = self.analyze(FIXTURE_PARAGRAPHS)
        point = self.coverage.metrics_at_k(hist, ranked, 7)
        self.assertEqual(hist["total_tokens"], 11)
        self.assertEqual(point["covered_tokens"], 9)
        self.assertEqual(point["retained_training_tokens"], 6)
        self.assertAlmostEqual(point["token_coverage_percent"], 9 / 11 * 100)
        self.assertEqual(point["sentence_keep_percent"], 50)

    def test_number_expansion_happens_before_count_rank_and_sentence_oov(self):
        raw = [[["2010/SN", "년/NNB", "./SF"],
                ["２０１１/SN", "년/NNB", "./SF"]],
               [["2-1/W_SERIAL", "이/VCP", "다/EF", "./SF"]]]
        normalized = expanded_paragraphs(raw)
        counts = count_paragraphs(normalized)
        self.assertNotIn("2010/SN", counts)
        self.assertNotIn("２０１１/SN", counts)
        self.assertEqual(counts["0/SN"], 3)
        self.assertEqual(counts["1/SN"], 4)
        hist, ranked = self.analyze(raw, counts=counts, splitter=independently_expand)
        self.assertEqual(hist["total_tokens"], sum(counts.values()))
        for k in range(4, len(ranked) + 5):
            self.assert_metrics_match(self.coverage.metrics_at_k(hist, ranked, k),
                                      brute_force_metrics(normalized, counts, k))

    def test_histogram_required_k_includes_reserved_slots(self):
        hist, _ = self.analyze(FIXTURE_PARAGRAPHS)
        self.assertEqual(hist["total_sentences"], 4)
        self.assertEqual(hist["total_paragraphs"], 2)
        # Ranking is ./SF, a/NNG, b/NNG, c/NNG, rare/NNP.
        self.assertEqual(hist["sentence_required_k"], Counter({6: 1, 7: 1, 8: 1, 9: 1}))
        self.assertEqual(hist["paragraph_any_required_k"], Counter({6: 1, 7: 1}))
        self.assertEqual(hist["paragraph_all_required_k"], Counter({8: 1, 9: 1}))
        self.assertEqual(hist["sentence_tokens_by_required_k"], Counter({6: 3, 7: 3, 8: 2, 9: 3}))

    def test_exact_65000_and_65536_boundaries(self):
        keys = [f"tok{index:05d}/NNG" for index in range(66000)]
        boundary_indexes = {64995, 64996, 65531, 65532}
        paragraphs = [[[keys[64995]], [keys[64996]]],
                      [[keys[65531]], [keys[65532]]],
                      [[token for index, token in enumerate(keys) if index not in boundary_indexes]]]
        counts = count_paragraphs(paragraphs)
        hist, ranked = self.analyze(paragraphs, counts)
        for k in (65000, 65536):
            self.assert_metrics_match(self.coverage.metrics_at_k(hist, ranked, k),
                                      brute_force_metrics(paragraphs, counts, k))
        self.assertEqual(self.coverage.metrics_at_k(hist, ranked, 65000)["kept_sentences"], 1)
        self.assertEqual(self.coverage.metrics_at_k(hist, ranked, 65536)["kept_sentences"], 3)

    def test_dense_curve_monotonic_and_full_vocabulary_preserves_everything(self):
        rng = random.Random(123)
        keys = [f"단어{index}/NNG" for index in range(27)]
        paragraphs = [[[rng.choice(keys) for _ in range(rng.randint(1, 12))]
                       for _ in range(rng.randint(1, 5))] for _ in range(35)]
        counts = count_paragraphs(paragraphs)
        hist, ranked = self.analyze(paragraphs, counts)
        previous = None
        for k in range(4, len(ranked) + 5):
            point = self.coverage.metrics_at_k(hist, ranked, k)
            self.assert_metrics_match(point, brute_force_metrics(paragraphs, counts, k))
            if previous:
                for key in ("covered_tokens", "kept_sentences", "lecture_kept_paragraphs",
                            "strict_kept_paragraphs", "retained_training_tokens",
                            "strict_retained_training_tokens"):
                    self.assertGreaterEqual(point[key], previous[key])
            previous = point
        for key in ("token_coverage_percent", "sentence_keep_percent",
                    "lecture_paragraph_keep_percent", "strict_paragraph_keep_percent"):
            self.assertEqual(previous[key], 100)
        self.assertEqual(previous["retained_training_tokens"], sum(counts.values()))

    def test_bounded_real_source_matches_independent_brute_force(self):
        paragraphs = real_sample()
        sentence_count = sum(len(paragraph) for paragraph in paragraphs)
        self.assertLessEqual(sentence_count, 100)
        counts = count_paragraphs(paragraphs)
        hist, ranked = self.analyze(paragraphs, counts)
        selected = {4, 5, 10, len(ranked) // 4 + 4, len(ranked) // 2 + 4,
                    3 * len(ranked) // 4 + 4, len(ranked) + 4, 65000, 65536}
        for k in sorted(selected):
            with self.subTest(k=k):
                self.assert_metrics_match(self.coverage.metrics_at_k(hist, ranked, k),
                                          brute_force_metrics(paragraphs, counts, k))

    def test_curve_points_use_exact_same_top_k_metrics(self):
        hist, ranked = self.analyze(FIXTURE_PARAGRAPHS)
        selected = list(range(4, len(ranked) + 5))
        curve = self.coverage.build_curve(hist, ranked, ks=selected)
        self.assertEqual([point["k"] for point in curve], selected)
        for point in curve:
            self.assert_metrics_match(point, brute_force_metrics(
                FIXTURE_PARAGRAPHS, count_paragraphs(FIXTURE_PARAGRAPHS), point["k"]))

    def test_retention_targets_use_first_exact_k_that_meets_each_target(self):
        hist, ranked = self.analyze(FIXTURE_PARAGRAPHS)
        thresholds = self.coverage.retention_thresholds(hist, targets=(50, 75, 100))
        for kind, metric in (("sentence_keep", "sentence_keep_percent"),
                             ("lecture_paragraph_keep", "lecture_paragraph_keep_percent"),
                             ("strict_paragraph_keep", "strict_paragraph_keep_percent")):
            for threshold in thresholds[kind]:
                k, target = threshold["k"], threshold["target_percent"]
                self.assertGreaterEqual(self.coverage.metrics_at_k(hist, ranked, k)[metric], target)
                if k > 4:
                    self.assertLess(self.coverage.metrics_at_k(hist, ranked, k - 1)[metric], target)

    def test_knee_is_explicit_operational_heuristic_with_bounded_domain(self):
        points = [{"k": k, "sentence_keep_percent": value}
                  for k, value in [(16384, 27.9), (32768, 44.3),
                                   (49152, 53.2), (65536, 58.9)]]
        knee = self.coverage.choose_knee(points, low=16384, high=65536)
        self.assertGreaterEqual(knee["k"], 16384)
        self.assertLessEqual(knee["k"], 65536)
        self.assertTrue(knee["method"])
        self.assertTrue(knee["metric"])
        self.assertIs(knee["operational_only"], True)


if __name__ == "__main__":
    unittest.main()
