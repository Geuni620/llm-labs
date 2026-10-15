from collections import Counter
import unittest

from count_ngrams import count_ngrams, context_totals


class CountNgramsTests(unittest.TestCase):
    def test_unigram_uses_all_characters_as_denominator(self):
        counts = count_ngrams(["하루", "하세"], 1)
        self.assertEqual(counts, Counter({"하": 2, "루": 1, "세": 1}))
        self.assertEqual(context_totals(counts), Counter({"": 4}))

    def test_bigram_uses_matching_context_as_denominator(self):
        counts = count_ngrams(["하루", "하세", "하세", "가나"], 2)
        self.assertEqual(counts, Counter({"하루": 1, "하세": 2, "가나": 1}))
        totals = context_totals(counts)
        self.assertEqual(totals, Counter({"하": 3, "가": 1}))
        self.assertAlmostEqual(counts["하세"] / totals["하"], 2 / 3)

    def test_trigram_keeps_both_context_characters(self):
        counts = count_ngrams(["하세가", "하세나", "하루가"], 3)
        self.assertEqual(counts, Counter({"하세가": 1, "하세나": 1, "하루가": 1}))
        self.assertEqual(context_totals(counts), Counter({"하세": 2, "하루": 1}))

    def test_answers_are_not_joined(self):
        self.assertEqual(count_ngrams(["하", "루"], 2), Counter())

    def test_repeated_overlapping_pairs_are_counted(self):
        self.assertEqual(count_ngrams(["하하하"], 2), Counter({"하하": 2}))

    def test_spaces_and_punctuation_are_characters(self):
        self.assertEqual(count_ngrams(["가 가!"], 2), Counter({"가 ": 1, " 가": 1, "가!": 1}))

    def test_general_ngram_and_short_answers(self):
        self.assertEqual(count_ngrams(["하루가요", "하세가요", "네"], 4), Counter({"하루가요": 1, "하세가요": 1}))

    def test_n_must_be_positive(self):
        with self.assertRaises(ValueError):
            count_ngrams(["하루"], 0)


if __name__ == "__main__":
    unittest.main()
