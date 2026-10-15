from pathlib import Path
from collections import Counter
import random
import tempfile
import unittest

from generate import generate_batch, generate_greedy, generate_ngram, load_bigrams, load_ngram, sample_candidate


class GenerateTests(unittest.TestCase):
    def setUp(self):
        self.model = {
            "하": [("루", 1), ("세", 2)],
            "세": [("요", 3), ("상", 1)],
            "요": [(".", 1)],
        }

    def test_greedy_path_and_conditional_probabilities(self):
        result = generate_greedy(self.model, "하")
        self.assertEqual(result["text"], "하세요.")
        self.assertEqual(result["stop_reason"], "punctuation")
        self.assertEqual([step["context"] for step in result["steps"]], ["하", "세", "요"])
        self.assertEqual([step["text"] for step in result["steps"]], ["하세", "하세요", "하세요."])
        first = result["steps"][0]
        self.assertEqual(first["context_total"], 3)
        self.assertEqual(first["selected_count"], 2)
        self.assertAlmostEqual(first["selected_model_probability"], 2 / 3)
        self.assertAlmostEqual(result["steps"][1]["selected_model_probability"], 3 / 4)

    def test_last_character_is_context_for_long_prefix(self):
        result = generate_greedy(self.model, "바하")
        self.assertEqual(result["text"], "바하세요.")
        self.assertEqual(result["steps"][0]["context"], "하")

    def test_frequency_tie_keeps_input_order(self):
        result = generate_greedy({"하": [("루", 1), ("세", 1)]}, "하", 2)
        self.assertEqual(result["text"], "하루")

    def test_space_is_preserved_as_a_character(self):
        result = generate_greedy({"가": [(" ", 1)], " ": [("요", 1)], "요": [("!", 1)]}, "가")
        self.assertEqual(result["text"], "가 요!")
        self.assertEqual(result["steps"][1]["context"], " ")

    def test_loop_stops_at_total_length_including_prefix(self):
        result = generate_greedy({"하": [("하", 1)]}, "하", 4)
        self.assertEqual(result["text"], "하하하하")
        self.assertEqual(result["stop_reason"], "max_length")
        self.assertEqual(len(result["steps"]), 3)

    def test_unseen_initial_context(self):
        result = generate_greedy(self.model, "없")
        self.assertEqual(result["text"], "없")
        self.assertEqual(result["stop_reason"], "unseen_context")
        self.assertEqual(result["steps"], [])

    def test_dead_end_after_one_selected_character(self):
        result = generate_greedy({"하": [("세", 1)]}, "하")
        self.assertEqual(result["text"], "하세")
        self.assertEqual(result["stop_reason"], "unseen_context")

    def test_completed_prefix_is_not_extended(self):
        result = generate_greedy(self.model, "하.")
        self.assertEqual(result["text"], "하.")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["stop_reason"], "punctuation")

    def test_prefix_already_at_limit(self):
        result = generate_greedy(self.model, "하세", 2)
        self.assertEqual(result["text"], "하세")
        self.assertEqual(result["steps"], [])
        self.assertEqual(result["stop_reason"], "max_length")

    def test_punctuation_at_limit_is_a_completed_sentence(self):
        result = generate_greedy({"하": [("?", 1)]}, "하", 2)
        self.assertEqual(result["text"], "하?")
        self.assertEqual(result["stop_reason"], "punctuation")

    def test_invalid_prefix_or_limit(self):
        for prefix, limit in [("", 100), ("하", 0), ("하세", 1)]:
            with self.subTest(prefix=prefix, limit=limit), self.assertRaises(ValueError):
                generate_greedy(self.model, prefix, limit)

    def test_hangul_prefix_uses_same_nfc_units_as_corpus(self):
        result = generate_greedy(self.model, "하")
        self.assertEqual(result["prefix"], "하")
        self.assertEqual(result["text"], "하세요.")

    def test_csv_preserves_spaces_and_candidate_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text("context,next,count\n하,루,1\n하,세,1\n가, ,2\n", encoding="utf-8")
            self.assertEqual(load_bigrams(path), {"하": [("루", 1), ("세", 1)], "가": [(" ", 2)]})

    def test_csv_rejects_invalid_bigram_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            for rows in ["하,세,0\n", "하세,요,1\n", "하,세,1\n하,세,2\n"]:
                with self.subTest(rows=rows):
                    path.write_text("context,next,count\n" + rows, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_bigrams(path)

    def test_current_corpus_matches_the_planned_three_steps(self):
        result = generate_greedy(load_bigrams(), "하")
        self.assertEqual(result["text"], "하세요.")
        self.assertEqual(result["stop_reason"], "punctuation")
        self.assertEqual([step["selected_count"] for step in result["steps"]], [104, 854, 1989])
        self.assertEqual([step["context_total"] for step in result["steps"]], [559, 881, 2140])


class GeneralNgramTests(unittest.TestCase):
    def test_requested_orders_use_the_whole_context_and_append_one_character(self):
        cases = [
            (5, "가나다라", {"가나다라": [("마", 1)], "나다라마": [(".", 1)]}, "가나다라마."),
            (7, "가나다라마바", {"가나다라마바": [("사", 1)], "나다라마바사": [(".", 1)]}, "가나다라마바사."),
            (10, "가나다라마바사아자", {"가나다라마바사아자": [("차", 1)], "나다라마바사아자차": [(".", 1)]}, "가나다라마바사아자차."),
        ]
        for n, prefix, model, expected in cases:
            with self.subTest(n=n):
                result = generate_greedy(model, prefix, n=n)
                self.assertEqual(result["n"], n)
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["stop_reason"], "punctuation")
                previous = prefix
                for step in result["steps"]:
                    self.assertEqual(len(step["context"]), n - 1)
                    self.assertEqual(step["context"], previous[-(n - 1):])
                    self.assertEqual(len(step["text"]), len(previous) + 1)
                    previous = step["text"]

    def test_unigram_uses_empty_context_even_for_nonempty_prefix(self):
        result = generate_ngram({"": [(".", 1)]}, "가나다", n=1)
        self.assertEqual(result["text"], "가나다.")
        self.assertEqual(result["steps"][0]["context"], "")

    def test_unigram_can_start_from_empty_prefix(self):
        result = generate_ngram({"": [("!", 1)]}, "", n=1)
        self.assertEqual(result["text"], "!")

    def test_short_prefix_is_rejected_for_requested_orders(self):
        for n in [5, 7, 10]:
            with self.subTest(n=n), self.assertRaises(ValueError):
                generate_ngram({}, "오늘은", n=n)

    def test_higher_order_unseen_context_stops_without_silent_backoff(self):
        result = generate_ngram({"지금은 ": [("좋", 1)]}, "오늘은 ", n=5)
        self.assertEqual(result["stop_reason"], "unseen_context")
        self.assertEqual(result["steps"], [])

    def test_general_loader_keeps_context_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counts.csv"
            path.write_text("context,next,count\n 말은 ,신,1\n 말은 ,한,1\n", encoding="utf-8")
            self.assertEqual(load_ngram(5, path), {" 말은 ": [("신", 1), ("한", 1)]})
            with self.assertRaises(ValueError):
                load_ngram(7, path)


class SamplingTests(unittest.TestCase):
    def test_every_integer_ticket_has_the_expected_two_to_one_weight(self):
        class TicketRng:
            def __init__(self, ticket):
                self.ticket = ticket

            def randrange(self, total):
                if total != 3:
                    raise AssertionError("Expected three tickets")
                return self.ticket

        candidates = [
            {"next": "세", "count": 2, "probability": 2 / 3},
            {"next": "루", "count": 1, "probability": 1 / 3},
        ]
        choices = []
        for ticket in range(3):
            selected, recorded = sample_candidate(candidates, TicketRng(ticket))
            self.assertEqual(recorded, ticket)
            choices.append(selected["next"])
        self.assertEqual(choices, ["세", "세", "루"])
        self.assertEqual(Counter(choices), Counter({"세": 2, "루": 1}))

    def test_sampling_records_the_selected_ticket_and_model_probability(self):
        result = generate_ngram({"하": [("세", 2), ("루", 1)]}, "하", method="sample", max_chars=2, rng=random.Random(42))
        step = result["steps"][0]
        self.assertEqual(step["sampling_ticket"], 2)
        self.assertEqual(step["selected"], "루")
        self.assertAlmostEqual(step["selected_model_probability"], 1 / 3)

    def test_seed_reproduces_whole_batch_and_is_not_reset_per_sentence(self):
        model = {"하": [("세", 1), ("루", 1)]}
        first = generate_batch(model, "하", seed=42, count=10)
        second = generate_batch(model, "하", seed=42, count=10)
        self.assertEqual(first, second)
        self.assertEqual([item["text"] for item in first], ["하세", "하세", "하루", "하세", "하세", "하세", "하세", "하세", "하루", "하세"])
        self.assertTrue(all(item["method"] == "sample" for item in first))

    def test_batch_does_not_remove_duplicate_sentences(self):
        results = generate_batch({"하": [(".", 1)]}, "하", count=10)
        self.assertEqual(len(results), 10)
        self.assertEqual([item["text"] for item in results], ["하."] * 10)

    def test_sampling_loop_obeys_length_limit(self):
        result = generate_ngram({"가나다라": [("라", 1)], "나다라라": [("라", 1)], "다라라라": [("라", 1)], "라라라라": [("라", 1)]}, "가나다라", n=5, method="sample", max_chars=8)
        self.assertEqual(result["text"], "가나다라라라라라")
        self.assertEqual(result["stop_reason"], "max_length")

    def test_invalid_order_method_and_batch_count(self):
        for n, method in [(0, "greedy"), (2, "unknown")]:
            with self.subTest(n=n, method=method), self.assertRaises(ValueError):
                generate_ngram({}, "하", n=n, method=method)
        with self.assertRaises(ValueError):
            generate_batch({}, "하", count=0)


if __name__ == "__main__":
    unittest.main()
