"""Membership edge cases independent of model inference."""
import unittest

import numpy as np

from sample_final_model import find_id_sequence, normalize_surface, strip_edge_spaces
from training_corpus import PARAGRAPH_DTYPE


class MembershipTests(unittest.TestCase):
    def records(self, lengths):
        rows = np.zeros(len(lengths), dtype=PARAGRAPH_DTYPE)
        start = 0
        for index, length in enumerate(lengths):
            rows[index]["start"] = start
            rows[index]["length"] = length
            rows[index]["doc_key"] = 100 + index
            start += length
        return rows

    def test_aligned_inside_paragraph(self):
        data = np.asarray([2] * 8 + [5, 6, 7, 3] + [2] * 8 + [5, 6, 3], dtype="<u2")
        hit = find_id_sequence(b"x" + data.tobytes(), 1, self.records([12, 11]), [5, 6])
        self.assertEqual(hit["count"], 2)
        self.assertEqual([x["train_paragraph_index"] for x in hit["examples"]], [0, 1])

    def test_wrong_byte_alignment_ignored(self):
        data = b"\x00\x05\x00\x06\x00" + bytes(25)
        self.assertEqual(find_id_sequence(data, 0, self.records([15]), [5, 6], 0)["count"], 0)

    def test_match_cannot_cross_paragraph_content(self):
        data = np.asarray([2] * 8 + [5, 6, 7] + [8] * 8 + [9, 3], dtype="<u2")
        self.assertEqual(find_id_sequence(data.tobytes(), 0, self.records([11, 10]), [6, 7, 8])["count"], 0)

    def test_special_tokens_and_empty_query_ignored(self):
        data = np.asarray([2] * 8 + [5, 3], dtype="<u2").tobytes()
        for query in ([], [2], [5, 3], [1]):
            self.assertEqual(find_id_sequence(data, 0, self.records([10]), query)["count"], 0)

    def test_surface_normalization_does_not_remove_punctuation(self):
        self.assertEqual(normalize_surface("서울은\t 수도이다.\r\n"), "서울은 수도이다.")
        self.assertNotEqual(normalize_surface("서울은 수도이다"), normalize_surface("서울은 수도이다."))

    def test_trim_only_sentence_edge_space_tokens(self):
        tokens = ["<sp>/WS", "서울/NNP", "<sp>/WS", "수도/NNG", "<sp>/WS"]
        self.assertEqual(strip_edge_spaces(tokens), tokens[1:-1])


if __name__ == "__main__":
    unittest.main()
