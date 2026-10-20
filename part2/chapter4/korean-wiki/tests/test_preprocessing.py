import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import step3_extract_paragraphs as step3
import step4_dedup_sentences as step4


def load_reference(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'processed-reference' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REFERENCE3 = load_reference('lecture_step3', 'step3_extract_paragraphs.py')
REFERENCE4 = load_reference('lecture_step4', 'step4_dedup_sentences.py')


class FilterTests(unittest.TestCase):
    def test_formula_drops_whole_document(self):
        text = '한국어 설명이다.\n{\\displaystyle x=2}\n마지막 설명이다.'
        result = step3.analyze_document(text)
        self.assertTrue(result['document_rejected'])
        self.assertEqual(result['paragraphs'], '')
        self.assertEqual(result['sentences'], '')

    def test_structural_lines_and_non_structural_boundaries(self):
        for value in ['', '  ', '#제목', '## 제목', ' - 목록', '* 목록', '| 표 |', '12. 목록']:
            with self.subTest(value=value):
                self.assertTrue(step3.is_structural(value))
        for value in ['-목록아님', '1.', '3.14는 원주율이다.', '보통 문장이다.']:
            with self.subTest(value=value):
                self.assertFalse(step3.is_structural(value))

    def test_math_symbol_scope_matches_lecture(self):
        for value in '±∓≠≤≥∈∪∑√∞':
            self.assertTrue(step3.has_math_symbol(value))
        for value in '×÷°½→←πασ':
            self.assertFalse(step3.has_math_symbol(value))

    def test_ratio_boundaries_and_denominators(self):
        self.assertEqual(step3.hanja_ratio('漢漢漢가나다라마바사123'), 0.3)
        self.assertGreater(step3.hanja_ratio('漢漢漢漢가나다라마바'), 0.3)
        self.assertEqual(step3.letter_ratio('가123.'), 0.2)
        self.assertEqual(step3.hangul_ratio('가나다123456.'), 0.3)

    def test_foreign_parentheses_removed_korean_and_numbers_preserved(self):
        source = '서울(Seoul)은 대한민국의 수도 (약 100만명)이다.'
        self.assertEqual(step3.clean_paragraph(source), '서울은 대한민국의 수도 (약 100만명)이다.')
        self.assertEqual(step3.clean_paragraph('수도(서울(Seoul), 약 100만명)이다.'),
                         '수도(서울, 약 100만명)이다.')
        self.assertEqual(step3.clean_paragraph('서울(Seoul(서울))은 수도이다.'), '서울은 수도이다.')

    def test_whitespace_and_period_cleanup(self):
        self.assertEqual(step3.clean_paragraph('  둘레 16\u00a0km이다. . .  '), '둘레 16 km이다.')

    def test_sentence_endings_are_exact(self):
        for source in ['문장이다.', '문장인가?', '문장이다!']:
            self.assertTrue(step3.is_sentence_shape(source))
        for source in ['제목', '.', '문장이다。[4]', '문장이다.[4]', '문장이다."']:
            self.assertFalse(step3.is_sentence_shape(source))

    def test_initials_decimals_and_numbered_fragments(self):
        self.assertEqual(step3.split_sentences('1953년 제임스 D. Watson은 DNA를 연구했다. 다음 문장이다.'),
                         ['1953년 제임스 D. Watson은 DNA를 연구했다.', '다음 문장이다.'])
        self.assertEqual(step3.split_sentences('제임스 D. R. Smith가 말했다.'),
                         ['제임스 D. R. Smith가 말했다.'])
        self.assertEqual(step3.split_sentences('길이는 3.14 km이다. 다음이다.'),
                         ['길이는 3.14 km이다.', '다음이다.'])
        self.assertEqual(step3.split_sentences('순서다. 1. 준비한다. 2. 시작한다.'),
                         ['순서다.', '준비한다.', '시작한다.'])

    def test_every_line_rejection_is_counted_once(self):
        text = ('# 제목\n서울(Seoul)은 수도 (약 100만명)이다.\n| 표 |\n가∑나다.\n'
                '漢漢漢한.\n(Seoul)\n123 !!!\nhello world.\n끝나지 않음\n')
        result = step3.analyze_document(text)
        stats = result['counts']
        for name in step3.FILTERS[1:]:
            self.assertGreater(stats.get('rejected_' + name, 0), 0, name)
        self.assertEqual(stats['clean_paragraphs'], 1)
        self.assertIn('(약 100만명)', result['paragraphs'])
        self.assertEqual(stats['raw_paragraphs'] - sum(stats.get('rejected_' + n, 0) for n in step3.FILTERS[2:]),
                         stats['clean_paragraphs'])

    def test_all_five_original_samples_match_reference(self):
        for page_id in [282586, 1792712, 3110350, 660723, 303062]:
            text = (ROOT / f'examples/kowiki-{page_id}.text.txt').read_text()
            actual = [io.StringIO(), io.StringIO(), io.StringIO()]
            expected = [io.StringIO(), io.StringIO(), io.StringIO()]
            self.assertEqual(step3._write_paragraphs(*actual, [text]),
                             REFERENCE3._write_paragraphs(*expected, [text]))
            self.assertEqual([b.getvalue() for b in actual], [b.getvalue() for b in expected])

    def test_real_first_1000_documents_match_reference(self):
        source = ROOT / step3.PARQUET_FILES[0]
        texts = pq.ParquetFile(source).read_row_group(0, columns=['text']).column('text').to_pylist()
        actual = [io.StringIO(), io.StringIO(), io.StringIO()]
        expected = [io.StringIO(), io.StringIO(), io.StringIO()]
        self.assertEqual(step3._write_paragraphs(*actual, texts),
                         REFERENCE3._write_paragraphs(*expected, texts))
        self.assertEqual([b.getvalue() for b in actual], [b.getvalue() for b in expected])


class DedupTests(unittest.TestCase):
    def test_normalization_preserves_english_case(self):
        self.assertEqual(step4.normalize(' Ａpple\t  사과 '), 'Apple 사과')
        self.assertNotEqual(step4.normalize('Apple'), step4.normalize('apple'))

    def test_cross_shard_dedup_and_empty_paragraph_removal_match_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals, duplicates = {}, Counter()
            reference_seen, reference_duplicates = {}, {}
            cases = []
            source_values = ['서울은 수도다.\n\n부산은 항구다.\n',
                             '서울은  수도다.\n\n새 문장이다.\n']
            for index, text in enumerate(source_values):
                source = root / f'input-{index}.txt'
                output, expected = root / f'out-{index}.txt', root / f'ref-{index}.txt'
                source.write_text(text)
                counts = step4.dedup_one_file(source, output, originals, duplicates, index, cases)
                expected_counts = REFERENCE4._dedup_one_file(str(source), str(expected), reference_seen, reference_duplicates)
                self.assertEqual(counts, expected_counts)
                self.assertEqual(output.read_text(), expected.read_text())
            self.assertEqual(duplicates, reference_duplicates)
            self.assertEqual((root / 'out-1.txt').read_text(), '새 문장이다.\n\n')
            first_texts = {key: reference_seen[key] for key in duplicates}
            actual, expected = root / 'removed.txt', root / 'ref-removed.txt'
            step4._write_removed_log(actual, first_texts, duplicates, 2, 4, 3)
            REFERENCE4._write_removed_log(str(expected), reference_seen, reference_duplicates, 2, 4, 3)
            self.assertEqual(actual.read_text(), expected.read_text())


if __name__ == '__main__':
    unittest.main()
