"""원본 보존, 강의 압축본과의 출력 일치, 중복 제거 후 전역 유일성 검증."""

import hashlib
import json
from pathlib import Path
import time
import unicodedata
import zlib

import py7zr


ROOT = Path(__file__).resolve().parent


def main():
    started = time.monotonic()
    prefix = 'Part2_neural_lm/src/ko_wiki_cleaning/'
    folders = {'paragraphs_raw', 'paragraphs', 'sentences', 'sentences_dedup'}
    with py7zr.SevenZipFile(ROOT / 'ko_wiki_cleaning.7z', mode='r') as archive:
        expected = [entry for entry in archive.list() if entry.is_file and
                    (entry.filename.removeprefix(prefix).split('/')[0] in folders or
                     entry.filename.removeprefix(prefix) == 'sentences_dedup_removed.txt')]
    if len(expected) != 129:
        raise ValueError('참고 압축본의 비교 대상 파일 구성이 예상과 다릅니다')
    comparisons = []
    for entry in expected:
        relative = entry.filename.removeprefix(prefix)
        path = ROOT / relative
        checksum = 0
        with path.open('rb') as source:
            while data := source.read(8 * 1024 * 1024):
                checksum = zlib.crc32(data, checksum)
        comparisons.append({'file': relative, 'bytes': path.stat().st_size,
                            'same_size': path.stat().st_size == entry.uncompressed,
                            'same_crc32': checksum == entry.crc32})
    print('압축본과 출력 비교:',sum(x['same_size'] and x['same_crc32'] for x in comparisons), '/', len(comparisons), flush=True)
    seen, duplicates, sentences = set(), 0, 0
    for index, path in enumerate(sorted((ROOT / 'sentences_dedup').glob('part_*.txt')), 1):
        with path.open(encoding='utf-8') as source:
            for line in source:
                if not line.strip():
                    continue
                # 생성 코드의 blake2b·정규식과 별개로 SHA-256·split/join으로 확인한다.
                normalized = ' '.join(unicodedata.normalize('NFKC', line).split())
                key = hashlib.sha256(normalized.encode('utf-8')).digest()
                sentences += 1
                if key in seen:
                    duplicates += 1
                seen.add(key)
        if index % 8 == 0:
            print(f'독립 중복 검증 {index}/32: {sentences:,}개, 잔여 중복 {duplicates}', flush=True)
    summary = json.loads((ROOT / 'reports/step4-summary.json').read_text())
    result = {'archive_compared_files': len(comparisons),
              'archive_matching_files': sum(x['same_size'] and x['same_crc32'] for x in comparisons),
              'archive_comparisons': comparisons,
              'verified_final_sentences': sentences, 'remaining_normalized_duplicates': duplicates,
              'matches_step4_count': sentences == summary['kept_sentences'],
              'elapsed_seconds': round(time.monotonic() - started, 2)}
    snapshot_path = Path('/tmp/korean-wiki-step3-before.json')
    if snapshot_path.exists():
        unchanged, changed = [], []
        for relative, before in json.loads(snapshot_path.read_text()).items():
            if relative == '.gitignore':
                continue
            path = ROOT / relative
            if 'sha256' in before:
                same = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == before['sha256']
            else:
                same = path.is_file() and path.stat().st_size == before['size'] and path.stat().st_mtime_ns == before['mtime_ns']
            (unchanged if same else changed).append(relative)
        result['preserved_input_and_reference_files'] = unchanged
        result['unexpected_changed_files'] = changed
    (ROOT / 'reports/verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    if duplicates or not result['matches_step4_count'] or result.get('unexpected_changed_files'):
        raise RuntimeError('전처리 검증 실패: reports/verification.json 참고')
    if result['archive_matching_files'] != result['archive_compared_files']:
        raise RuntimeError('강의 압축본과 다른 출력이 있습니다: reports/verification.json 참고')
    print(json.dumps({key:value for key,value in result.items() if key not in ['archive_comparisons','preserved_input_and_reference_files']},ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
