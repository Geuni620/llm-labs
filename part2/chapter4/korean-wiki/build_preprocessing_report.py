"""실제 실행 기록으로 필터별 제거 사례와 1·2·3단계 비교 자료를 만든다."""

from pathlib import Path
import json

from step3_extract_paragraphs import clean_paragraph, split_sentences, hanja_ratio, letter_ratio, hangul_ratio


ROOT = Path(__file__).resolve().parent
LABELS = {
    'has_formula': '수식 포함 문서 전체 제외',
    'is_structural': '제목·목록·번호·표·빈 행 제외',
    'has_math_symbol': '수학 전용 기호 포함 행 제외',
    'hanja_ratio': '한자/(한자+한글) > 30% 행 제외',
    'clean_paragraph': '외국어 괄호 제거·한글/숫자 괄호 보존·공백 정리',
    'letter_ratio': '문자 비율 < 20% 행 제외',
    'hangul_ratio': '한글 비율 < 30% 행 제외',
    'is_sentence_shape': '마침표·물음표·느낌표로 끝나지 않는 행 제외',
}


def main():
    report = ROOT / 'reports'
    step3 = json.loads((report / 'step3-summary.json').read_text())
    step4 = json.loads((report / 'step4-summary.json').read_text())
    comparison = json.loads((report / 'step1-3-comparison.json').read_text())
    verification_path = report / 'verification.json'
    verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
    filters = []
    for entry in step3['filters']:
        name = entry['filter']
        example = step3['cases'].get('clean_paragraph_changed' if name == 'clean_paragraph' else name)
        row = {**entry, 'label': LABELS[name], 'example': example}
        if name == 'clean_paragraph':
            row['emptied_example'] = step3['cases'].get('clean_paragraph')
        if example:
            before = example['before']
            inspected = example['after'] if example['after'] is not None else before
            if name == 'hanja_ratio': row['example_ratio_percent'] = hanja_ratio(before) * 100
            if name == 'letter_ratio': row['example_ratio_percent'] = letter_ratio(inspected) * 100
            if name == 'hangul_ratio': row['example_ratio_percent'] = hangul_ratio(inspected) * 100
        filters.append(row)
    counts = step3['counts']
    parentheses_input = '서울(Seoul)은 대한민국의 수도 (약 100만명)이다.'
    initials_input = '1953년 제임스 D. Watson은 DNA를 연구했다. 다음 문장이다.'
    result = {
        'step1': comparison['step1'], 'step2': comparison['step2'], 'step3': step3,
        'step4': step4, 'verification': verification, 'filters': filters,
        'samples': comparison['samples'],
        'content_paragraphs_removed': counts['raw_paragraphs'] - counts['clean_paragraphs'],
        'content_paragraphs_removed_percent': (counts['raw_paragraphs'] - counts['clean_paragraphs']) / counts['raw_paragraphs'] * 100,
        'source_lines_removed': counts['source_lines'] - counts['clean_paragraphs'],
        'source_lines_removed_percent': (counts['source_lines'] - counts['clean_paragraphs']) / counts['source_lines'] * 100,
        'validated_examples': {
            'parentheses': {'kind': 'synthetic validation example', 'before': parentheses_input,
                            'after': clean_paragraph(parentheses_input)},
            'initials': {'kind': 'synthetic validation example', 'before': initials_input,
                         'after': split_sentences(initials_input)},
            'actual_parentheses_preserved': step3['cases'].get('parentheses_preserved'),
            'actual_initials_preserved': step3['cases'].get('initials_preserved'),
        },
        'scope_notes': [
            '문서 수와 본문행 수, 문장 수는 서로 다른 단위이며 비율을 합산하지 않는다.',
            '라인 필터는 원본 text의 한 줄(문단 후보)에 적용한 뒤 문장 분리한다.',
            '샘플 5개는 확인용이며 전체 처리 입력은 원본 582,035개 문서이다.',
            '한 문단은 여러 문장을 포함할 수 있다.',
            '5번은 부분 정리이며 변경된 행과 완전히 빈 행이 되어 제외된 경우를 구분한다.',
            '원문행 전체 제외율에는 수식 문서 안의 행과 빈 행·제목·목록·표가 포함된다.',
            '중복 제거는 NFKC와 공백 정리 후 완전 일치 기준이며 영문 대소문자를 보존한다.',
            '규칙 기반 필터이므로 문법적 완성도나 의미가 같은 문장의 중복까지 보장하지 않는다.',
        ],
    }
    (report / 'preprocessing-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    lines = ['한국어 위키 전처리 결과', '',
             f"Step 1: 원본 문서 {comparison['step1']['total_rows']:,}개, 14개 열, Parquet 2개.",
             'Step 2: text 열의 원문을 그대로 담은 TXT 샘플 5개. 원본과 내용 일치.',
             f"Step 3: 본문행 {counts['raw_paragraphs']:,}개 → 정제 문단 {counts['clean_paragraphs']:,}개.",
             f"본문행 제외 {result['content_paragraphs_removed']:,}개 ({result['content_paragraphs_removed_percent']:.2f}%).",
             f"원문행 전체 {counts['source_lines']:,}개 중 {result['source_lines_removed']:,}개 제외 ({result['source_lines_removed_percent']:.2f}%). 빈 행·구조 행과 수식 문서의 행 포함.",
             f"문장 분리 결과 {counts['sentences']:,}개.",
             f"Step 4: {step4['input_sentences']:,}개 → {step4['kept_sentences']:,}개.",
             f"중복 문장 제외 {step4['removed_sentences']:,}개 ({step4['removed_percent']:.2f}%).", '',
             '필터별 실제 사례 — 앞 필터에서 통과한 항목 중 최초 탈락 사유만 집계', '']
    for entry in filters:
        lines += [f"{entry['number']}. {entry['label']}",
                  f"완전 제외: {entry['removed']:,} / {entry['denominator']:,} ({entry['denominator_label']}, {entry['removed_percent']:.2f}%)"]
        if entry['number'] == 5:
            lines += [f"내용 변경: {entry['changed_lines']:,} / {entry['changed_input_lines']:,}행 ({entry['changed_lines'] / entry['changed_input_lines'] * 100:.2f}%, 괄호 정리에 진입한 행 기준). 삭제 문자: {entry['characters_removed']:,}자."]
        example = entry['example']
        if example:
            position = f"본문 {example['line']}행" if example['line'] else '문서 전체'
            lines += [f"출처: {example['source'].get('title')} ({example['source'].get('id')}), {position}",
                      '[원문]', example['before'], '[결과]',
                      example['after'] if entry['number'] == 5 else ('문서 전체 제외' if entry['number'] == 1 else '해당 행 제외')]
        else:
            lines += ['해당 조건에 맞는 실제 사례 없음.']
        if entry.get('emptied_example'):
            empty = entry['emptied_example']
            lines += [f"괄호 정리 후 행 전체가 빈 경우: {empty['source']['title']} ({empty['source']['id']})",
                      empty['before'], '→ 빈 행이 되어 제외']
        lines += ['']
    lines += ['9. 전역 중복 문장 제거', '']
    for example in step4['top_duplicates'][:5]:
        lines += [example['text'], f"{example['occurrences_before']:,}회 등장 → 1회 보존 / {example['removed']:,}회 제외", '']
    lines += ['요청한 예외 처리의 인공 검증 예시',
              parentheses_input, result['validated_examples']['parentheses']['after'],
              initials_input, json.dumps(result['validated_examples']['initials']['after'],ensure_ascii=False), '',
              '비율 해석', *result['scope_notes']]
    (report / 'preprocessing-summary.txt').write_text('\n'.join(lines)+'\n')
    changes = []
    for sample in comparison['samples']:
        processed = sample['step3']
        changes += [f"===== {sample['title']} ({sample['id']}) =====", '',
                    '[Step 1] Parquet 원본: 14개 열. 아래는 text 열.', sample['step1_row']['text'],
                    '[Step 2] TXT 추출: 위 text와 바이트 단위로 동일.', sample['step2_file'], '',
                    '[Step 3] 정제 문단', processed['paragraphs'] or '(문서 전체 제외)',
                    '[Step 3] 문장 분리', processed['sentences'] or '(문서 전체 제외)', '']
    (report / 'step1-3-examples.txt').write_text('\n'.join(changes)+'\n')
    print(json.dumps({'documents': counts['documents'], 'raw_paragraphs': counts['raw_paragraphs'],
                      'clean_paragraphs': counts['clean_paragraphs'], 'sentences': counts['sentences'],
                      'dedup_kept': step4['kept_sentences'],
                      'filters': [{k:v for k,v in entry.items() if k!='example'} for entry in filters]},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
