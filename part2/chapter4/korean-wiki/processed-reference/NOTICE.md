# NOTICE — ko_wiki_cleaning 데이터셋 출처 및 라이선스

이 데이터셋은 한국어 위키백과 콘텐츠의 가공물입니다.
원본 라이선스를 상속하므로 재배포 시 반드시 본 파일과 `LICENSE`를 함께 포함하고
동일 라이선스 조건을 유지해야 합니다.

---

## 1. 원본 출처

- **데이터셋**: `HuggingFaceFW/finewiki` (서브셋 `kowiki`)
  - <https://huggingface.co/datasets/HuggingFaceFW/finewiki>
- **유래**: 한국어 위키백과
  - <https://ko.wikipedia.org>

## 2. 라이선스

원본과 동일하게 다음 듀얼 라이선스 하에 배포됩니다:

- **CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0 International)
  - 전문: 본 폴더의 `LICENSE` 파일
  - 요약: <https://creativecommons.org/licenses/by-sa/4.0/>
- **GFDL 1.3** (GNU Free Documentation License)
  - 전문: <https://www.gnu.org/licenses/fdl-1.3.txt>

## 3. 재배포 의무

CC BY-SA 4.0의 조건에 따라 본 데이터셋(또는 그 파생물)을 재배포할 때는
다음을 반드시 지켜야 합니다:

1. **출처 표시 (Attribution)** — 본 NOTICE 파일을 그대로 동봉하거나
   동등한 출처 정보를 명시할 것
2. **변경 사항 표시** — 추가 가공이 있다면 어떤 변경을 했는지 명시할 것
3. **동일 라이선스 (Share-Alike)** — 재배포물도 CC BY-SA 4.0
   (또는 호환 라이선스) 하에 배포할 것
4. **DRM/접근 제한 금지** — 본 라이선스에 명시된 권리 행사를 막는
   기술적 보호 조치를 적용하지 말 것

상업적 이용은 허용됩니다 (강의 자료 일부로 묶어 판매 가능).

## 4. 가공 내역

본 폴더 `Part2_neural_lm/src/ko_wiki_cleaning/`의 스크립트로 다음 가공을
수행했습니다 (자세한 절차: `README.md`).

| 단계 | 내용 | 산출물 |
|------|------|--------|
| step1 | HuggingFace에서 parquet 2개 다운로드 | `parquet/data/kowiki/*.parquet` |
| step3 | 8단계 필터로 마크업·템플릿·표·반복 제거 | `paragraphs/`, `sentences/` |
| step4 | blake2b 16-byte 해시 기반 전역 dedup | `sentences_dedup/` |
| step5a | Kiwi 형태소 분석기로 형태소/품사 토큰화 | `tokens/` |
| step6a | 빈도 기반 어휘 구축 + OOV 문장 필터링 | `vocab_*.txt`, `tokens_filtered_*/` |
| step7 | 토큰 → uint16 정수 배열 인코딩 | `token_ids_*/` |

가공 산출물(`paragraphs/`, `sentences/`, `sentences_dedup/`, `tokens/`,
`tokens_filtered_*/`)은 위키 콘텐츠의 파생물이므로 원본 라이선스를 상속합니다.

어휘 파일(`vocab_*.txt`)과 정수 배열(`token_ids_*/`)은 통계적·기계적
산출물에 가깝지만, 보수적으로 같은 라이선스 하에 배포합니다.

## 5. 사용 도구

본 가공 파이프라인은 다음 외부 라이브러리를 사용하지만, 이들 자체는
본 데이터셋에 포함되지 않으며 각자의 라이선스를 따릅니다:

- **Kiwi** (`kiwipiepy`, 모델: `cong-global`) — 형태소 분석
- **huggingface_hub** — 데이터 다운로드
- **pyarrow** — parquet 처리
- **NumPy** — 배열 인코딩

## 6. 면책

본 데이터셋은 위키백과 문서의 스냅샷에서 유래하므로 사실 관계, 최신성,
편향 등에 한계가 있을 수 있습니다. 학습용 코퍼스로 사용할 때는 이러한
특성을 인지하고 사용해 주세요.

---

문의: 본 강의 자료의 가공 절차에 대한 질문은 `README.md` 참고.
원본 데이터셋·위키백과 정책 관련 문의는 위 출처 링크 참조.
