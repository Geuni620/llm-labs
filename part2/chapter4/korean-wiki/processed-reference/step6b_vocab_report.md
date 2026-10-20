# Step 6b. Vocab 시각화 보고서

Step 6a로 생성된 여러 vocab 설정을 자동 탐색하여 비교/개별 보고서를 생성한다.

## 자동 탐색

`vocab_*.txt` 파일을 정규식 `^vocab_(\d+)\.txt$`로 탐색한다.
예: `vocab_32768.txt`, `vocab_49152.txt`, `vocab_65536.txt` → 3개 설정 발견.

## 분석 (멀티프로세싱)

### 1st pass: 토큰 빈도 + 문장 유지율

32 샤드를 32 워커가 병렬 처리. 각 워커가 토큰 빈도 카운팅과
모든 vocab 설정에 대한 문장 유지 여부를 동시에 계산한다.

### 2nd pass: 문장별 최소 토큰 빈도

각 문장에서 가장 빈도가 낮은 토큰의 빈도를 기록한다.
이 값으로 임의의 min_count에 대한 문장 유지율을 사후 계산할 수 있다.

## 보고서 종류

### vocab_comparison.png

4패널 비교 보고서:

1. **순위-빈도 분포** — Zipf 곡선 + vocab 경계 영역 색칠
2. **토큰 커버리지 vs 문장 유지율** — 그룹 막대 (커버리지 ~99%인데 유지율 ~50%인 갭 시각화)
3. **vocab size vs 문장 유지율** — 1K–200K 범위 연속 곡선 + 각 설정 마커
4. **구간별 추가 토큰 품사 분포** — 적층 막대 (한국어 라벨: 일반명사, 고유명사, 동사, ...)

### vocab_report_full.png

전체 vocab 범위(~140만 토큰)에 대한 4패널 보고서:

1. 순위-빈도 (Zipf)
2. 빈도 히스토그램
3. min_count별 vocab size (+ Bengio 2003, Word2Vec 참조선)
4. min_count별 문장 유지율

### vocab_report_{VS}.png (설정별)

각 vocab 설정에 해당하는 범위만 표시하는 4패널 보고서.
경계 빈도 이상의 min_count 범위만 표시하고 축 스케일도 해당 범위에 맞춤.

## 입출력

```
입력: tokens/part_NN.txt    (32 샤드, 원본 토큰)
      vocab_*.txt            (Step 6a 출력)
출력: vocab_comparison.png
      vocab_report_full.png
      vocab_report_{VS}.png  (설정별, 예: vocab_report_32768.png)
```

## CLI

```bash
python step6b_vocab_report.py
```

인자 없음. `vocab_*.txt` 파일이 존재하면 자동 실행.

## 생성된 보고서

실제 생성된 PNG 파일:

- [vocab_comparison.png](vocab_comparison.png) — 3개 vocab 설정 비교 (순위-빈도, 커버리지 vs 유지율, vocab size 곡선, POS 분포)
- [vocab_report_full.png](vocab_report_full.png) — 전체 ~140만 토큰 범위 (Zipf, 히스토그램, min_count 곡선)
- [vocab_report_32768.png](vocab_report_32768.png) — 32K vocab 설정별 보고서
- [vocab_report_49152.png](vocab_report_49152.png) — 49K vocab 설정별 보고서
- [vocab_report_65536.png](vocab_report_65536.png) — 65K vocab 설정별 보고서