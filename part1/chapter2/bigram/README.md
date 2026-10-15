# 바이그램 — N=2

연속한 글자 두 개를 묶습니다. `하루가`에서는 `하루`, `루가`를 셉니다. 앞 글자 하나를 문맥으로 사용해 다음 글자의 확률을 계산합니다.

프로젝트 루트에서 실행합니다.

```sh
python3 part1/chapter2/count_ngrams.py --n 2 --context 하 --top 10
python3 part1/chapter2/count_ngrams.py --n 2 --context 세 --top 10
```

- `counts.csv`: 관측된 모든 두 글자 묶음의 횟수와 조건부 확률
- `summary.json`: 같은 입력 데이터에서 센 전체 묶음 수와 서로 다른 묶음 수

현재 데이터에서 `하세`는 104회, `하루`는 5회입니다. `하` 뒤에 글자가 나온 경우는 총 559회이므로 각각의 비율은 약 18.60%, 0.89%입니다. 분모는 같은 문맥 `하`를 가진 모든 묶음의 횟수 합입니다.

생성할 때 최고 확률을 고르면 `하` 다음에 `세`를 붙입니다. 그다음에는 마지막 글자 `세`를 문맥으로 조회합니다. 확률에 따라 추첨하는 생성은 같은 빈도표를 사용하는 다른 선택 방법입니다.

## 최대 빈도 문장 생성

```sh
python3 part1/chapter2/generation/generate.py --n 2 --prefix 하 --method greedy --trace
```

생성된 `하세요.`는 [generation/result.txt](generation/result.txt)에, 한 글자씩 선택한 과정은 [generation/trace.json](generation/trace.json)에 저장됩니다. 실행 옵션과 종료 기준은 [생성 실습 설명](../generation/README.md)에서 확인합니다.

다음 단계: [트리그램](../trigram/README.md)에서 앞 글자 두 개를 함께 조건으로 사용합니다.

## 무작위 바이그램 10개에서 다음 글자 선택

[random_examples.csv](random_examples.csv)와 [random_examples.json](random_examples.json)에 시드 42의 추출 결과를 저장했습니다. 관측된 5,157개 바이그램 종류에서 중복 없이 10개를 균일하게 뽑고, 각 묶음의 마지막 글자 뒤에서 최대 빈도 후보를 골랐습니다. 다음 후보의 비율이 높은 순서로 정렬했습니다.

이어지는 바이그램은 한 글자를 공유합니다. 예를 들어 `게요` 뒤에서 문맥 `요`의 최대 빈도 후보 `.`를 고르면, 다음 바이그램은 `요.`이고 연결 결과는 `게요.`입니다. 두 묶음 사이의 `요`를 한 번만 출력합니다.

이 비율의 분모는 마지막 글자 뒤에 나온 전체 횟수입니다. 시작 바이그램을 무작위로 뽑는 단계와 다음 글자를 최대 빈도로 고르는 단계의 선택 방법을 구분합니다.
