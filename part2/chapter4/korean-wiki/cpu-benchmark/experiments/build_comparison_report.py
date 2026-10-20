#!/usr/bin/env python3
"""Summarize the recorded short trials; never starts a benchmark or training."""

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "cpu-benchmark/results/cpu-optimizations-mac-real-wiki-20261018-12s.json"
REPORTS = ROOT / "reports"
STEM = "cpu-optimization-comparison"
LABELS = {
    "baseline": ("기준 O3", "O3 + Accelerate"),
    "native": ("native", "+ -mcpu=native"),
    "relaxed_simd": ("native + fast-math", "+ -mcpu=native -ffast-math"),
    "softmax_omp": ("softmax OMP", "softmax 행을 OpenMP로 분담"),
    "reduction_omp": ("리덕션 OMP", "bias 합산·gradient norm 병렬화"),
    "adamw_omp": ("AdamW OMP", "동일 AdamW의 파라미터 업데이트 병렬화"),
    "tiled32": ("SGEMM 타일 32", "행렬곱을 직접 구현한 32×32 타일로 대체"),
    "tiled64": ("SGEMM 타일 64", "행렬곱을 직접 구현한 64×64 타일로 대체"),
    "fast_exp": ("fast-exp", "softmax 내부 expf를 범위 제한 근사로 대체"),
}


def main():
    raw = SOURCE.read_bytes()
    data = json.loads(raw)
    assert data["status"] == "complete"
    assert data["source_before"] == data["source_after"]
    assert data["corpus"]["post_measurement_source_verification"]["passed"]
    assert data["same_initialization_validation"]["passed_trials_checked"] == 18
    assert len(data["trials"]) == 18
    rows = []
    for result in data["summary"]:
        name = result["case"]
        label, change = LABELS[name]
        trials = sorted((x for x in data["trials"] if x["case"] == name), key=lambda x: x["repeat"])
        assert len(trials) == 2 and all(x["passed"] for x in trials)
        gate = next(x for x in data["correctness"] if x["case"] == name)
        assert gate["passed"] and gate["report"]["passed_cases"] == 8
        observations = []
        for trial in trials:
            assert trial["target_prediction_events"] == trial["steps"] * 128
            assert math.isclose(trial["targets_per_second"], trial["target_prediction_events"] / trial["elapsed_seconds"], rel_tol=1e-12)
            observations.append({
                "repeat": trial["repeat"], "seconds": trial["elapsed_seconds"],
                "steps": trial["steps"], "targets": trial["target_prediction_events"],
                "rate": trial["targets_per_second"],
            })
        median = statistics.median(x["rate"] for x in observations)
        assert math.isclose(median, result["median_targets_per_second"], rel_tol=1e-12)
        rows.append({
            "case": name, "label": label, "change": change, "trials": observations,
            "median": median, "minimum": min(x["rate"] for x in observations),
            "maximum": max(x["rate"] for x in observations),
            "numeric_cases_passed": 8,
            "numeric_max_absolute_error": gate["report"]["maximum_absolute_error"],
        })
    assert len(rows) == 9
    fast = json.loads((ROOT / "cpu-benchmark/experiments/fast-exp-verification.json").read_text())
    summary = {
        "source": str(SOURCE), "source_sha256": hashlib.sha256(raw).hexdigest(),
        "configuration": data["configuration"], "rows": rows,
        "actual_seconds_min": min(x["elapsed_seconds"] for x in data["trials"]),
        "actual_seconds_max": max(x["elapsed_seconds"] for x in data["trials"]),
        "corpus_owned_ram_bytes": data["trials"][0]["corpus"]["owned_ram_bytes"],
        "maximum_process_rss_bytes": max(x["peak_rss_bytes"] for x in data["trials"]),
        "numeric_cases_passed": sum(x["report"]["passed_cases"] for x in data["correctness"]),
        "numeric_comparisons_passed": sum(x["report"]["comparison_count"] for x in data["correctness"]),
        "numeric_max_absolute_error": max(x["numeric_max_absolute_error"] for x in rows),
        "fast_exp_samples": fast["total_in_domain_samples"],
        "fast_exp_max_relative_error": fast["max_relative_error_vs_double_exp"],
        "fast_exp_softmax_probability_max_absolute_error": fast["max_probability_absolute_error_float32_normalization"],
        "interpretation": data["interpretation"],
        "combination": data["combined_case"],
        "review": {
            "calculation_recheck": "passed",
            "source_hashes_unchanged": True,
            "actual_input_hashes_postrun": data["corpus"]["post_measurement_source_verification"],
            "independent_reviewers": ["integer_array_review", "numeric_tokens"],
            "learned_quality_evaluated": False,
            "five_epochs_executed": False,
        },
    }
    (REPORTS / f"{STEM}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "Mac CPU 최적화 단시간 비교 결과",
        "Apple M1 / CPU / V35328 C8 D64 H128 / batch128 / seed42 / OpenMP 실제 팀4",
        "실제 Wiki train 전체를 RAM에 적재. 각 조건 12초 요청 ×2회, 순서 정방향/역방향.",
        f"실제 계산시간 {summary['actual_seconds_min']:.6f}–{summary['actual_seconds_max']:.6f}초. RAM 로딩·초기화·워밍업 제외.",
        "처리량 단위: 다음 토큰 예측 횟수/초. 업데이트 1회 =128개 예측.",
        "설정\t변경\t1회 초\t1회 업데이트\t1회 예측/초\t2회 초\t2회 업데이트\t2회 예측/초\t중앙값 예측/초",
    ]
    for row in rows:
        a, b = row["trials"]
        lines.append(f"{row['label']}\t{row['change']}\t{a['seconds']:.6f}\t{a['steps']}\t{a['rate']:.2f}\t{b['seconds']:.6f}\t{b['steps']}\t{b['rate']:.2f}\t{row['median']:.2f}")
    lines += [
        "", "결론: 현재 부하에서는 정밀 순위를 확정할 수 없음. O3+Accelerate를 기준으로 유지.",
        "기준 설정 자체가 두 측정 사이 4.34배 변동. 중앙값의 비율은 최적화의 인과 효과가 아님.",
        "두 반복 모두 기준보다 빨랐던 설정이 없어 조합은 미실행. 조합 속도를 더해 추정하지 않음.",
        "O3와 Accelerate도 이미 SIMD 가능. native는 CPU 타깃 지정, fast-math는 부동소수점 규칙 완화도 포함.",
        "컴파일 옵션은 C 실험 코드에 적용하며 사전 컴파일된 Accelerate는 그대로임.",
        "리덕션: 여러 값을 합계·최댓값 하나로 집계. 각 스레드 부분 결과를 마지막에 병합.",
        "이 실험은 bias 합산과 gradient norm을 병렬화. softmax는 서로 독립적인 배치 행을 분담.",
        "AdamW: 기울기와 그 제곱의 이동평균으로 가중치를 갱신하고 별도의 weight decay 적용.",
        "원래도 AdamW를 사용. 이번 변경은 같은 알고리즘의 파라미터 업데이트를 병렬 실행.",
        f"수치 검증: 72개 조건/1656개 비교 통과. 최대 절대 오차 {summary['numeric_max_absolute_error']:.9g}.",
        "허용 기준: abs(error) <=2e-6 +2e-4*abs(reference). 합성 두 크기, 스레드1/4, clipping1/.05 검사.",
        f"fast-exp: [-80,0] 구간 {summary['fast_exp_samples']:,}개 표본에서 exp 대비 최대 상대 오차 {summary['fast_exp_max_relative_error']:.9g}.",
        f"softmax 확률 최대 절대 오차 {summary['fast_exp_softmax_probability_max_absolute_error']:.9g}. 범위 밖은 expf 사용.",
        "표본 수치 일치는 장시간 학습 품질 검증이 아님. 종료 loss는 서로 다른 업데이트 수라 품질 비교에 쓰지 않음.",
        "SGEMM 모듈 O3/OMP/UBSan 검사 통과. ASan은 런타임 초기화 교착으로 수행하지 못함.",
        "실제 OpenMP 팀은4명. Accelerate 내부 작업자 수는 측정하지 않음.",
        "Ubuntu 측정·컴퓨터 선택·실제 전체 에폭 학습은 이번 범위에 포함하지 않음.",
        f"원본 측정: {SOURCE}", f"원본 SHA256: {summary['source_sha256']}",
        "개념 출처: https://www.openmp.org/spec-html/5.1/openmpsu117.html",
        "개념 출처: https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html",
    ]
    (REPORTS / f"{STEM}.txt").write_text("\n".join(lines) + "\n")
    template = (REPORTS / f"{STEM}.template.html").read_text()
    fragment = template.replace("__DATA__", json.dumps(summary, ensure_ascii=False).replace("<", "\\u003c"))
    assert "__DATA__" not in fragment and len(fragment.encode()) < 1_000_000
    (REPORTS / f"{STEM}.html").write_text(fragment)
    shell = (REPORTS / "visualization-shell.html").read_text()
    (REPORTS / f"{STEM}.preview.html").write_text(shell.replace("__CONTENT__", fragment))
    print(json.dumps({"cases": len(rows), "trials": len(data["trials"]), "numeric_cases_passed": summary["numeric_cases_passed"], "source_sha256": summary["source_sha256"], "ranking": summary["interpretation"]["ranking_status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
