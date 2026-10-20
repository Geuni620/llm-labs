"""Assemble measured evidence and bounded forecasts; never train a model."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

WIKI = Path(__file__).resolve().parent.parent
REPORTS = WIKI / "reports"
TARGETS = 124232535


def read(name):
    return json.loads((REPORTS / name).read_text())


def identity(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def peak_rss(report, name):
    values = []
    for row in report["trials"]:
        if row["case"] == name:
            text = Path(row["stderr_log"]).read_text()
            match = re.search(r"(?m)^\s*(\d+)\s+maximum resident set size\s*$", text)
            if not match:
                raise ValueError("Missing RSS")
            values.append(int(match.group(1)))
    return max(values)


def main():
    cpu = read("cpu-ilp-round2-comparison-20261018-v1.json")
    gpu = read("mps-combinations-round2-comparison-20261018-v1.json")
    threads = read("cpu-thread-round2-comparison-20261018-v1.json")
    budget = read("nplm-budget-model-20261018-v2.json")
    budget_gate = read("nplm-budget-model-verification-20261018-v1.json")
    assert budget['status'] == 'complete' and all(r['passed'] for r in budget['trials'])
    assert budget_gate['passed'] and budget_gate['failed_comparisons'] == 0 and budget_gate['comparison_count'] == 128
    for name, report in (("cpu", cpu), ("gpu", gpu), ("threads", threads)):
        assert report["status"] == "complete", name
        assert all(row["passed"] for row in report["trials"]), name
    assert all(cpu["ilp_final_state_bit_identical"].values())
    for filename in ("cpu-thread-round2-verification-20261018-v1.json",
                     "cpu-ilp-round2-verification-20261018-v1.json",
                     "mps-combinations-round2-verification-20261018-v1.json"):
        report = read(filename)
        assert report["passed"] and report["failed_comparisons"] == 0
    unit = read("cpu-ilp-round2-unit-20261018-v1.json")
    assert unit["finite_gate_passed"] and unit["nonfinite_probe"]["bitwise_identical"]
    for report in (cpu, gpu):
        for row in report["summary"]:
            rate = row["median_targets_per_second"]
            for epoch in (2, 3, 5):
                hours = epoch * TARGETS / rate / 3600
                key = f"epoch{epoch}_hours"
                if key in row:
                    assert abs(hours - row[key]) < 1e-10
    c = {row["case"]: row for row in cpu["summary"]}
    g = {row["case"]: row for row in gpu["summary"]}
    paired = {}
    for t in (4, 6):
        comparisons = []
        for repeat in (1, 2, 3):
            a = next(row["measurement"] for row in cpu["trials"] if row["case"] == f"baseline_t{t}" and row["repeat"] == repeat)
            b = next(row["measurement"] for row in cpu["trials"] if row["case"] == f"ilp_t{t}" and row["repeat"] == repeat)
            comparisons.append({"repeat": repeat,
                                "wall_time_reduction_percent": (1-b["elapsed_seconds"]/a["elapsed_seconds"])*100,
                                "softmax_phase_reduction_percent": (1-b["phase_wall_seconds"]["softmax"]/a["phase_wall_seconds"]["softmax"])*100,
                                "baseline_seconds": a["elapsed_seconds"], "candidate_seconds": b["elapsed_seconds"]})
        paired[str(t)] = {"comparisons": comparisons,
                          "median_wall_time_reduction_percent": (1-c[f"ilp_t{t}"]["median_seconds"]/c[f"baseline_t{t}"]["median_seconds"])*100,
                          "all_three_wall_observations_improved": all(r["wall_time_reduction_percent"] > 0 for r in comparisons),
                          "all_three_softmax_phases_improved": all(r["softmax_phase_reduction_percent"] > 0 for r in comparisons),
                          "decision": "Improvement observed at six threads, but strong control variability prevents a universal fixed speedup claim. Four-thread signs are mixed; retain as experimental candidate."}
    memory = {"cpu_batch512_peak_rss_bytes": peak_rss(cpu, "batch512_t6"),
              "cpu_ilp6_peak_rss_bytes": peak_rss(cpu, "ilp_t6"),
              "mps_combination_peak_rss_bytes": peak_rss(gpu, "metal_compiled_zero"),
              "scope": "Maximum observed process RSS across three executions, includes setup/final audits. MPS shares unified RAM; driver snapshots must not be added to RSS. Not system-wide RAM or UI responsiveness."}
    memory_label = (f"관측 최대 RSS · CPU 배치512 {memory['cpu_batch512_peak_rss_bytes']/2**20:.0f} MiB"
                    f" · MPS 조합 {memory['mps_combination_peak_rss_bytes']/2**20:.0f} MiB")
    earlier = read("mps-optimization-comparison-20261018-v1.json")
    old = next(row for row in earlier["summary"] if row["case"] == "c_cpu")
    b = {r['hidden_dim']: r for r in budget['summary']}
    budget_memory = {}
    for hidden in (128,64):
        values = [int(re.search(r'(\d+)\s+maximum resident set size',Path(r['stderr_log']).read_text()).group(1))
                  for r in budget['trials'] if r['hidden_dim']==hidden]
        budget_memory[str(hidden)] = max(values)
    memory['budget_process_peak_rss_bytes'] = budget_memory
    memory_label = (f"확인 측정 최대 RSS · H128 {budget_memory['128']/2**20:.0f} MiB"
                    f" · H64 {budget_memory['64']/2**20:.0f} MiB · MPS 조합 {memory['mps_combination_peak_rss_bytes']/2**20:.0f} MiB")
    report = {"kind": "NPLM_ADDITIONAL_OPTIMIZATION_REVIEW", "status": "complete",
              "created_at": datetime.now().astimezone().isoformat(), "cpu": cpu, "gpu": gpu,
              "thread_summary": threads["summary"], "memory": memory, "memory_label": memory_label,
              "budget": budget, "budget_gate": {k:budget_gate[k] for k in ('status','passed','comparison_count','failed_comparisons')},
              "paired_cpu_ilp": paired,
              "gpu_combination_time_reduction_percent": (1-g['metal_compiled_zero']['median_seconds']/g['metal_clip']['median_seconds'])*100,
              "same_cohort_cpu_speed_ratio_to_best_mps": g['metal_compiled_zero']['median_seconds']/g['c_cpu']['median_seconds'],
              "earlier_cpu_reference": {"rate": old["median_targets_per_second"],
                                        "epoch_hours": {str(e): e*TARGETS/old['median_targets_per_second']/3600 for e in (2,3,5)},
                                        "scope": "Earlier cohort, not the denominator for a new improvement claim."},
              "six_eight_hour_requirements": {str(e): {str(h): e*TARGETS/(h*3600) for h in (6,8)} for e in (2,3,5)},
              "decision": {"fixed_B128": "Six-thread ILP is bit-identical over the tested 512 updates, with three shorter softmax/wall observations; keep experimental because background load is highly variable.",
                           "larger_batch": "B512 is the fastest throughput observation in this CPU cohort. It changes Adam update count and learning statistics; quality is unmeasured.",
                           "budget": "Lower-load confirmation at B512: H128 full two epochs extrapolate to about 7.42h; H64 about 6.02h. H128 under previous heavier load extrapolated to 10.57h. These are candidates, not an eight-hour guarantee. H64 changes model capacity and B512 changes update count; evaluate held-out loss in actual training.",
                           "long_training": "Deferred. This bounded C executable supports at most 4096 updates and does not provide the actual epoch/checkpoint/evaluation trainer. No actual 2/3/5 epoch loss, PPL or generation comparison exists.",
                           "word2vec": "Cancelled by latest user direction; NPLM only. No Word2Vec model was trained.",
                           "unmeasured_devices": "Intel i5-8250U and NVIDIA not measured. Access deferred; no cloud rented."},
              "review_responses": [
                  "Claude Opus5.5 Ultrathink reviewed advice, implementation, and final code. Root retained failures and unchanged tolerances.",
                  "Strong load/control variability is recorded. Four-thread ILP is inconclusive. Six-thread shorter observations and softmax phase reductions are observations, not proof of a fixed general speedup.",
                  "GPU and CPU include their real input preparation. GPU preload/transfer is included, CPU gathers per update. Matched initial/input/sampler work, not identical input scheduling.",
                  "Small array storage does not establish that system RAM is unconstrained; other apps and unified memory compete. Only measured process RSS is reported.",
                  "Benchmark throughput and a short training loss cannot prove convergence or generation quality. Changing epochs does not imply a known quality penalty."],
              "validation": {"cpu_thread_cases": 24, "cpu_ilp_model_cases": 24, "gpu_gates": 24,
                             "finite_softmax_cases": unit["finite_cases"],
                             "softmax_bit_differences": unit["gradient_bit_differences"]+unit["loss_bit_differences"],
                             "full_B512_H128_H64_gates": 4, "full_B512_comparisons": 128,
                             "short_wiki_timing_observations": len(cpu["trials"])+len(gpu["trials"])+len(threads["trials"])+12,
                             "ilp_final_state_bit_identical": cpu["ilp_final_state_bit_identical"]}}
    destination = REPORTS / "nplm-additional-time-summary.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    text = ["NPLM 추가 최적화: 실제 짧은 측정과 2/3/5에폭 환산", "",
            "모든 시간은 단기 처리량 외삽입니다. 전체 2/3/5에폭 학습이나 해당 Loss 측정이 아닙니다.",
            "1epoch=124,232,535 target predictions. 전체 데이터·사전·숫자분리·FP32는 유지했습니다.",
            "실측 당시 부하가 달라 과거27.4h와 이번 코호트를 섞어 개선율을 만들지 않습니다.", "",
            "CPU 추가실험 코호트 | 실제3회초 | 2ep h | 3ep h | 5ep h"]
    for row in cpu["summary"]:
        text.append(f"{row['case']} | {', '.join(f'{x:.6f}' for x in row['actual_seconds'])} | "
                    f"{row['epoch2_hours']:.3f} | {row['epoch3_hours']:.3f} | {row['epoch5_hours']:.3f}")
    text += ["", "GPU 조합 코호트(별도CPU대조) | 실제3회초 | 2ep h | 3ep h | 5ep h"]
    for row in gpu["summary"]:
        h = row['epoch5_hours_extrapolated']
        text.append(f"{row['case']} | {', '.join(f'{x:.6f}' for x in row['actual_seconds'])} | {h*2/5:.3f} | {h*3/5:.3f} | {h:.3f}")
    text += ['', '예산 확인 코호트: 동일 어휘/전체자료/B512/6스레드, 공통 256update/131072타깃, 3회씩. H64는 용량변경.']
    for row in budget['summary']:
        text.append(f"H{row['hidden_dim']} params={row['parameter_count']} | {row['actual_seconds']}s | "
                    + ' | '.join(f"{e}ep {row['epoch_hours'][str(e)]:.3f}h" for e in (2,3,5)))
    text += ["", "기존 별도 코호트 CPU4 참고: " + json.dumps(report['earlier_cpu_reference']['epoch_hours']),
             "", memory_label, memory['scope'], "",
             "CPU ILP6: 갱신512회 후 weights/m1/m2가 대조군과 bit동일. 3회 모두 softmax/wall구간 감소 관측.",
             "CPU ILP4: 실행간 부호가 엇갈려 속도 개선 판정 보류. 대조군 자체 편차가 큽니다.",
             "B256/B512: 타깃65536개는 같지만 Adam갱신수가512→256/128로 감소. 학습품질 미검증.",
             "MPS: best조합은 Metalclip+Nonegradient+modelcompile. 같은 코호트CPU4가 여전히 빠릅니다.",
             "6~8h 후보: 확인 코호트 H128/B512 전체2epoch 7.42h, H64/B512 6.02h 환산. 같은 H128/B512도 부하 높은 앞 코호트에서는 10.57h였습니다.",
             "권고: 용량유지는 H128/B512 2epoch가 목표에 가깝지만 8h 여유가 작습니다. H64는 시간 여유를 늘리는 별도 NPLM 후보이며 품질 저하 가능성을 검증Loss로 판단해야 합니다.",
             "5epoch/3epoch은 확인 코호트에서도 두 후보 모두8h 초과. 실제 장기 실행 시 시간상한·저장·평가가 필요합니다.",
             "2epoch는 전체자료를2회,5epoch는5회 사용합니다. 2epoch품질이 얼마나 낮은지와5epoch이득은 검증Loss없어 모릅니다.",
             "장기 발열/평가/저장/다른앱 지연은 환산에 미반영. 최종 후보의 실제 장시간 실행에서 확인해야 합니다.",
             "현재C는 최대4096update의 측정기입니다. 전체학습/체크포인트/평가 통합은 다음챕터 대상입니다.",
             "Intel/NVIDIA: 미측정. 접속/대여 없음. Word2Vec: 사용자 지시로 중단, 모델학습 없음.",
             "수치검사72개+전체B512H128/H64별도4gate128비교 모두통과, 단위유한23861개, 실제위키짧은측정54회(초기짧은pilot6회포함). 원본과Gitindex 보존 감사는 별도audit참조."]
    (REPORTS / "nplm-additional-time-summary.txt").write_text("\n".join(text) + "\n")
    compact = {"cpu": {"summary": cpu["summary"]}, "gpu": {"summary": gpu["summary"]},
               "budget": {"summary":budget['summary']}, "memory_label": memory_label}
    fragment = (REPORTS / "nplm-additional-time.template.html").read_text().replace('__REPORT_DATA__',json.dumps(compact,ensure_ascii=False))
    assert '__REPORT_DATA__' not in fragment
    (REPORTS / "nplm-additional-time.html").write_text(fragment)
    shell = (REPORTS / "visualization-shell.html").read_text()
    (REPORTS / "nplm-additional-time.preview.html").write_text(shell.replace('__CONTENT__',fragment))
    pre = read("time-optimization-round2-preflight-20261018.json")
    source_results = {path: identity(Path(path)) == wanted for path,wanted in pre['preserved_sources'].items()}
    index = Path(pre['index']['path'])
    audit = {"created_at": datetime.now().astimezone().isoformat(), "preserved_files": source_results,
             "all_preserved_sources_and_corpora_unchanged": all(source_results.values()),
             "git_index_unchanged": identity(index) == {k:pre['index'][k] for k in ('bytes','sha256')},
             "long_training_status": read('experiment-continuation.json')['status'],
             "long_training_stop_present": (WIKI/'training-runs/wiki-k35328/STOP').exists(),
             "word2vec_environment_removed": not (WIKI/'benchmark-round2/.artifacts/venv-word2vec').exists(),
             "comparison_json": identity(destination),
             "new_code": {str(p): identity(p) for p in [* (WIKI/'benchmark-round2').glob('*.py'),
                          *(WIKI/'gpu-benchmark/optimization_round2').glob('*.py'),
                          *(WIKI/'cpu-benchmark/experiments/round2').glob('*.c')]}}
    assert audit['all_preserved_sources_and_corpora_unchanged'] and audit['git_index_unchanged']
    assert audit['long_training_status']=='stopped' and audit['long_training_stop_present']
    (REPORTS/'nplm-additional-time-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({"summary":str(destination),"validation":report['validation'],"memory":memory,
                      "gpu_time_reduction_percent":report['gpu_combination_time_reduction_percent'],
                      "preserved":audit['all_preserved_sources_and_corpora_unchanged'],"index":audit['git_index_unchanged']},indent=2))


if __name__ == '__main__':
    main()
