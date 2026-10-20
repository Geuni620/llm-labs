"""Summarize completed observations without running or importing Torch."""

from datetime import datetime
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def main():
    measured = json.loads((REPORTS / "mps-optimization-comparison-20261018-v1.json").read_text())
    gates = json.loads((REPORTS / "mps-optimization-refined-verification-20261018-v1.json").read_text())
    rejected = json.loads((REPORTS / "mps-optimization-verification-20261018-v1.json").read_text())
    assert measured["status"] == "complete" and all(row["passed"] for row in measured["summary"])
    assert gates["passed"] and gates["passed_gates"] == 24 and gates["comparison_count"] == 768
    assert rejected["failed_comparisons"] == 4 and not rejected["passed"]
    assert len(measured["trials"]) == 15 and all(row["passed"] for row in measured["trials"])
    labels = {"c_cpu": ("C/CPU (기존 최적화)", "C/CPU"), "control": ("기존 MPS", "기존 MPS"),
              "zero_none": ("기울기 None 초기화", "grad=None"),
              "metal_clip": ("Metal 블록 리덕션", "Metal 리덕션"),
              "compiled_model": ("모델 컴파일", "모델 컴파일")}
    rows = []
    for summary in measured["summary"]:
        trials = [row["measurement"] for row in measured["trials"] if row["case"] == summary["case"]]
        assert len(trials) == 3
        for row in trials:
            assert row["steps"] == 512 and row["target_prediction_events"] == 65536
            assert math.isclose(row["targets_per_second"], 65536/row["elapsed_seconds"], rel_tol=1e-12)
        seconds = [row["elapsed_seconds"] for row in trials]
        hours = 621162675 * statistics.median(seconds)/65536/3600
        assert math.isclose(hours,summary["epoch5_hours_extrapolated"],rel_tol=1e-12)
        rows.append({"id": summary["case"], "label": labels[summary["case"]][0],
                     "axisLabel": labels[summary["case"]][1], "seconds": statistics.median(seconds),
                     "observations": seconds, "range": [min(seconds),max(seconds)], "hours": hours,
                     "hoursRange": summary["epoch5_hours_range"],
                     "rate": statistics.median([row["targets_per_second"] for row in trials])})
    control = next(row for row in rows if row["id"] == "control")
    best = min((row for row in rows if row["id"] != "c_cpu"),key=lambda row:row["seconds"])
    first_updates = [trial["measurement"]["warmup_seconds"][0] for trial in measured["trials"]
                     if trial["case"] == "compiled_model"]
    data = {"rows": rows, "gates": 24, "comparisons": 768,
            "compileLabel": "모델 컴파일: 실제 forward/backward 확인 · 측정 중 재컴파일 0",
            "bestGpu": best["id"], "bestGpuTimeReductionPercent": (1-best["seconds"]/control["seconds"])*100}
    report = {"status": "complete", "created_at": datetime.now().astimezone().isoformat(), "visual_data": data,
              "fixed_conditions": "V35328/C8/D64/H128, B128, FP32, seed42, exact C initializer and affine Wiki target stream, AdamW foreach=True/fused=False, clip1, same512 updates/65536 predictions",
              "reference_cpu": "Same existing optimized C executable rerun in this cohort; no new CPU optimization.",
              "actual_training": "No full epoch or long training. Previous training remains deferred to the next chapter.",
              "forecast": "621162675 / measured targets per second / 3600; excludes setup/loading/warmup/reset/evaluation/checkpointing; long-run heat and UI responsiveness unmeasured.",
              "independent_options": "Each GPU candidate changes exactly one control option. Their combination was not tested.",
              "accepted_gate_scope": "Two small geometries and the full real-Wiki geometry, each at clip1 and.05, four consecutive updates after2 warmups and exact reset; independent CPU oracle.",
              "native_fused": {"status": "not_adopted_under_existing_tolerance", "prior_campaign_attempted_gates":30,
                               "prior_campaign_comparisons":960,"native_fused_candidate_gates":24,
                               "native_fused_candidate_comparisons":768,"failed_comparisons":4,
                               "failure_scope":"One moment1 element at full-Wiki update4/clip1 in each native-fused candidate; exact failing index not present in saved summaries. No tolerance relaxed. This is not evidence of degraded learned quality.",
                               "speed_measurement":"Excluded because the numerical gate failed."},
              "compile_initial_update_seconds_observed":first_updates,
              "compile_setup_scope":"Shared cache; initial update includes compilation or cache loading plus execution, not pure cold compiler time.",
              "recommendation":{"gpu_case":best["id"],"shortest_observed_case":min(rows,key=lambda row:row["seconds"])["id"],
                                "quality":"Short numerical tests and throughput do not establish long-run quality equivalence."},
              "evidence":["reports/mps-optimization-comparison-20261018-v1.json",
                          "reports/mps-optimization-refined-verification-20261018-v1.json",
                          "reports/mps-optimization-verification-20261018-v1.json",
                          "reports/mps-optimization-phase-profile-20261018-v3.json"]}
    with (REPORTS/"mps-optimization-summary.json").open("x") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    lines=["M1 GPU 최적화 비교 — 실제 짧은 측정 / 5에폭 외삽", "",
           "조건: V35328/C8/D64/H128, B128, FP32, AdamW foreach=True/fused=False, clip1, seed42.",
           "각 실행은 동일한512회 갱신 / 65,536개 예측. 각조건3회, 순서정방향/역방향/정방향.",
           "원본.npy 전체를 RAM으로 읽지만 일부 target만 측정. 5에폭은621,162,675개 예측에서 환산.", "",
           "조건\t실제초 3회\t중앙초\t예측/초\t5에폭 환산시간\t기존 MPS 대비 시간변화"]
    for row in rows:
        change="CPU참고" if row["id"]=="c_cpu" else f"{(row['seconds']/control['seconds']-1)*100:+.2f}%"
        lines.append(f"{row['label']}\t{', '.join(f'{v:.6f}' for v in row['observations'])}\t{row['seconds']:.6f}\t{row['rate']:.2f}\t{row['hours']:.3f}\t{change}")
    lines += ["",f"현재 GPU 후보의 최선: {best['label']} / 기존 MPS 대비 중앙 처리시간 {data['bestGpuTimeReductionPercent']:.2f}% 감소.",
              "GPU는같은학습조건에서옵션을하나씩변경했으며옵션조합은미측정입니다.",
              "GPU후보24개 gate·768개 vector/scalar비교 통과. 원래오차허용기준 유지.",
              "1차 전체 캠페인(control포함)은30gate/960비교. Native fused AdamW4후보(24gate/768비교)의각1성분이기준을넘어기본채택과속도비교에서제외. 기존실패보고서는보존.",
              "기울기 None 초기화: grad=None으로설정하여0을덮어쓰는메모리작업을줄입니다.",
              "이실패만으로학습품질저하나지원불가를단정하지않습니다. 실제fused API/컴파일은작동했습니다.",
              "Metal clip: 256threads×4elements 블록부분합→최종norm/scale→제자리scaling, GPU커널3개. scratch NaNpoison·scalar history·두clip·실제모델4연속갱신검증.",
              "모델compile: strict Inductor, fullgraph=True/dynamic=False, 실제fwd/back 및codegen/cache확인, 측정중재컴파일없음.",
              "초기실행시간은공유cache의컴파일또는캐시로딩+실행을포함하며순수cold compile시간이아닙니다.",
              "구간진단은phase마다동기화를강제하여오버헤드가있습니다. 그phase비율을무계측훈련시간·GPU사용률로해석하면안됩니다.",
              "C/CPU는동일기존코드재측정입니다. PyTorch도수치연산은GPU커널에서실행하며커널구성·호출·메모리·동기화경로차이가이번비교에포함됩니다.",
              "실제전체5에폭/장기품질/장시간발열/MacUI반응성은측정하지않았습니다. 장시간학습은다음챕터로보류한상태입니다.",
              "", "재사용 API: gpu-benchmark/optimization/engine_refined.py의 build_model_optimizer(config,case), train_update, reset_after_warmup.",
              "권장GPU case: "+best['id'], "", "공식참고: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations"]
    with (REPORTS/"mps-optimization-summary.txt").open("x") as stream:stream.write("\n".join(lines)+"\n")
    template=(REPORTS/"mps-optimization.template.html").read_text()
    assert template.count("__OPT_DATA__")==1
    fragment=template.replace("__OPT_DATA__",json.dumps(data,ensure_ascii=False,allow_nan=False))
    with (REPORTS/"mps-optimization.html").open("x") as stream:stream.write(fragment)
    shell=(REPORTS/"visualization-shell.html").read_text()
    with (REPORTS/"mps-optimization.preview.html").open("x") as stream:stream.write(shell.replace("__CONTENT__",fragment))
    print(json.dumps({"status":"complete","best_gpu":best["id"],"time_cut_percent":data["bestGpuTimeReductionPercent"],"rows":rows},ensure_ascii=False))


if __name__=="__main__":main()
