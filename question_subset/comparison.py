"""Use identical document/hypothesis rows and Q8 exposure groups in all arms."""

import random
from collections import defaultdict
from pathlib import Path

from contract_experiment import CONDITION_NAMES, CONDITIONS
from contract_experiment.experiment import validated_results
from contract_experiment.metrics import confusion_counts, percentile, scores
from contract_experiment.storage import read_json, read_jsonl, run_lock, unique_index, write_csv, write_json

from .state import CONTEXT_KEY, JOURNALS, PREDICTION_KEY, load_subset_run, reference_paths


ARMS = ("q0", "q8", "q17")
CONTRASTS = ("q0", "q17")
METRICS = ("accuracy", "macro_f1")
SCOPE_NAMES = {"all": "전체", "shown_in_q8": "Q8 제공", "omitted_in_q8": "Q8 미제공"}


def pair_predictions(predictions, assignment):
    indices = {arm: unique_index(predictions[arm], PREDICTION_KEY) for arm in ARMS}
    if set(indices["q0"]) != set(indices["q8"]) or set(indices["q17"]) != set(indices["q8"]):
        raise ValueError("Q0, Q8 and Q17 must have identical judgment keys.")
    pairs = []
    for key, q8 in indices["q8"].items():
        if any(indices[arm][key]["gold"] != q8["gold"] for arm in CONTRASTS):
            raise ValueError(f"Gold mismatch across arms: {key}")
        document, condition, hypothesis = key
        if condition == "original":
            continue
        pair = {"document": document, "condition": condition, "hypothesis": hypothesis,
                "shown_in_q8": hypothesis in assignment[document], "gold": q8["gold"]}
        for arm in ARMS:
            pair[f"prediction_{arm}"] = indices[arm][key]["prediction"]
            pair[f"status_{arm}"] = indices[arm][key]["status"]
        pairs.append(pair)
    return pairs


def counts(pairs, arm):
    return confusion_counts({"gold": row["gold"], "prediction": row[f"prediction_{arm}"]} for row in pairs)


def effect(pairs):
    results = {arm: scores(counts(pairs, arm)) for arm in ARMS}
    return {"questions": len(pairs), **{
        f"{metric}_{arm}": results[arm][metric] for metric in METRICS for arm in ARMS}, **{
        f"{metric}_q8_minus_{arm}": results["q8"][metric] - results[arm][metric]
        for metric in METRICS for arm in CONTRASTS}}


def bootstrap_weights(documents, repetitions, seed):
    import numpy as np

    rng = random.Random(seed)
    population = range(len(documents))
    return np.stack([np.bincount(rng.choices(population, k=len(documents)), minlength=len(documents))
                     for _ in range(repetitions)])


def intervals(pairs, documents, weights):
    import numpy as np

    grouped = defaultdict(list)
    for row in pairs:
        grouped[row["document"]].append(row)
    draws = {}
    for arm in ARMS:
        by_document = np.array([counts(grouped[doc], arm) for doc in documents])
        # The same document weights are applied to every arm and exposure group.
        values = [scores(counts_row) for counts_row in (weights @ by_document).tolist()]
        draws[arm] = {metric: np.array([sample[metric] for sample in values]) for metric in METRICS}
    result = {}
    for arm in CONTRASTS:
        for metric in METRICS:
            low, high = np.quantile(draws["q8"][metric] - draws[arm][metric], [0.025, 0.975]).tolist()
            result.update({f"{metric}_q8_minus_{arm}_ci_low": low, f"{metric}_q8_minus_{arm}_ci_high": high})
    return result


def resource_rows(paths, predictions, documents):
    rows = []
    for arm in ARMS:
        contexts = unique_index(read_jsonl(paths[arm] / JOURNALS[0]), CONTEXT_KEY)
        if set(contexts) != {(doc, condition) for doc in documents for condition in CONDITIONS}:
            raise ValueError(f"{arm.upper()} contexts are incomplete. Complete compression before analysis.")
        for condition in CONDITIONS[1:]:
            selected = [contexts[(doc, condition)] for doc in documents]
            judgments = [row for row in predictions[arm] if row["condition"] == condition]
            lengths = [row["retention"] for row in selected if row["retention"] is not None]
            rows.append({"condition": condition, "arm": arm, "reused": arm != "q8",
                         "retention_mean": sum(lengths) / len(lengths) if lengths else None,
                         "retention_median": percentile(lengths, 0.5), "valid_length_documents": len(lengths),
                         "compression_failures": sum(row["status"] != "ok" for row in selected),
                         "parse_failures": sum(row["status"] == "parse_failed" for row in judgments),
                         "model_judgments": sum(row["generation"] is not None for row in judgments),
                         "compression_seconds": sum(row["compression_seconds"] for row in selected),
                         "judgment_seconds": sum(row["judgment_seconds"] for row in judgments)})
    return rows


def report(path, summary, effects):
    lines = ["# Q8 가설 목록 제공 결과", "",
             f"Test {summary['documents']}문서. 새 압축 {summary['new_compressions']}건, "
             f"새 판단 기록 {summary['new_judgment_records']}건. Original {summary['reused_original_judgments']}건 재사용.", "",
             "동일 문서·가설·Gold에 Q8의 제공 여부를 붙여 세 조건을 비교했다. 전체가 주 비교이며 제공·미제공은 보조 비교다.", "",
             "| 방법 | 평가 문항 | 건수 | Q0 | Q8 | Q17 | Q8−Q0 %p [95% CI] | Q8−Q17 %p [95% CI] |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in effects:
        accuracy = [f"{row[f'accuracy_{arm}']:.2%}" for arm in ARMS]
        differences = []
        for arm in CONTRASTS:
            key = f"accuracy_q8_minus_{arm}"
            differences.append(f"{100 * row[key]:+.2f} [{100 * row[key + '_ci_low']:+.2f}, {100 * row[key + '_ci_high']:+.2f}]")
        cells = [CONDITION_NAMES[row["condition"]], SCOPE_NAMES[row["scope"]], str(row["questions"]), *accuracy, *differences]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Macro-F1은 `subset_effects.csv`, 실제 보존율·형식 실패·시간은 `subset_resources.csv`에 기록했다.",
              "신뢰구간은 고정된 가설 배정에서 문서를 묶어 추정한 탐색적 구간이다. 가설 조합과 자연 출력 길이의 영향도 결과에 포함될 수 있다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def compare(path):
    with run_lock(Path(path)):
        run, manifest, config, _, dataset = load_subset_run(path)
        paths = {**reference_paths(run), "q8": run}
        assignment = read_json(run / "assignment.json")["by_document"]
        predictions = {arm: validated_results(paths[arm], dataset) for arm in ARMS}
        pairs = pair_predictions(predictions, assignment)
        documents = sorted(assignment)
        repetitions, seed = (config["analysis"][k] for k in ("bootstrap_repetitions", "bootstrap_seed"))
        weights = bootstrap_weights(documents, repetitions, seed)
        effects, hypotheses = [], []
        for condition in CONDITIONS[1:]:
            method = [pair for pair in pairs if pair["condition"] == condition]
            groups = {"all": method, "shown_in_q8": [p for p in method if p["shown_in_q8"]],
                      "omitted_in_q8": [p for p in method if not p["shown_in_q8"]]}
            for scope, selected in groups.items():
                effects.append({"condition": condition, "scope": scope, **effect(selected),
                                **intervals(selected, documents, weights)})
                if scope != "all":
                    for hid in dataset["hypotheses"]:
                        same_hypothesis = [row for row in selected if row["hypothesis"] == hid]
                        if same_hypothesis:
                            hypotheses.append({"condition": condition, "scope": scope, "hypothesis": hid,
                                               **effect(same_hypothesis)})
        resources = resource_rows(paths, predictions, documents)
        summary = {"experiment": manifest["experiment"], "documents": len(documents),
                   "hypotheses_evaluated_per_document": manifest["hypotheses_per_document"],
                   "new_compressions": manifest["expected_new_compressions"],
                   "new_judgment_records": manifest["expected_new_judgments"],
                   "new_model_judgments": sum(r["model_judgments"] for r in resources if r["arm"] == "q8"),
                   "reused_original_judgments": manifest["reused_original_judgments"],
                   "original": scores(confusion_counts(row for row in predictions["q0"] if row["condition"] == "original")),
                   "bootstrap": {"unit": "document", "paired": True, "seed": seed, "repetitions": repetitions,
                                 "interval": "percentile 95%", "multiple_comparison_adjustment": False},
                   "group_policy": "All arms use the same document/hypothesis keys and Q8 exposure mask.",
                   "failure_policy": "Compression failures and invalid predictions count as incorrect.",
                   "length_policy": "Natural output lengths; Qwen3 context tokens divided by original tokens.",
                   "time_policy": "Q0/Q17 times are historical references; all arms exclude model loading."}
        out = run / "results"
        for filename, rows in (("subset_effects.csv", effects), ("subset_hypothesis_effects.csv", hypotheses),
                               ("paired_predictions.csv", pairs), ("subset_resources.csv", resources)):
            write_csv(out / filename, rows, list(rows[0]))
        write_json(out / "subset_analysis.json", summary)
        report(out / "subset_report.md", summary, effects)
    print(f"Paired Q0/Q8/Q17 results: {out}")
