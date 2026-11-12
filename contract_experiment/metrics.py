"""Main metrics with paired document-cluster bootstrap; no model dependencies."""

import random
from collections import defaultdict
from pathlib import Path

from . import CONDITION_NAMES, CONDITIONS, LABELS
from .experiment import validated_results
from .settings import load_run
from .storage import read_jsonl, run_lock, unique_index, write_csv, write_json


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def confusion_counts(rows):
    """3 gold classes × (3 predictions + Invalid); failures remain false negatives."""
    counts = [0] * 12
    for row in rows:
        gold = LABELS.index(row["gold"])
        prediction = LABELS.index(row["prediction"]) if row["prediction"] in LABELS else 3
        counts[gold * 4 + prediction] += 1
    return counts


def scores(counts):
    total = sum(counts)
    accuracy = sum(counts[i * 4 + i] for i in range(3)) / total if total else None
    f1, recalls = [], []
    for i in range(3):
        tp = counts[i * 4 + i]
        actual = sum(counts[i * 4:i * 4 + 4])
        predicted = sum(counts[j * 4 + i] for j in range(3))
        f1.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
        recalls.append(tp / actual if actual else None)
    return {"accuracy": accuracy, "macro_f1": sum(f1) / 3, "recall": recalls}


def cluster_bootstrap(rows, conditions, repetitions=10000, seed=42):
    """Resample shared document IDs; retain every hypothesis and recompute F1."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["document"], row["condition"])].append(row)
    documents = sorted({row["document"] for row in rows})
    if not documents or repetitions < 1:
        raise ValueError("Bootstrap requires documents and positive repetitions.")
    vectors = {condition: [confusion_counts(grouped[(doc, condition)]) for doc in documents]
               for condition in conditions}
    samples = {c: {"accuracy": [], "macro_f1": [], "accuracy_drop": [], "macro_f1_drop": []} for c in conditions}
    rng = random.Random(seed)
    for _ in range(repetitions):
        selected = rng.choices(range(len(documents)), k=len(documents))
        current = {}
        for condition in conditions:
            totals = [0] * 12
            for index in selected:
                for cell, count in enumerate(vectors[condition][index]):
                    totals[cell] += count
            current[condition] = scores(totals)
        for condition in conditions:
            for metric in ("accuracy", "macro_f1"):
                value = current[condition][metric]
                samples[condition][metric].append(value)
                if "original" in current:
                    samples[condition][metric + "_drop"].append(current["original"][metric] - value)
    return {condition: {metric: [percentile(values, 0.025), percentile(values, 0.975)]
                        for metric, values in sample.items() if values}
            for condition, sample in samples.items()}


def analyze(run_path, conditions=None):
    run, manifest, config, models, dataset = load_run(run_path)
    conditions = tuple(conditions or CONDITIONS)
    if len(set(conditions)) != len(conditions) or any(c not in CONDITIONS for c in conditions):
        raise ValueError("Unknown/duplicate conditions.")
    if "original" not in conditions:
        raise ValueError("Include original for paired baseline comparisons.")
    if manifest["split"] != "test":
        raise ValueError("Dev is for technical checks. Use diagnose; main performance analysis is for Test.")
    with run_lock(run):
        rows = validated_results(run, dataset, conditions)
        contexts = unique_index(read_jsonl(run / "contexts.jsonl"), ("document", "condition"))
        expected_contexts = {(doc["key"], c) for doc in dataset["documents"] for c in conditions}
        if expected_contexts - set(contexts):
            raise ValueError("Missing context records.")
        original = {(r["document"], r["hypothesis"]): r for r in rows if r["condition"] == "original"}
        baseline_correct = {key for key, row in original.items() if row["prediction"] == row["gold"]}
        ci = cluster_bootstrap(rows, conditions, **{
            "repetitions": config["analysis"]["bootstrap_repetitions"], "seed": config["analysis"]["bootstrap_seed"]})
        metrics, confusion, times, lengths = [], [], [], []
        observed = {}
        for condition in conditions:
            selected = [r for r in rows if r["condition"] == condition]
            counts = confusion_counts(selected)
            result = scores(counts)
            observed[condition] = result
            condition_contexts = [contexts[(doc["key"], condition)] for doc in dataset["documents"]]
            retention = [r["retention"] for r in condition_contexts if r["retention"] is not None]
            retained = sum((r["document"], r["hypothesis"]) in baseline_correct and r["prediction"] == r["gold"] for r in selected)
            metrics.append({"condition": condition, "name": CONDITION_NAMES[condition], "questions": len(selected),
                "accuracy": result["accuracy"], "accuracy_ci_low": ci[condition]["accuracy"][0], "accuracy_ci_high": ci[condition]["accuracy"][1],
                "macro_f1": result["macro_f1"], "macro_f1_ci_low": ci[condition]["macro_f1"][0], "macro_f1_ci_high": ci[condition]["macro_f1"][1],
                "correct_decision_retention": retained / len(baseline_correct) if baseline_correct else None,
                "parse_failures": sum(r["status"] == "parse_failed" for r in selected),
                "parse_failure_rate": sum(r["status"] == "parse_failed" for r in selected) / len(selected),
                "compression_failed_documents": sum(r["status"] != "ok" for r in condition_contexts),
                "length_available_documents": len(retention),
                "retention_mean": sum(retention) / len(retention) if retention else None,
                "retention_median": percentile(retention, 0.5), "retention_p05": percentile(retention, 0.05),
                "retention_p95": percentile(retention, 0.95), "retention_min": min(retention) if retention else None,
                "retention_max": max(retention) if retention else None,
                **{f"recall_{label}": result["recall"][i] for i, label in enumerate(LABELS)}})
            compression_seconds = sum(r["compression_seconds"] for r in condition_contexts)
            judgment_seconds = sum(r["judgment_seconds"] for r in selected)
            times.append({"condition": condition, "compression_seconds": compression_seconds,
                          "judgment_seconds": judgment_seconds, "total_seconds": compression_seconds + judgment_seconds})
            for gold_index, label in enumerate(LABELS):
                confusion.append({"condition": condition, "gold": label,
                                  **{prediction: counts[gold_index * 4 + j] for j, prediction in enumerate((*LABELS, "Invalid"))}})
            for row in condition_contexts:
                gen = row["generation"] or {}
                lengths.append({"document": row["document"], "condition": condition, "status": row["status"],
                    "original_tokens": row["original_tokens"], "context_tokens": row["context_tokens"], "retention": row["retention"],
                    "stop_reason": gen.get("stop_reason", "source"), "cap_reached": gen.get("cap_reached", False),
                    "selected_unit_count": len(row["selected_ids"]) if row.get("selected_ids") is not None else ""})
        comparisons = [{"condition": condition, "comparison": "Original minus compressed",
            "accuracy_drop": observed["original"]["accuracy"] - observed[condition]["accuracy"],
            "accuracy_drop_ci_low": ci[condition]["accuracy_drop"][0], "accuracy_drop_ci_high": ci[condition]["accuracy_drop"][1],
            "macro_f1_drop": observed["original"]["macro_f1"] - observed[condition]["macro_f1"],
            "macro_f1_drop_ci_low": ci[condition]["macro_f1_drop"][0], "macro_f1_drop_ci_high": ci[condition]["macro_f1_drop"][1]}
            for condition in conditions if condition != "original"]
        predictions = [{k: row[k] for k in ("document", "condition", "hypothesis", "gold", "prediction", "status", "judgment_seconds")} for row in rows]
        out = run / "results"
        if conditions == ("original",):
            out = out / "baseline"
        elif set(conditions) != set(CONDITIONS):
            out = out / ("subset-" + "-".join(conditions))
        for filename, table in (("metrics.csv", metrics), ("confusion_matrices.csv", confusion),
                                ("times.csv", times), ("lengths.csv", lengths), ("predictions.csv", predictions),
                                ("comparisons.csv", comparisons)):
            if table:
                write_csv(out / filename, table, list(table[0]))
        write_json(out / "analysis.json", {"document_count": manifest["document_count"], "conditions": conditions,
            "judgment_count": len(rows), "failure_policy": "All questions retained; invalid predictions are incorrect.",
            "length_policy": "Qwen3 context tokens only; failed contexts have unavailable length, not zero length.",
            "bootstrap": {**config["analysis"], "unit": "document", "ci": "percentile 95%, paired across conditions"},
            "time_policy": "Model loading excluded from condition times; see runtime_*.json and events.jsonl.",
            "metrics": metrics, "comparisons": comparisons})
        lines = ["# Main experiment results", "", f"Documents: {manifest['document_count']}; judgments: {len(rows)}.", "",
                 "| Condition | Accuracy | Macro-F1 | Median retention | Parsing failures |", "|---|---:|---:|---:|---:|"]
        for row in metrics:
            retention = f"{row['retention_median']:.2%}" if row["retention_median"] is not None else "unavailable"
            lines.append(f"| {row['name']} | {row['accuracy']:.2%} | {row['macro_f1']:.4f} | {retention} | {row['parse_failures']} |")
        lines.extend(["", "The main experiment uses natural output lengths. It does not establish superiority at equal lengths.",
            "Invalid predictions and compression failures remain in the performance denominator.",
            "Lengths exclude prompts, hypotheses and unit ID markers. Unavailable lengths are reported separately.",
            "See comparisons.csv for paired document-cluster confidence intervals of Original minus each method.",
            "A confidence interval is uncertainty in the paired difference, not proof of a causal mechanism.", ""])
        (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Main results: {out.resolve()}")
