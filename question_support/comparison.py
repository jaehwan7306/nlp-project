"""Paired document-level effects of providing the fixed hypothesis list."""

import random
from collections import Counter

from contract_experiment import CONDITIONS, CONDITION_NAMES, LABELS
from contract_experiment.experiment import validated_results
from contract_experiment.metrics import confusion_counts, percentile, scores
from contract_experiment.storage import read_json, read_jsonl, unique_index, write_csv, write_json
from .state import inspect_main, load_question_run


def paired_bootstrap(main_rows, question_rows, documents, conditions, repetitions, seed):
    if not documents or repetitions < 1:
        raise ValueError("Bootstrap needs documents and positive repetitions.")
    main = unique_index(main_rows, ("document", "condition", "hypothesis"))
    question = unique_index(question_rows, ("document", "condition", "hypothesis"))
    if set(main) != set(question) or any(main[key]["gold"] != question[key]["gold"] for key in main):
        raise ValueError("Q0 and Q17 require the same document/hypothesis/Gold keys.")
    vectors = {c: [] for c in conditions}
    for c in conditions:
        for doc in documents:
            vectors[c].append((confusion_counts([r for k, r in main.items() if k[:2] == (doc, c)]),
                               confusion_counts([r for k, r in question.items() if k[:2] == (doc, c)])))
    samples = {c: {"accuracy": [], "macro_f1": []} for c in conditions}
    rng = random.Random(seed)
    for _ in range(repetitions):
        selected = rng.choices(range(len(documents)), k=len(documents))
        for c in conditions:
            totals = [[0] * 12, [0] * 12]
            for index in selected:
                for arm in (0, 1):
                    for cell, value in enumerate(vectors[c][index][arm]):
                        totals[arm][cell] += value
            q0, q17 = scores(totals[0]), scores(totals[1])
            for metric in samples[c]:
                samples[c][metric].append(q17[metric] - q0[metric])
    return {c: {metric: [percentile(values, .025), percentile(values, .975)] for metric, values in result.items()} for c, result in samples.items()}


def compare(path):
    run, manifest, config, _, dataset = load_question_run(path)
    lineage = read_json(run / "lineage.json")
    main = inspect_main((run / lineage["main_run_relative"]).resolve())
    q17 = validated_results(run, dataset)
    q0 = list(main["predictions"].values())
    docs = sorted(manifest["document_keys"])
    ci = paired_bootstrap(q0, q17, docs, CONDITIONS[1:], config["analysis"]["bootstrap_repetitions"], config["analysis"]["bootstrap_seed"])
    contexts = unique_index(read_jsonl(run / "contexts.jsonl"), ("document", "condition"))
    q0_index = main["predictions"]
    q17_index = unique_index(q17, ("document", "condition", "hypothesis"))
    effects, label_rows, times = [], [], []
    for c in CONDITIONS:
        rows0 = [r for r in q0 if r["condition"] == c]
        rows17 = [r for r in q17 if r["condition"] == c]
        ct0 = [main["contexts"][(d, c)] for d in docs]
        ct17 = [contexts[(d, c)] for d in docs]
        old_compression = sum(r["compression_seconds"] for r in ct0)
        new_compression = sum(r["compression_seconds"] for r in ct17)
        old_judgment = sum(r["judgment_seconds"] for r in rows0)
        new_judgment = sum(r["judgment_seconds"] for r in rows17)
        times.append({"condition": c, "main_compression_seconds": old_compression, "main_judgment_seconds": old_judgment,
            "q17_compression_seconds": new_compression, "q17_judgment_seconds": new_judgment,
            "additional_actual_seconds": 0.0 if c == "original" else new_compression + new_judgment,
            "original_reused": c == "original"})
        if c == "original":
            continue
        old, new = scores(confusion_counts(rows0)), scores(confusion_counts(rows17))
        lengths0 = [r["retention"] for r in ct0 if r["retention"] is not None]
        lengths17 = [r["retention"] for r in ct17 if r["retention"] is not None]
        pairs = [(q0_index[(d, c, h)], q17_index[(d, c, h)]) for d in docs for h in dataset["hypotheses"]]
        lost = sum(a["prediction"] == a["gold"] and b["prediction"] != b["gold"] for a, b in pairs)
        gained = sum(a["prediction"] != a["gold"] and b["prediction"] == b["gold"] for a, b in pairs)
        row = {"condition": c, "name": CONDITION_NAMES[c], "documents": len(docs), "questions": len(rows17),
            "accuracy_q0": old["accuracy"], "accuracy_q17": new["accuracy"], "accuracy_gain": new["accuracy"] - old["accuracy"],
            "accuracy_gain_ci_low": ci[c]["accuracy"][0], "accuracy_gain_ci_high": ci[c]["accuracy"][1],
            "macro_f1_q0": old["macro_f1"], "macro_f1_q17": new["macro_f1"], "macro_f1_gain": new["macro_f1"] - old["macro_f1"],
            "macro_f1_gain_ci_low": ci[c]["macro_f1"][0], "macro_f1_gain_ci_high": ci[c]["macro_f1"][1],
            "retention_mean_q0": sum(lengths0) / len(lengths0) if lengths0 else None,
            "retention_mean_q17": sum(lengths17) / len(lengths17) if lengths17 else None,
            "retention_median_q0": percentile(lengths0, .5), "retention_median_q17": percentile(lengths17, .5),
            "valid_length_documents_q0": len(lengths0), "valid_length_documents_q17": len(lengths17),
            "compression_failed_documents_q0": sum(r["status"] != "ok" for r in ct0),
            "compression_failed_documents_q17": sum(r["status"] != "ok" for r in ct17),
            "parse_failures_q0": sum(r["status"] == "parse_failed" for r in rows0),
            "parse_failures_q17": sum(r["status"] == "parse_failed" for r in rows17),
            "q0_correct_to_q17_wrong": lost, "q0_wrong_to_q17_correct": gained,
            "prediction_flip_rate": sum(a["prediction"] != b["prediction"] for a, b in pairs) / len(pairs)}
        effects.append(row)
        predicted0 = Counter(r["prediction"] for r in rows0)
        predicted17 = Counter(r["prediction"] for r in rows17)
        for i, label in enumerate(LABELS):
            label_rows.append({"condition": c, "label": label, "recall_q0": old["recall"][i], "recall_q17": new["recall"][i],
                "predictions_q0": predicted0[label], "predictions_q17": predicted17[label]})
    out = run / "results"
    for name, rows in (("question_effects.csv", effects), ("question_label_effects.csv", label_rows), ("question_times.csv", times)):
        write_csv(out / name, rows, list(rows[0]))
    summary = {"experiment": manifest["experiment"], "main_run": str(main["run"].resolve()),
        "documents": len(docs), "hypotheses_per_compression": 17, "hypotheses_per_judgment": 1,
        "reused_original_judgments": manifest["reused_original_judgments"], "new_judgment_records": 68 * len(docs),
        "bootstrap": {**config["analysis"], "unit": "document", "paired": True, "ci": "percentile 95%", "multiple_comparison_adjustment": False},
        "length_control": "none; question effects can include changed summary length",
        "failure_policy": "all documents/questions retained; invalid/compression failures incorrect",
        "incremental_test_seconds": sum(r["additional_actual_seconds"] for r in times), "effects": effects}
    write_json(out / "question_analysis.json", summary)
    generic = read_json(out / "analysis.json")
    generic["question_analysis"] = {k: v for k, v in summary.items() if k != "effects"}
    generic["time_policy"] = "Original rows and their observed time are reused references, not new computation. See question_times.csv for incremental time. Model loading is excluded."
    write_json(out / "analysis.json", generic)
    lines = ["# Additional analysis 1: fixed 17-hypothesis list", "",
        f"Documents: {len(docs)}. New compression: {4*len(docs)}. New judgment records: {68*len(docs)}. Reused Original: {manifest['reused_original_judgments']}.", "",
        "All 17 hypotheses are provided together to the compressor once per document/method. The judge receives one hypothesis per call. Main model, quantization, decoder, memory implementation and base task prompts are preserved.", "",
        "| Method | Q0 accuracy | Q17 accuracy | Gain (pp) | Paired 95% CI (pp) | Q0 Macro-F1 | Q17 Macro-F1 |", "|---|---:|---:|---:|---|---:|---:|"]
    for r in effects:
        lines.append(f"| {r['name']} | {r['accuracy_q0']:.2%} | {r['accuracy_q17']:.2%} | {100*r['accuracy_gain']:.2f} | [{100*r['accuracy_gain_ci_low']:.2f}, {100*r['accuracy_gain_ci_high']:.2f}] | {r['macro_f1_q0']:.4f} | {r['macro_f1_q17']:.4f} |")
    lines += ["", "| Method | Q0 median retention | Q17 median retention | Q0 compression failures | Q17 compression failures |", "|---|---:|---:|---:|---:|"]
    for r in effects:
        a = f"{r['retention_median_q0']:.2%}" if r['retention_median_q0'] is not None else "unavailable"
        b = f"{r['retention_median_q17']:.2%}" if r['retention_median_q17'] is not None else "unavailable"
        lines.append(f"| {r['name']} | {a} | {b} | {r['compression_failed_documents_q0']} | {r['compression_failed_documents_q17']} |")
    lines += ["", "The estimand is awareness of a fixed hypothesis list, not per-question tailored compression. Gold labels/evidence are never supplied to the compressor. Lengths are natural outputs, so an accuracy gain does not isolate a length-independent mechanism. Failure-rate changes may contribute to the gain. All questions remain in the primary denominator.", "",
        "Confidence intervals use the same resampled documents for Q0 and Q17 and recompute Macro-F1 every iteration. They are not multiplicity-adjusted. See question_effects.csv for Macro-F1 gain intervals, decision flips, and failure counts.", "",
        f"New Test work recorded (excluding Original reuse, loading, preflight and pauses): {summary['incremental_test_seconds']/3600:.2f} hours.", "",
        "question_label_effects.csv reports class-level recall/label shifts. question_times.csv distinguishes reused reference costs and new execution costs. Generic metrics.csv describes Q17 and the reused Original baseline; comparisons.csv is Original minus Q17, not the Q17-minus-Q0 question effect.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Q17 versus Main question effects: {out / 'question_effects.csv'}")
