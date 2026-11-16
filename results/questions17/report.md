# Additional analysis 1: fixed 17-hypothesis list

Documents: 119. New compression: 476. New judgment records: 8092. Reused Original: 2023.

All 17 hypotheses are provided together to the compressor once per document/method. The judge receives one hypothesis per call. Main model, quantization, decoder, memory implementation and base task prompts are preserved.

| Method | Q0 accuracy | Q17 accuracy | Gain (pp) | Paired 95% CI (pp) | Q0 Macro-F1 | Q17 Macro-F1 |
|---|---:|---:|---:|---|---:|---:|
| 생성형·일반 (Naive) | 61.05% | 63.67% | 2.62 | [0.59, 4.75] | 0.5101 | 0.5450 |
| 생성형·보존 우선 (Contextualized) | 61.34% | 60.80% | -0.54 | [-2.67, 1.58] | 0.5119 | 0.5089 |
| 추출형·일반 | 61.84% | 60.41% | -1.43 | [-3.51, 0.74] | 0.5000 | 0.4958 |
| 추출형·보존 우선 | 62.14% | 59.96% | -2.17 | [-4.10, -0.30] | 0.4978 | 0.4891 |

| Method | Q0 median retention | Q17 median retention | Q0 compression failures | Q17 compression failures |
|---|---:|---:|---:|---:|
| 생성형·일반 (Naive) | 66.26% | 61.23% | 0 | 0 |
| 생성형·보존 우선 (Contextualized) | 40.90% | 50.34% | 0 | 0 |
| 추출형·일반 | 90.60% | 46.98% | 1 | 0 |
| 추출형·보존 우선 | 79.89% | 49.77% | 0 | 0 |

The estimand is awareness of a fixed hypothesis list, not per-question tailored compression. Gold labels/evidence are never supplied to the compressor. Lengths are natural outputs, so an accuracy gain does not isolate a length-independent mechanism. Failure-rate changes may contribute to the gain. All questions remain in the primary denominator.

Confidence intervals use the same resampled documents for Q0 and Q17 and recompute Macro-F1 every iteration. They are not multiplicity-adjusted. See question_effects.csv for Macro-F1 gain intervals, decision flips, and failure counts.

New Test work recorded (excluding Original reuse, loading, preflight and pauses): 8.02 hours.

question_label_effects.csv reports class-level recall/label shifts. question_times.csv distinguishes reused reference costs and new execution costs. Generic metrics.csv describes Q17 and the reused Original baseline; comparisons.csv is Original minus Q17, not the Q17-minus-Q0 question effect.
