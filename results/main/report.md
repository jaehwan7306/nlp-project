# 메인 실험 · Original과 Q0 결과

이 파일은 원문 기준선과 질문 없이 압축한 Q0의 결과다. 질문 17개를 모두 제공한 Q17과의 차이가 메인 비교다. [Q17−Q0 결과](../questions17/report.md) · [공개 결과 목록](../README.md#main)

Documents: 119; judgments: 10115.

| Condition | Accuracy | Macro-F1 | Median retention | Parsing failures |
|---|---:|---:|---:|---:|
| Original | 63.72% | 0.5255 | 100.00% | 0 |
| 생성형·일반 (Naive) | 61.05% | 0.5101 | 66.26% | 0 |
| 생성형·보존 우선 (Contextualized) | 61.34% | 0.5119 | 40.90% | 0 |
| 추출형·일반 | 61.84% | 0.5000 | 90.60% | 0 |
| 추출형·보존 우선 | 62.14% | 0.4978 | 79.89% | 0 |

The main experiment uses natural output lengths. It does not establish superiority at equal lengths.
Invalid predictions and compression failures remain in the performance denominator.
Lengths exclude prompts, hypotheses and unit ID markers. Unavailable lengths are reported separately.
See comparisons.csv for paired document-cluster confidence intervals of Original minus each method.
A confidence interval is uncertainty in the paired difference, not proof of a causal mechanism.


## Attention memory execution record

All Test compression and judgments were generated under the same frozen memory policy from the first model call. No prior predictions or compressed contexts were reused. Full inputs and base model settings were preserved. Old Dev output equality was descriptive, not an acceptance criterion. Read runtime_memory_policy.json for source hashes and technical validation. Short/long inputs follow the same fixed dispatch rule; their operation counts and timing naturally differ.
