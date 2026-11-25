# Main experiment results

Documents: 119; judgments: 2023.

| Condition | Accuracy | Macro-F1 | Median retention | Parsing failures |
|---|---:|---:|---:|---:|
| Original | 63.72% | 0.5255 | 100.00% | 0 |

The main experiment uses natural output lengths. It does not establish superiority at equal lengths.
Invalid predictions and compression failures remain in the performance denominator.
Lengths exclude prompts, hypotheses and unit ID markers. Unavailable lengths are reported separately.
See comparisons.csv for paired document-cluster confidence intervals of Original minus each method.
A confidence interval is uncertainty in the paired difference, not proof of a causal mechanism.


## Attention memory execution record

All Test compression and judgments were generated under the same frozen memory policy from the first model call. No prior predictions or compressed contexts were reused. Full inputs and base model settings were preserved. Old Dev output equality was descriptive, not an acceptance criterion. Read runtime_memory_policy.json for source hashes and technical validation. Short/long inputs follow the same fixed dispatch rule; their operation counts and timing naturally differ.
