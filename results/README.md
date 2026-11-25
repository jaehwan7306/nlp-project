# 공개 결과

[프로젝트 개요](../README.md) · [메인 실험 해설](../docs/02_main_q0_vs_q17.md) · [분석 1 해설](../docs/03_analysis1_q8.md) · [팀 분석용 파일 안내](../docs/reproducibility.md#team-data)

이 폴더는 실행이 만든 결과 파일의 색인이다. 결과 해석은 `docs/`의 각 문서에 있다. 모든 압축 조건은 ContractNLI Test **119문서 · 2,023문항**을 공유하며, CSV의 0.30은 30%를 뜻한다. 압축·파싱 실패 문항은 정확도와 Macro-F1의 분모에 포함한다.

## 공통

| 파일 | 내용 |
|---|---|
| [dataset.json](dataset.json) | 실제 평가한 119문서의 원문·원문 단위·17개 가설·정답·Gold Evidence 위치 |
| [manifest.json](manifest.json) | 공개 파일별 SHA256, 원본 실행 경로, 공개용 편집 내역 |

<a id="main"></a>

## main/ · 메인 실험 (Q0 vs Q17)

| 폴더 | 내용 | 주요 파일 |
|---|---|---|
| [original/](main/original/) | 원문 기준선 단독 집계 | [metrics.csv](main/original/metrics.csv) · [predictions.csv](main/original/predictions.csv) |
| [q0/](main/q0/) | Original과 질문 없는 압축 4조건 | [metrics.csv](main/q0/metrics.csv) · [comparisons.csv](main/q0/comparisons.csv)(Original − 각 조건) · [predictions.csv](main/q0/predictions.csv) · [contexts.jsonl](main/q0/contexts.jsonl) |
| [q17/](main/q17/) | 가설 17개 제공 압축 4조건 + Original | [question_effects.csv](main/q17/question_effects.csv)(**Q17 − Q0**) · [question_label_effects.csv](main/q17/question_label_effects.csv) · [predictions.csv](main/q17/predictions.csv) · [contexts.jsonl](main/q17/contexts.jsonl) |
| [analysis/](main/analysis/) | 사후 분석 요약과 그림 | [comparison.png](main/analysis/comparison.png) · [hypothesis_effects.json](main/analysis/hypothesis_effects.json) · [document_effects.json](main/analysis/document_effects.json) · [format_audit_summary.json](main/analysis/format_audit_summary.json) |

![Q0와 Q17의 정확도, 토큰 보존율, 차이의 신뢰구간](main/analysis/comparison.png)

회색은 Q0, 파랑은 Q17이다. 정확도 그래프의 점선은 Original 63.72%다. Q0와 Q17의 문항별 예측은 각각 10,115행이며, Q17 예측표에는 같은 Original 2,023건이 함께 들어 있다.

<a id="analysis-1"></a>

## analysis1_q8/ · 분석 1 (가설 8개 제공)

| 파일 | 내용 |
|---|---|
| [subset_report.md](analysis1_q8/subset_report.md) · [subset_analysis.json](analysis1_q8/subset_analysis.json) | 실행이 만든 집계 요약과 설정 |
| [subset_effects.csv](analysis1_q8/subset_effects.csv) | 전체·보여 준 가설·빠진 가설별 정확도·Macro-F1, Q8−Q0·Q8−Q17과 CI |
| [subset_hypothesis_effects.csv](analysis1_q8/subset_hypothesis_effects.csv) | 가설별 제공·미제공 비교 |
| [subset_resources.csv](analysis1_q8/subset_resources.csv) | 보존율·실패·시간 |
| [paired_predictions.csv](analysis1_q8/paired_predictions.csv) | 문항별 Q8 제공 여부와 Q0·Q8·Q17 예측 |
| [assignment.json](analysis1_q8/assignment.json) | 문서별 가설 배정(seed 42) |
| [contexts.jsonl](analysis1_q8/contexts.jsonl) · [predictions.jsonl](analysis1_q8/predictions.jsonl) | 실제 압축문과 판단 기록 |

## analysis2_position/ · analysis3_evidence/

분석 2·3의 결과가 들어올 자리다. [분석 2 계획](../docs/04_analysis2_position.md) · [분석 3 계획](../docs/05_analysis3_evidence.md)

## 파일 형식

- `contexts.jsonl`: 문서·조건별 판단 문맥(`text`), 토큰 수, 보존율, 압축 상태. 추출형은 선택한 원문 범위(`source_ranges`)도 기록한다.
- `predictions.jsonl` / `predictions.csv`: 문서·조건·가설별 정답, 예측, 상태, 판단 시간. JSONL에는 생성 토큰과 종료 상태도 있다.
- 각 폴더의 `diagnostics/technical_summary.json`: 실패·종료·상한 도달 등 실행 상태.

## 출처

계약서 원문과 근거 주석의 출처는 Hitachi America, Ltd.의 **ContractNLI**다. 공개 스냅샷은 Koreeda와 Manning의 공식 Test에서 중복·라벨 불일치 문서 네 개를 제외하고 실험용 원문 단위를 추가한 버전이며, 원문과 근거의 문자 위치를 유지했다. 문맥 파일에는 이 원문과 실험 모델이 생성하거나 선택한 압축문이 들어 있다. [ContractNLI 논문](https://aclanthology.org/2021.findings-emnlp.164/) · [공식 데이터 출처](https://stanfordnlp.github.io/contract-nli/) · [데이터 라이선스 CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
