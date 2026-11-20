# 공개 실험 결과

[프로젝트 개요](../README.md) · [연구 질문별 결과 보고서](../docs/RESEARCH_QUESTION_RESULTS.md) · [재현 명령](../docs/REPRODUCIBILITY.md)

완료한 Main과 Analysis 1의 결과다. 세 조건은 ContractNLI Test **119문서·2,023문항**을 공유한다. Q0, Q8, Q17은 압축기에 보여준 가설 수를 뜻한다.

| 실행 | 보고서 | 결과 자료 |
|---|---|---|
| Main · Q0 | [원문과 질문 없는 압축](main/report.md) | [성능](main/metrics.csv) · [Original 대비](main/comparisons.csv) · [문항별 예측](main/predictions.csv) |
| Analysis 1 · Q17 | [17개를 모두 보여준 결과](questions17/report.md) | [Q17−Q0 비교](questions17/question_effects.csv) · [성능](questions17/metrics.csv) · [문항별 예측](questions17/predictions.csv) |
| Analysis 1 · Q8 | [8개를 보여준 결과](questions8/report.md) | 같은 문항에서 Q0·Q8·Q17을 비교한 전체 및 제공 여부별 집계표 |

문항별 비교의 차이는 Q17−Q0 또는 Q8−Q0·Q8−Q17 순서로 계산한다. `comparisons.csv`의 기준은 Original−압축 조건이다. CSV의 0.30은 30%를 뜻한다. 실패 문항은 정확도와 Macro-F1의 분모에 포함한다.

## Q0과 Q17 비교 그림

![Q0와 Q17의 정확도, 토큰 보존율, 차이의 신뢰구간](analysis/comparison.png)

회색은 Q0, 파랑은 Q17이다. 정확도 그래프의 점선은 Original **63.72%**다. 성능은 각 방법이 자연스럽게 만든 문맥 길이에서 측정했다.

- [질문 제공 효과 상세 분석](analysis/QUESTION_EFFECTS.md)
- [그림 PNG](analysis/comparison.png) · [그림 PDF](analysis/comparison.pdf)
- [문서별 변화](analysis/document_effects.json) · [가설별 변화](analysis/hypothesis_effects.json)
- [출력 형식 점검](analysis/format_audit_summary.json)

## Main과 Q17 세부 자료

| 항목 | Main | Q17 |
|---|---|---|
| 문서별 길이·보존율 | [길이](main/lengths.csv) | [길이](questions17/lengths.csv) |
| 혼동행렬 | [혼동행렬](main/confusion_matrices.csv) | [혼동행렬](questions17/confusion_matrices.csv) |
| 시간 | [실행 시간](main/times.csv) | [질문 제공 실행 시간](questions17/question_times.csv) |
| 집계 설정·정의 | [Main 분석 설정](main/analysis.json) | [Q17 분석 설정](questions17/question_analysis.json) |
| 라벨별 성능 | [클래스 Recall](main/metrics.csv) | [라벨별 변화](questions17/question_label_effects.csv) |

Main과 Q17의 공개 예측은 각각 10,115건이다. Q17에는 Main에서 재사용한 Original 2,023건이 포함된다. Q8의 새 판단 기록은 네 압축 방법의 8,092건으로, 공개 보고서에는 집계표를 담았다.

## 출처와 후속 분석

공개 파일의 원본 위치와 SHA256, 편집 내역은 [결과 manifest](manifest.json)에서 확인할 수 있다. 세부 원시 실행 폴더는 재현할 때 생성된다. [로컬 결과 위치](../docs/REPRODUCIBILITY.md#outputs)

입력 내 근거 위치를 살피는 [Analysis 2](../design/EXPERIMENT_DESIGN.md#analysis-2)와 압축 후 근거를 대조하는 [Analysis 3](../design/EXPERIMENT_DESIGN.md#analysis-3)은 예정 단계다.
