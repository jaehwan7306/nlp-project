# 공개 실험 결과

[프로젝트 개요](../README.md) · [연구 질문별 결과 보고서](../docs/RESEARCH_QUESTION_RESULTS.md) · [재현 명령](../docs/REPRODUCIBILITY.md)

메인 실험 **Q0 대 Q17**과 후속 **Analysis 1의 Q8** 결과다. 모든 압축 조건은 ContractNLI Test **119문서·2,023문항**을 공유한다. Q0, Q17, Q8은 압축기에 미리 보여준 가설 수를 뜻한다.

<a id="main"></a>

## 메인 실험 · 질문 유무 비교

원문을 그대로 판단한 Original을 기준선으로 두고, 같은 네 압축 방법에서 Q17과 Q0의 차이를 계산했다.

| 실행 | 보고서 | 결과 자료 |
|---|---|---|
| Original·Q0 | [원문과 질문 없는 압축](main/report.md) | [성능](main/metrics.csv) · [Original 대비](main/comparisons.csv) · [문항별 예측](main/predictions.csv) |
| Q17 | [17개를 모두 보여준 결과](questions17/report.md) | [메인 비교 Q17−Q0](questions17/question_effects.csv) · [성능](questions17/metrics.csv) · [문항별 예측](questions17/predictions.csv) |

`question_effects.csv`는 **Q17−Q0**, `comparisons.csv`는 **Original−압축 조건**을 보고한다. CSV의 0.30은 30%를 뜻한다. 실패 문항은 정확도와 Macro-F1의 분모에 포함한다.

### Q0과 Q17 비교 그림

![Q0와 Q17의 정확도, 토큰 보존율, 차이의 신뢰구간](analysis/comparison.png)

회색은 Q0, 파랑은 Q17이다. 정확도 그래프의 점선은 Original **63.72%**다. 성능은 각 방법이 자연스럽게 만든 문맥 길이에서 측정했다.

- [질문 제공 효과 상세 분석](analysis/QUESTION_EFFECTS.md)
- [그림 PNG](analysis/comparison.png) · [그림 PDF](analysis/comparison.pdf)
- [문서별 변화](analysis/document_effects.json) · [가설별 변화](analysis/hypothesis_effects.json)
- [출력 형식 점검](analysis/format_audit_summary.json)

### 길이·판단·실행 기록

| 항목 | Original·Q0 | Q17 |
|---|---|---|
| 문서별 길이·보존율 | [길이](main/lengths.csv) | [길이](questions17/lengths.csv) |
| 혼동행렬 | [혼동행렬](main/confusion_matrices.csv) | [혼동행렬](questions17/confusion_matrices.csv) |
| 시간 | [실행 시간](main/times.csv) | [질문 제공 실행 시간](questions17/question_times.csv) |
| 집계 설정·정의 | [Q0 분석 설정](main/analysis.json) | [Q17 분석 설정](questions17/question_analysis.json) |
| 라벨별 성능 | [클래스 Recall](main/metrics.csv) | [라벨별 변화](questions17/question_label_effects.csv) |

Q0와 Q17의 공개 예측은 각각 10,115건이다. Q17의 예측표에는 앞서 실행한 Original의 2,023건이 함께 들어 있다.

<a id="analysis-1"></a>

## Analysis 1 · 질문 8개 제공

Q8을 메인 실험의 Q0·Q17과 같은 문항에서 비교한다. 전체와 미리 보여준 질문·빠진 질문의 정확도와 신뢰구간을 공개 보고서에서 확인할 수 있다. 새 판단 기록은 네 압축 방법에서 8,092건이다. [Q8 전체 및 제공 여부별 집계](questions8/report.md)

## 출처와 후속 분석

공개 파일의 원본 위치와 SHA256, 편집 내역은 결과 manifest에 있다. 모델이 읽은 계약서 문맥인 `contexts.jsonl`과 원문 근거 주석을 포함한 `dataset.json`은 로컬 실행 폴더에 저장된다. [결과 manifest](manifest.json) · [팀 분석에 필요한 파일](../docs/REPRODUCIBILITY.md#team-data)

입력 내 근거 위치를 살피는 [Analysis 2](../design/EXPERIMENT_DESIGN.md#analysis-2)와 압축 후 근거를 대조하는 [Analysis 3](../design/EXPERIMENT_DESIGN.md#analysis-3)은 예정 단계다.
