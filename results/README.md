# 공개 실험 결과

[프로젝트 개요](../README.md) · [결과 요약과 해석](../docs/RESULTS.md) · [재현 명령](../docs/REPRODUCIBILITY.md)

2026-11-16 기준 완료한 **Main과 추가 분석 1**의 결과 스냅샷이다. Test 119문서·2,023문항을 사용했다. Main은 질문 없이 압축하고, 추가 분석 1은 고정 17개 가설 목록을 한꺼번에 제공했다. 모델을 새로 실행하지 않고 완료 파일을 정리했다.

| 실행 | 보고서 | 성능·비교 | 문항별 예측 |
|---|---|---|---|
| Main: 질문 없음 | [report.md](main/report.md) | [metrics.csv](main/metrics.csv) · [Original 대비](main/comparisons.csv) | [predictions.csv](main/predictions.csv) |
| Q17: 질문 목록 제공 | [report.md](questions17/report.md) | [질문 효과](questions17/question_effects.csv) · [metrics.csv](questions17/metrics.csv) | [predictions.csv](questions17/predictions.csv) |

**질문 효과는 `question_effects.csv`의 Q17−Q0다.** 각 실행의 `comparisons.csv`는 Original−압축 조건이므로 부호와 기준을 구분한다. CSV의 `0.30`은 30%다. 실패 문항도 성능 분모에 유지했다.

## 비교 그림과 상세 분석

![질문 제공 전후 성능·길이·신뢰구간 비교](analysis/comparison.png)

회색은 질문 없는 Main, 파랑은 17개 가설 목록 제공 조건이다. 정확도 그래프의 점선은 Original 63.72%다. 자연 출력 길이와 출력 형식이 함께 변했으므로 같은 길이의 요약 능력만 비교한 결과로 해석하지 않는다.

- [상세 분석 보고서](analysis/QUESTION_EFFECTS.md)
- [그림 PNG](analysis/comparison.png) · [그림 PDF](analysis/comparison.pdf)
- [문서별 변화](analysis/document_effects.json) · [가설별 변화](analysis/hypothesis_effects.json)
- [출력 형식 점검 요약](analysis/format_audit_summary.json)

## 길이·혼동행렬·시간

| 항목 | Main | Q17 |
|---|---|---|
| 문서별 길이·보존율 | [lengths.csv](main/lengths.csv) | [lengths.csv](questions17/lengths.csv) |
| 혼동행렬 | [confusion_matrices.csv](main/confusion_matrices.csv) | [confusion_matrices.csv](questions17/confusion_matrices.csv) |
| 시간 | [times.csv](main/times.csv) | [question_times.csv](questions17/question_times.csv) |
| 집계 설정·정의 | [analysis.json](main/analysis.json) | [question_analysis.json](questions17/question_analysis.json) |
| 라벨 변화 | Main 성능표의 클래스 Recall | [question_label_effects.csv](questions17/question_label_effects.csv) |

Main은 10,115개 판단 기록, Q17도 재사용 Original 2,023개를 포함해 10,115개다. Q17에서 새 모델 판단은 8,092회다. Original의 관측 시간을 Q17의 추가 실행 비용으로 세지 않는다.

## 파일 출처와 검증

[manifest.json](manifest.json)에 각 파일의 원본 위치·원본 SHA256·공개본 SHA256·변환 내역을 기록했다. CSV는 원본 그대로 복사했고 수치를 고치지 않았다. 일부 JSON의 컴퓨터 절대경로와 보고서 링크만 공개 구조에 맞게 정리했다. 전체 원시 run과 계약서·모델 캐시는 별도 보관한다.

실험 규칙·제외 결정·출력 문제와 한계는 [결과 요약](../docs/RESULTS.md), 고정 모델과 실행 명령은 [재현 안내](../docs/REPRODUCIBILITY.md)에 있다.
