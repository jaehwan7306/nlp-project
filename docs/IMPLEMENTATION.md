# 메인 구현 규칙과 검증 범위

현재 설계의 메인 5조건만 구현했다. 추가 분석의 목표 보존율·허용오차·질문 제공 실험·Gold Evidence 보존 지표는 구현하지 않았다. 코드와 로그에는 과거 결과나 제외 ID를 채워 넣지 않았다.

## 데이터 경계

공식 데이터 스키마의 `labels[hid].hypothesis`, `annotation_sets[0].annotations`, `spans`를 사용한다. 원본 문자 인덱스는 Python 문자열 기준의 start-inclusive/end-exclusive다. 공식 분할 크기 423/61/123과 17개 가설을 확인한다. 준비된 데이터에는 모델 실행용 Dev/Test만 넣고 Train은 감사에서만 읽는다.

사람이 review.csv에서 `duplicate`·`exclude_both`·`distinct`를 선택한다. 최종 제외 처리는 이 명시적 선택만 적용하며 Gold 라벨 차이로 변경하지 않는다. 라벨 불일치 계산은 비교 자료의 표시용이다. 이전 자동 제외 정책의 prepared는 재사용하지 않고 새로 준비하며, 원본 데이터와 감사 자료는 유지한다.

`compression_messages(condition, units)`는 가설이나 라벨을 매개변수로 받지 않는다. 모든 후보 단위를 같은 형식으로 제공한다. Gold Evidence는 중복 검토와 원본 스냅샷에 남아 있지만 압축 프롬프트에 삽입하지 않는다. 선택 결과의 provenance는 ID와 문자 위치로 남긴다.

## 프롬프트 경계

공통 system은 원문을 데이터로만 취급하고 사실적 의미를 유지하도록 지시한다. user는 생성 또는 ID 선택 작업과 원문을 제공한다. 두 우선 조건에 동일한 보존 문장 하나를 더한다. 문단별·순서별 요약을 요구하지 않고 목표 길이·개수도 요구하지 않는다. 최종 추출 문맥의 원문 순서 정렬은 모델 지시가 아니라 코드의 공통 조합 규칙이다.

판단 프롬프트는 문맥과 가설을 삽입하고 세 라벨만 요구한다. thinking off는 Qwen3 chat template 인자로 적용한다. 출력의 설명을 자동 재작성하거나 오류를 재질문하지 않는다.

전문은 `contract_experiment/prompts.py`, 명령 출력은 `show-prompts`, run 스냅샷은 `prompts.json`이다. 실제 렌더링 입력은 단위/문맥·가설과 동결된 chat template로 재구성할 수 있으며 매 호출 렌더링 프롬프트의 SHA256을 기록한다.

## 실제 길이와 실패

생성 모델의 safety cap과 native generated token count는 Qwen2.5 기준이다. 보고용 context retention은 Qwen3 토크나이저로 별도로 측정한다. 둘이 같다고 가정하지 않는다. 추출 ID 목록의 생성 토큰 수도 압축된 문맥 토큰 수가 아니다. 선택한 원문을 조합한 다음 Qwen3로 측정한다.

압축 실패의 `text`, `context_tokens`, `retention`은 null이다. 정상 빈 출력은 `text=""`, 토큰 수 0이다. 두 상황을 합치지 않는다. 모든 문항은 성능 분모에 포함한다. Original을 맞힌 문항이 하나도 없으면 정답 유지율은 정의되지 않아 null이다.

런타임 오류는 명령을 중단하고 완료 행을 유지한다. 다른 조건에만 정밀도·상한·offload를 적용하는 자동 대체는 없다. 소스나 설정을 고치면 동일 run의 재개가 차단되므로 새 Dev에서 재점검해야 한다.

## 최소 검증

검증은 Python 3.11 표준 라이브러리와 가상의 세 문서·17가설로 진행한다. 실제 ContractNLI 다운로드, 토크나이저/가중치 다운로드, GPU 모델 실행, Dev/Test 실험은 수행하지 않는다.

- 같은 방식에서 우선 문장만 다른지, 압축 입력에 질문·길이 지시가 없는지.
- 원문 span·비공백 gap·전체 단위 유지, 원문 순서, 중복/빈/없는 ID와 형식 실패.
- 라벨 파싱과 실패의 Accuracy·Macro-F1 반영.
- 사람의 명시적 한 문서 유지·양쪽 제외·별개 결정과 Test 우선 유지, 서로 충돌하는 검토의 중단.
- 행 기록의 중단 복구와 중복 키 차단.
- 문항 대신 문서를 묶는 paired bootstrap.
- 합성 데이터로 audit → prepare → Dev → freeze → Test → 분석과 재개. 모델 호출은 stub으로 대체한다.

실제 GPU의 메모리·모델 출력 품질·실제 보존율·처리시간은 로컬에서 확인해야 한다. 요구사항 버전과 공식 API는 문서를 대조했지만 전체 CUDA 의존성 설치를 에이전트 환경에서 수행하지 않았다.

검증 기록(2026-11-12): Python 3.11.9에서 합성 데이터 CPU 테스트 9개 통과, compileall 구문 검사 통과, CLI 도움말·명령 인자·프롬프트 JSON·설정·UTF-8 검사 통과. CLI 점검 시 torch/transformers 모듈이 로드되지 않았음을 확인했다.

추가 검증(2026-11-12): 중복 검토용 HTML 보고서의 원문 diff·라벨 충돌·Gold Evidence 표시, HTML 이스케이프, CSV/audit/후보 문서 비변경을 확인했다. 전체 CPU 테스트 10개와 `review-report --help` 점검이 통과했다. 기존 감사 자료만으로 보고서를 생성하며 모델이나 외부 라이브러리를 요구하지 않는다.

수동 제외 변경 검증(2026-11-12): 라벨이 달라도 duplicate는 하나 유지하고, 라벨이 같아도 exclude_both는 둘 다 제외함을 확인했다. 겹친 행의 유지·제외 충돌은 중단하며, 이전 자동 정책의 prepared 재사용도 차단한다. HTML의 선택 처리 표시와 전체 합성 워크플로를 포함한 CPU 테스트 11개가 통과했다. 실제 데이터·모델 실행은 수행하지 않았다.

## 디코딩 수정과 압축 재사용 검증 · 2026-11-14

설정 객체에 적힌 값과 `generate`에 실제 적용되는 값은 다를 수 있다. 수정판은 `use_model_defaults=False`와 매 호출의 greedy 인자를 함께 적용하고 `_prepare_generation_config` 결과를 검사·기록한다. 해당 내부 API는 고정된 Transformers 4.57.1에서 CPU 회귀 검사로 확인한다.

`tests/test_generation.py`는 다운로드나 모델 생성 없이 실제 라이브러리 설정 병합을 실행한다. Qwen3형 모델 기본값에서 기존 코드의 샘플링 전환을 재현하고, 수정된 실제 생성 호출 인자가 이를 막는지 확인한다. Qwen2.5형 이전 버전에서는 수정 전후 설정이 같음을 확인한다. 의존성 설치 전 표준 라이브러리 실행에서는 이 검사만 skip되며 설치 후 다시 실행한다.

`recovery.py`는 알려진 이전 소스 아카이브·스냅샷·환경과 동일 전체 Dev를 확인하고, 캐시된 모델 설정과 토크나이저만으로 압축 동등성을 검증한다. 압축 결과는 원본 비용·실패를 포함해 그대로 복사하고, 이전 판단을 복사하지 않는다. 대상 journal은 원자적으로 교체하고 출처를 별도 기록한다. 소스 무결성 검사를 생략하거나 이전 protocol을 수정하는 우회는 제공하지 않는다.

기존 로컬 Dev를 읽는 CPU 검증에서 61문서·244개 압축의 원본 출력, 프롬프트 해시, 추출 조합, 토큰 수와 실제 디코딩 동등성을 확인했다. 원본 파일은 수정하지 않았다. GPU 모델 추론과 실험 재실행은 수행하지 않았다.

검증 결과: Python 3.11.9 / Transformers 4.57.1에서 CPU 테스트 16개 통과. 작업 폴더의 별도 검증용 run으로 기존 244개 압축 복사까지 확인했으며 판단 0건, 원본 파일 비변경을 확인했다. 실제 사용자 run에 쓰거나 GPU 모델을 로드하지 않았다.

## 전체 문맥 메모리 확장 · 2026-11-15

아래는 D013 당시 구현/검증 기록이다. 현재 실행 흐름은 다음 D014 항목의 「처음부터 동일 정책」을 따른다.

`memory_support/attention.py`는 토큰 축을 자르지 않고 head 축을 순차 실행한다. GQA에서는 query head 번호를 query/KV head 비율로 나눈 원래 KV head에 연결한다. mask에 head 축이 있으면 해당 head만 취하고, 나머지 길이·causal·scale을 유지한다. native math SDPA의 결과를 원래 head 순서로 합친다. 1,024 이하의 query와 decode는 기존 native 호출이다. 예외에서도 전역 함수 교체를 복원한다. 단일 모델·단일 추론 프로세스의 기존 실행 구조를 전제로 한다.

`memory_support/runner.py`는 기존 `load_run`, 환경·runtime 검사를 재사용한다. 원래 패키지/entry point의 무결성 검사를 우회하거나 동결 해시를 고치지 않는다. 별도 확장의 소스와 해시·검증 보고서·적용 경계를 추가 보관하며, 원래 runtime JSON은 기본 모델 설정, `runtime_memory_policy.json`은 실제 추가 실행 정책을 설명한다. 모델 단계는 보완 entry point로 통일하고 출력마다 정책을 기록한다. 일반 entry point로 생성해 정책 기록을 빠뜨린 이후 행은 보완 재개가 거부한다.

Dev 출력 검사는 longest Original 문서 1개에서 판단 5개와 압축 4개를 대조한다. exact token sequence 등을 비교하며 Gold나 정확도는 사용하지 않는다. 이 9개만 실제 모델로 실행하는 절차는 사용자 로컬 명령이다. 에이전트는 작은 CPU 텐서와 stub 모델의 합성 흐름만 검증한다. 기존 Dev/Test 원본 기록과 동결 파일에는 쓰지 않는다.

검증 결과: Python 3.11.9에서 CPU 테스트 21개 통과. 기존 Dev 5,185건과 Test 544건의 기본 source identity가 그대로 일치함을 확인했다. Dev 판단 SHA256은 `39ed13e2e35569680b5f04f62ef3496e121db16f78f7ddb7825022a9c52fb4cb`, Test 판단은 `9ccc37c5c77ef6aa06e31cb88c4b6c4b0af381d78c3a473631917a664f96b0e0`로 보존했다. 추가 실행 파일 구문·CLI·torch 비로딩 import 검사도 통과했다. GPU 모델의 실제 메모리와 출력 비교는 아직 실행하지 않았다.

## 처음부터 동일 정책인 Test · D014

`init-test`는 기본 모델의 full Dev protocol을 읽고 기존 식별자/환경을 확인한다. 새 `protocol.json`의 `memory_execution.mode=from_start`에 보완 정책과 기술 점검을 결합하고, 두 모델 runtime에 정책 marker를 추가한다. 새 protocol 디렉터리의 새 Test 등록은 기존 initializer가 수행하며, 원본 protocol의 Test 등록을 지우지 않는다.

일반 Main 소스와 config는 그대로여서 기본 Dev source identity가 유지된다. 추가 소스의 사본·해시는 별도로 동결하며, 메모리 모델의 실제 runtime에도 같은 해시를 기록한다. 따라서 canonical `model_details` 비교가 native 모델 실행을 생성 전에 거부한다. 기존 해시 검사를 완화하거나 원래 manifest를 고치지 않는다.

초기 run은 원문 문맥만 N개, 예측 0개여야 한다. `runtime_memory_policy.json`이 최초 생성부터 이 정책을 고정하고, 이후의 모든 모델 출력은 같은 hash를 가져야 한다. 이미 모델 출력이 있는 이전 Test를 채택할 수 없다. 검사 보고서는 run/protocol 안에 보관하므로 초기화 뒤 외부 검사 경로에 계속 의존하지 않는다.

`check-dev`의 새 schema는 형식·동일 입력·greedy·메모리 정책의 기술 통과와 기존 출력 동일성을 구분한다. 서로 다른 정상 출력도 기술 점검을 통과할 수 있다. 모든 새 출력은 별도 `outputs.jsonl`에 기록한다. 오래된 schema의 PASS/FAIL을 편집해 재사용하지 않는다.

검증 결과: Python 3.11.9 CPU 테스트 22개 통과. 합성 데이터로 새 동결과 초기화, native 진입 차단, 5조건의 모든 새 판단, 재개 중복 방지, 진단/분석, 새 protocol 덮어쓰기 차단과 이전 결과 비변경을 확인했다. 실제 GPU 모델 실험은 수행하지 않았다.
