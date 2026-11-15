# ContractNLI 문맥 압축 Main Experiment

Python **3.11** 프로젝트다. ContractNLI 원문과 질문 없는 압축 4조건을 같은 판단 모델로 평가한다. 프로젝트를 새로 구성할 때 이전 실험의 코드·결과·중복 제외 목록을 가져오지 않았다. **추가 분석 1(17개 가설 목록 제공)**은 별도 실행 파일로 구현했다. 목표 보존율 제약과 Gold Evidence 오류 사례 분석은 이후 구현한다.

**현재 Test 실행은 `run_memory_safe.py`로 처음부터 통일한다.** 이미 `runs/dev_greedy`와 `protocol_greedy`를 완료한 경우 아래 **「OOM 보완으로 Test 전체를 새로 실행」** 절차부터 진행한다. 기존 Test 판단·압축 결과는 새 Test로 가져오지 않는다.

**`runs/test_memory_safe` 분석까지 완료했다면 「추가 분석 1: 17개 가설 목록 제공」부터 진행한다.** Main을 다시 실행하거나 데이터를 다시 검수할 필요는 없다. 추가 분석은 `run_question_analysis.py`를 사용한다.

| 조건 ID | 조건 | 압축 출력 |
|---|---|---|
| `original` | Original | 원문 |
| `generative_general` | 생성형·일반 / Naive | 새 요약문 |
| `generative_priority` | 생성형·보존 우선 / Contextualized | 새 요약문 |
| `extractive_general` | 추출형·일반 | 원문 단위 ID 선택 |
| `extractive_priority` | 추출형·보존 우선 | 원문 단위 ID 선택 |

압축·선택은 모두 **Qwen2.5-7B-Instruct**, 판단은 **Qwen3-8B, thinking off**다. 두 모델 모두 같은 4bit NF4 설정을 사용한다. 같은 출력 방식에서는 보존 우선순위 문장 하나만 다르다. 네 조건 모두 질문·정답·Gold Evidence와 목표 보존율을 압축 입력에 넣지 않는다.

모델은 자연 종료하며 최소 생성 길이를 강제하지 않는다. `4096` 생성 토큰은 네 압축 작업에 공통으로 적용하는 **실행 안전 상한의 초기값**이다. 목표 보존율이나 원문 길이에 따른 예산이 아니다. 상한 도달·가짜 대화·빈 출력·원문 전체 복사 등을 기록하고 Dev에서 적절성을 확인한다. 결과가 같거나 실제 길이가 같다고 가정하지 않는다.

**2026-11-14 디코딩 수정:** 이전 코드에서는 Qwen3의 모델 기본값이 요청한 `do_sample=False`를 덮어써 실제 판단이 샘플링으로 실행됐다. 수정판은 모델 기본값 상속을 끄고 매 생성 호출에서 greedy를 명시하며, 적용된 설정을 검증·기록한다. 이전 run은 아래 **기존 실행 복구** 절차를 따른다. 기존 Test에 `--resume`으로 새 판단을 섞지 않는다.

## 1. 실행 환경 준비

프로젝트를 내려받아 **README.md와 run_experiment.py가 있는 폴더를 IDE에서 열고**, 그 IDE의 터미널에서 실행한다. 특정 사용자 경로나 폴더 이름으로 이동하는 명령은 필요 없다. 아래 명령은 Windows PowerShell 기준이다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

마지막 출력이 `Python 3.11.x`인지 확인한다. `py`가 없으면 설치한 Python 3.11의 `python -m venv .venv`를 사용한다. PowerShell 활성화가 차단되면 해당 터미널에서 `Set-ExecutionPolicy -Scope Process Bypass` 후 다시 활성화하거나, 아래의 `python` 대신 `.\.venv\Scripts\python.exe`를 사용한다.

**모델 실행 없이 확인하는 최소 검증**은 외부 패키지 설치 전에도 가능하다.

```powershell
python -m unittest discover -s tests -v
python run_experiment.py --help
python run_experiment.py show-prompts
```

GPU 실행에는 NVIDIA GPU와 CUDA 지원 PyTorch가 필요하다. 아래는 CUDA 12.8 빌드를 사용하는 고정 설치 예시다. GPU 드라이버가 지원하는지는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/#v280)를 확인한다. CUDA Toolkit 전체를 별도 설치하는 절차는 기본 실행에 포함하지 않는다.

```powershell
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
python -m pip freeze > environment-lock.txt
```

`CUDA: True`를 확인한다. `requirements.txt`의 PyTorch 버전은 CUDA 빌드의 기본 버전과 같으므로 이미 설치한 GPU 빌드를 다시 CPU 빌드로 바꿀 필요가 없다. 모델은 한 번에 하나만 GPU에 올린다. **12GB에서 가장 긴 문서까지 실행되는지는 로컬 사전 점검이 필요하다.** 코드가 원문을 자동으로 자르거나 CPU 오프로딩으로 전환하지는 않는다. bf16 미지원 GPU라면 Dev 시작 전에 `config.toml`의 `compute_dtype`를 `float16`으로 바꾸어 두 모델에 공통 적용한다.

## 2. 공식 데이터 확보와 새 중복 검토

```powershell
python run_experiment.py fetch-data --dest data
python run_experiment.py audit-data --data data --out data_audit --threshold 0.95
```

`fetch-data`는 공식 압축파일에서 Train/Dev/Test JSON만 가져오고 해시를 기록한다. 기존 파일이 내려받기 기록과 같으면 검증 후 재사용하며 덮어쓰지 않는다. Train은 스키마·교차 중복 점검에만 쓰며 모델 추론에 사용하지 않는다.

`audit-data`는 문서·17개 라벨·근거 참조·span 범위를 검사하고 완전/정규화/유사 중복 후보를 만든다. 유사도는 **정규화한 연속 5단어 집합의 Jaccard, 초기 후보 임계값 0.95**다. 이 숫자는 문서 제외 기준이 아니다. 원문·정답은 수정하지 않는다.

다음 파일을 먼저 확인한다.

- **`data_audit/review_report.html`**: 브라우저에서 보는 원문 차이·가설별 라벨·Gold Evidence 비교 화면.
- `data_audit/review.csv`: 후보별 검토 입력표.
- `data_audit/candidate_documents.json`: 후보 문서 원문·가설·라벨·근거. 압축 입력과는 별개의 사람 검토 자료다.
- `data_audit/audit.json`: 유사도·라벨 충돌·근거 텍스트 차이 등 후보 정보.
- `data_audit/unit_checks.json`: 공식 span 사이의 내용 누락과 짧은 구간 점검 자료.

새 `audit-data`는 HTML 보고서도 함께 만든다. 이미 감사 파일을 만들었다면 기존 자료에서 보고서만 생성할 수 있다. 아래 명령은 원문·CSV·제외 결정을 변경하지 않으며 모델을 실행하지 않는다.

```powershell
python run_experiment.py review-report --audit data_audit/audit.json
Invoke-Item data_audit/review_report.html
```

보고서의 후보 목록에서 문서 쌍을 누르면 다른 원문 줄과 주변 3줄이 양쪽에 표시된다. 빨강은 삭제, 초록은 추가, 노랑은 변경이다. 전체 원문은 펼쳐서 볼 수 있다. 라벨이 다른 가설은 양쪽 라벨과 실제 Gold Evidence 본문·문자 위치를 함께 보여준다. 전체 17가설의 라벨·근거도 펼칠 수 있다.

`similarity=0.98`은 문자나 계약 내용의 98%가 같다는 뜻이 아니다. **정규화한 연속 5단어 묶음 집합의 Jaccard 점수**다. `raw_exact`는 원문 완전 일치, `normalized_exact`는 정규화 후 일치, `similar`는 실제 차이를 검토해야 하는 후보를 뜻한다. 보고서의 차이와 원문을 읽고 CSV에 판단 근거를 남긴다. 보고서는 읽기 전용이므로 CSV의 메모를 바꾼 뒤 화면에도 반영하려면 보고서 명령을 다시 실행한다.

**여기서 바로 다음 명령으로 넘어가지 않는다.** `review_required=True`인 모든 행의 `decision`과 `notes`를 사람이 채운 뒤 저장한다. CSV의 행이나 `pair_id`는 삭제·변경하지 않는다.

| 사람이 선택한 처리 | `decision` | 적용 결과 |
|---|---|---|
| 중복 중 한 문서만 남김 | `duplicate` | Test 우선, 같은 split이면 작은 ID 하나 유지 |
| 양쪽 문서 모두 제외 | `exclude_both` | 라벨 충돌 등 사람이 기록한 사유에 따라 두 문서 제외 |
| 이 쌍을 이유로 제외하지 않음 | `distinct` | 이 쌍으로 추가 제외하지 않음 |

Gold 라벨 불일치는 비교 자료로 표시할 뿐, 코드가 최종 처리 방식을 결정하지 않는다. 원문·라벨·근거를 사람이 검토하고 `decision`과 `notes`에 최종 처리를 기록한다. Gold 라벨은 변경하지 않는다.

사용자가 정한 예시에서는 `test:52|test:97`에 `duplicate`를 입력하면 52를 유지하고 97을 제외한다. `test:537|test:539`는 **`exclude_both`**를 입력해야 537·539가 모두 제외된다. 이 ID들은 코드에 제외 목록으로 하드코딩하지 않는다.

**기존 CSV를 사용하는 경우:** 이전 코드에서는 `duplicate`를 입력하면 라벨 충돌에 따라 코드가 양쪽 제외를 선택했다. 현재는 `duplicate`가 항상 한 문서 유지라는 뜻이다. 양쪽을 제외하려는 기존 행은 사람이 직접 `exclude_both`로 바꿔야 한다. 자동 변환하거나 CSV를 덮어쓰지 않는다. 같은 문서가 여러 후보에 걸쳐 있고 유지·제외 결정이 충돌하면 자동으로 해결하지 않고 해당 행을 다시 검토하도록 중단한다.

Train과의 중복은 보고만 하므로 `review_required=False` 행은 비워도 된다. 정확히 같은 원문 후보도 이번에는 사람이 확인한다. 검토를 마쳤으면 실행한다.

```powershell
python run_experiment.py prepare --audit data_audit/audit.json --review data_audit/review.csv --out prepared
```

미검토 후보가 있으면 `prepare`가 중단되는 것이 정상이다. `duplicate`로 연결된 그룹은 Test를 우선하고 같은 split에서는 작은 ID 하나를 유지한다. `exclude_both`는 그 행의 두 문서를 명시적으로 제외한다. 라벨 차이만으로 다른 문서를 자동 제외하지 않는다. Dev/Test 교차 중복에서 `duplicate`를 선택하면 Test를 유지하고 Dev에서 제외한다.

`prepared/exclusions.csv`에는 제외 문서·유지 문서·최종 decision·검토한 pair ID·사람의 notes를 함께 기록한다. **이미 이전 방식으로 prepared를 만들었다면** 검토 CSV를 수정한 뒤 새 폴더로 준비한다. 이후 명령의 `--prepared`에도 같은 새 폴더를 지정한다.

```powershell
python run_experiment.py prepare --audit data_audit/audit.json --review data_audit/review.csv --out prepared_manual
```

아직 prepared를 만들지 않았다면 위의 기본 `--out prepared` 명령을 쓰면 된다. `data`와 `data_audit`는 그대로 사용할 수 있다.

`prepared/exclusions.csv`의 제외 사유와 `prepared/manifest.json`의 최종 문서 수를 확인한다. 과거의 119·122 등 문서 수를 강제하지 않는다. Test가 N문서이면 메인은 **5 × 17N**건이다. 원본 데이터를 바꾸면 새 audit를 만들고, 검토 결정을 바꾸면 새 prepared 폴더를 사용한다.

공식 span 전체를 후보로 사용하고, 사이에 원문의 비공백 내용이 있으면 그대로 별도 단위로 추가한다. 공백만 있는 틈은 생략한다. 모든 압축 조건이 같은 번호 붙은 원문을 받는다. 추출형 선택 결과는 원문 순서로 연결한다. **예외가 별도 단위로 나뉘어 있으면 예외를 선택하지 않을 수도 있다.** 원문 단위를 그대로 남긴다고 모든 예외가 자동 보존되는 것은 아니다.

## 3. 모델 버전 동결과 다운로드

```powershell
python run_experiment.py lock-models --config config.toml --out model_lock.json
python run_experiment.py download-models --config config.toml --models model_lock.json
```

첫 명령은 각 모델의 현재 커밋 SHA를 `model_lock.json`에 고정하고 가중치는 내려받지 않는다. 두 번째 명령이 그 SHA의 가중치와 토크나이저를 Hugging Face 공통 캐시에 내려받는다. 이미 같은 SHA가 캐시에 있으면 중복 다운로드를 줄인다. 기본 설정에서는 추론 중 네트워크 다운로드를 허용하지 않는다.

팀원과 정확한 리비전을 재현하려면 **model_lock.json을 공유**한다. 설정·데이터·환경·프롬프트·코드는 각 run에 별도로 스냅샷된다.

## 4. 입력 길이 점검: 토크나이저만 사용

```powershell
python run_experiment.py inspect-inputs --config config.toml --prepared prepared --models model_lock.json --out runs/input_stats
```

`runs/input_stats/summary.json`의 `over_context_limit`이 빈 목록인지 확인한다. `input_stats.csv`에는 Qwen2.5/Qwen3 원문 토큰 수, 번호가 붙은 실제 압축 프롬프트 길이, Original 판단 프롬프트 길이가 있다. 모델 가중치를 올리거나 판단을 수행하지 않는다. 최대 문맥 길이를 넘으면 자동 절삭하지 않고 원인과 실행 설정을 먼저 점검한다.

## 5. 소규모 Dev 사전 실행

`--documents 3`은 준비된 Dev에서 **원문 문자 수가 가장 긴 3문서**를 결정적으로 고른다. 동률이면 작은 ID를 우선한다. 전체 Dev 결과와 구분해 별도 폴더를 사용한다.

```powershell
python run_experiment.py init-run --split dev --documents 3 --out runs/preflight
python run_experiment.py judge --run runs/preflight --conditions original
python run_experiment.py compress --run runs/preflight
python run_experiment.py judge --run runs/preflight --resume
python run_experiment.py diagnose --run runs/preflight
```

3문서라면 255개 판단이다. 첫 판단은 Original만 수행한다. 두 번째 판단 명령의 `--resume`은 기존 Original을 건너뛰고 압축 4조건을 평가한다.

`runs/preflight/diagnostics/technical_summary.json`과 `lengths.csv`에서 다음을 확인한다. 이 진단은 정확도를 계산하지 않는다.

- 다섯 조건의 문서/판단 건수가 맞는지.
- `eos`·`stop_string`·`max_tokens` 종료와 안전 상한 도달 현황.
- 가짜 대화, 빈 출력, 생성형의 원문 전체 복사, 추출 ID 오류·중복 현황.
- 판단 라벨 파싱 실패와 실제 토큰 보존율.
- 긴 문서의 GPU 메모리와 실행 시간이 감당되는지.

자세한 원본 출력은 `contexts.jsonl`, `predictions.jsonl`에 있다. 내용 감사를 메인 진행의 필수 게이트로 요구하지 않는다. 기술 문제로 설정을 바꾸면 의사결정을 기록하고 **새 Dev run**을 만든다. Dev 정확도를 높이는 방향으로 프롬프트·설정을 선택하지 않는다.

## 6. 전체 Dev 기술 점검과 규칙 동결

```powershell
python run_experiment.py init-run --split dev --out runs/dev
python run_experiment.py judge --run runs/dev --conditions original
python run_experiment.py compress --run runs/dev
python run_experiment.py judge --run runs/dev --resume
python run_experiment.py diagnose --run runs/dev
```

진단 파일을 확인한 뒤 규칙을 동결한다. 소규모 preflight 결과로는 동결할 수 없다.

```powershell
python run_experiment.py freeze --run runs/dev --out protocol
```

동결은 전체 Dev의 다섯 조건과 판단 건수, 최신 진단, 실제 실행 환경을 확인한다. 정확도나 사람 충실도 점수를 통과 조건으로 사용하지 않는다. 토큰 상한·형식 문제를 고칠 필요가 있다면 동결 전에 새 Dev에서 재점검한다. 추가 분석 2의 목표 보존율이나 허용오차를 지금 정할 필요는 없다.

## 7. Test: Original을 먼저 평가하고 메인 완료

아래는 6단계에서 `runs/dev`와 `protocol`을 만든 신규 실행의 예다. 전체 Dev에서 메모리 보완 기술 점검을 하고, 보완 정책을 포함한 **새 protocol과 새 Test**를 만든다. 기술 점검은 최장 Dev 1문서의 9개 출력만 실행한다.

```powershell
python run_memory_safe.py check-dev --run runs/dev --out runs/memory_check_fresh
python run_memory_safe.py init-test --base-protocol protocol --protocol-out protocol_memory_safe --out runs/test_memory_safe
python run_memory_safe.py judge --run runs/test_memory_safe --conditions original
python run_memory_safe.py analyze --run runs/test_memory_safe --conditions original
```

첫 명령의 `Dev technical check: PASS`를 확인한 뒤 나머지를 진행한다. 이전 출력과의 일치 개수는 참고 정보이며 통과 조건이 아니다. Original 기준선은 `runs/test_memory_safe/results/baseline/`에서 확인한다. Test 결과를 보고 설정을 바꾸지 않는다.

```powershell
python run_memory_safe.py compress --run runs/test_memory_safe
python run_memory_safe.py judge --run runs/test_memory_safe --resume
python run_memory_safe.py diagnose --run runs/test_memory_safe
python run_memory_safe.py analyze --run runs/test_memory_safe
```

하나의 새 protocol은 Test run 하나만 시작할 수 있다. 중단되면 같은 메모리 보완 코드와 폴더에서 재개한다. `init-test`는 원래 Dev protocol을 보존하고 새로운 실행 정책을 명시적으로 동결한다. 이미 사용한 새 protocol을 덮어써서 Test를 반복 초기화할 수는 없다.

## 8. 결과를 보는 곳

전체 메인 결과는 **`runs/test_memory_safe/results/`**에 생성된다. CSV 비율은 `0.30 = 30%`다.

| 파일 | 내용 |
|---|---|
| `report.md` | 다섯 조건의 성능·중앙 토큰 보존율 요약 |
| `metrics.csv` | Accuracy, Macro-F1, 클래스 Recall, 정답 유지율, 실패율, 보존율 평균·중앙값·분위수 |
| `comparisons.csv` | Original − 각 압축 방법의 정확도·Macro-F1 차이와 95% CI |
| `confusion_matrices.csv` | 세 정답 클래스와 `Invalid`를 포함한 예측 혼동행렬 |
| `lengths.csv` | 문서·방법별 최종 토큰 수, 실제 보존율, 종료 이유, 선택 단위 수 |
| `times.csv` | 압축·판단·합산 시간. 모델 로딩 시간은 별도 로그 |
| `predictions.csv` | 문서·가설·방법별 Gold/예측/상태 |
| `analysis.json` | 집계 결과, 분모와 시간 처리 원칙, 부트스트랩 설정 |

원시 기록과 재현 정보는 run 바로 아래에 있다.

- `contexts.jsonl`: 최종 문맥, 원본 출력·토큰 ID·종료 이유·선택 ID·문자 위치·압축 시간.
- `predictions.jsonl`: 판단 출력·파싱 결과·프롬프트 해시·판단 시간.
- `dataset.json`, `prepared_manifest.json`: 해당 실행의 문서·Gold와 새 전처리 해시.
- `prompts.json`, `config.json`, `models.json`: 실제 지시·설정·모델 리비전.
- `runtime_compressor.json`, `runtime_judge.json`: 실제 GPU·전체 패키지 버전·채팅 템플릿·생성/양자화 설정. 수정판은 적용된 `generation_config`, `generation_overrides`, `decoding_policy=explicit_greedy_v1`을 기록한다.
- `source/`, `manifest.json`: 실행 시점 소스 사본과 무결성 해시.
- `runtime_memory_policy.json`, `memory_extension_source/`: 처음부터 고정한 OOM 실행 정책·소스·기술 점검. 이전 Test 예측·압축 결과 재사용은 0건.
- `events.jsonl`: 모델 로딩 시간 등 실행 이벤트.

성능 분모에는 모든 문항을 유지한다. 추출 파싱 실패는 해당 17문항을 압축 실패 오답으로 기록하고, 판단 파싱 실패도 오답으로 처리한다. 길이를 알 수 없는 압축 실패는 보존율을 0으로 꾸미지 않고 결측으로 남긴다. `[]`는 정상적인 빈 선택이므로 빈 문맥을 판단하고 보존율 0으로 측정한다.

문서 단위 클러스터 부트스트랩을 사용한다. 문서를 복원추출하고 그 문서의 17문항을 함께 포함하며, 같은 표본으로 방법을 비교하고 Macro-F1도 매 반복 재계산한다. 기본은 10,000회, seed 42, percentile 95% CI다. 메인의 길이는 통제하지 않았으므로 같은 길이에서의 우열을 의미하지 않는다.

## 9. 중단·재개와 변경 기록

Original 판단 도중 중단했다면 같은 새 Test의 Original만 재개한다.

```powershell
python run_memory_safe.py judge --run runs/test_memory_safe --conditions original --resume
```

완료한 레코드는 즉시 저장한다. 예를 들어 Test 압축 또는 판단 중 중단했다면 필요한 단계만 재개한다.

```powershell
python run_memory_safe.py compress --run runs/test_memory_safe --resume
python run_memory_safe.py judge --run runs/test_memory_safe --resume
```

기록된 실패 출력도 완료된 관측으로 유지하며 다시 생성하지 않는다. GPU OOM 같은 실행 오류는 그 항목을 완료로 저장하지 않고 중단한다. 동일한 설정으로 환경 자원을 확보한 뒤 재개한다. 소스·프롬프트·설정이 달라지면 같은 run에 섞어서 재개할 수 없다. 도중에 끊긴 JSONL 마지막 행만 원본 백업 후 복구하고, 중간 행 손상이나 중복 키는 오류로 처리한다.

향후 변경 때 다음을 남긴다.

1. [CHANGELOG.md](CHANGELOG.md): 무엇을 바꿨고 어떤 범위에 영향을 주는지.
2. [docs/DECISIONS.md](docs/DECISIONS.md): 문제, 선택지, 선택 이유, 변인통제 영향과 검증.
3. 동결 전 변경이라면 새 Dev run. 동결한 소스·설정·Test 결과는 덮어쓰지 않는다.

같은 하드웨어·버전에서 재현할 수 있도록 seed와 결정적 연산 설정을 기록한다. greedy라도 다른 GPU·라이브러리 사이의 비트 단위 일치를 보장하지는 않는다. 결정적 구현이 없는 연산 경고도 실행 로그와 함께 보관한다.

## 기존 실행 복구 — 2026-11-14 greedy 설정 수정

이전 실행 중이라면 `Ctrl+C`로 중단한다. GitHub 최신 코드를 업데이트하되 **`.venv`, `data`, `data_audit`, `prepared`, `model_lock.json`, 기존 `runs`·`protocol`은 보존**한다. 같은 고정 의존성을 쓰면 재설치·모델 재다운로드·중복 검수는 필요 없다. 이전 `manifest.json`이나 `protocol.json`의 해시를 고치지 않는다.

수정 전 Qwen3 판단은 샘플링이므로 Dev/Test의 이전 판단을 새 greedy 결과로 이어 쓰지 않는다. Qwen2.5 압축은 모델 설정의 버전 차이로 기존에도 greedy였다. `reuse-compression`은 알려진 이전 소스, 데이터·모델·프롬프트·환경, 실제 디코딩 설정의 동등성, 모든 압축 문맥을 검사한 뒤 **압축 결과만** 새 전체 Dev에 복사한다. 출력 실패·상한 도달도 그대로 포함하며 판단은 한 건도 복사하지 않는다. CPU와 캐시된 토크나이저/설정만 사용하고 모델 가중치는 로드하지 않는다.

다음은 기존 전체 Dev가 `runs/dev`에 있는 경우다. 새 폴더는 이전 샘플링 결과와 분리하기 위한 이름이다.

```powershell
python -m unittest discover -s tests -v
python run_experiment.py init-run --split dev --out runs/dev_greedy
python run_experiment.py reuse-compression --source-run runs/dev --run runs/dev_greedy --check-only
python run_experiment.py reuse-compression --source-run runs/dev --run runs/dev_greedy
python run_experiment.py judge --run runs/dev_greedy
python run_experiment.py diagnose --run runs/dev_greedy
```

검사에서 불일치가 나오면 오류를 무시하거나 해시를 수정하지 않는다. 원인을 확인하고, 호환성이 없는 압축은 새 Dev에서 다시 생성해야 한다. 복구 명령은 새로 초기화한 Dev에만 적용한다. 복구 후 `compress`를 다시 실행할 필요는 없다. 판단 도중 중단하면 **새** 폴더에서 `judge --run runs/dev_greedy --resume`으로 이어간다.

새 greedy Dev 진단을 확인한 다음 Dev 기준 protocol을 만든다. 이미 만들어진 `protocol_greedy`가 있다면 이 명령은 다시 실행하지 않는다.

```powershell
python run_experiment.py freeze --run runs/dev_greedy --out protocol_greedy
```

이후 Test는 아래 「OOM 보완으로 Test 전체를 새로 실행」 절차를 따른다. 예전에 시작한 `runs/test_greedy`는 보관하고 새 결과에 합치지 않는다.

로딩 직후 `Decoding verified: judge / do_sample=False / num_beams=1 / model defaults disabled`를 확인한다. `torch_dtype is deprecated`는 별도의 API 명칭 경고이며 이 샘플링 문제의 원인은 아니다.

복구의 출처·해시·동등성 검사·판단 미복사 여부는 `compression_reuse.json`에 남는다. `compression_seconds`는 원래 실제로 걸린 압축 시간을 유지한다. 재사용했다고 압축 비용이 0이었던 것으로 보고하지 않는다. 이전 Dev의 보존율과 압축 특성 관측은 유효하지만 판단 출력·파싱·시간은 샘플링 실행의 관측으로 구분한다. 오류 수정 때문에 새 실행을 시작한 이유는 결정 D012에 기록한다.

## OOM 보완으로 Test 전체를 새로 실행

2026-11-15 Test 58번(원문 7,908토큰)에서 재시작 직후에도 SDPA attention 연산의 OOM이 재현됐다. 완료된 544건은 저장되어 있었다. 이 경우 단순 재개만으로 해결됐다고 가정하지 않는다.

추가 실행 파일 `run_memory_safe.py`는 긴 입력을 읽는 attention 계산을 head 하나씩 순차 실행한다. 각 head에는 전체 query/key 토큰을 그대로 제공하고, GQA의 대응 KV head와 마스크·인과 관계를 유지한다. query 길이가 1,024를 초과하면 native SDPA MATH로 나눠 계산하고, 짧은 입력·후속 1토큰 생성은 기존 native 호출을 사용한다. 원문·프롬프트·4bit 정밀도·greedy·출력 상한은 유지한다. 모델을 CPU로 옮기거나 긴 문서를 제외하지 않는다.

**최종 결정:** Test의 Original을 포함한 5조건 전체를 이 정책으로 새로 생성한다. 기존 544건을 포함한 이전 판단과 압축 결과는 가져오지 않는다. 변인통제를 위해 첫 Test 모델 호출부터 같은 보완 정책을 적용한다. 이전에 제안했던 「544건 보존 후 이어가기」는 사용하지 않는다(D014).

이전 `runs/memory_check/report.json`의 FAIL은 출력 동일성 검사 결과로 보존한다. 실제 확인에서 판단 5개·추출형 일반 1개는 같았고 나머지 압축 3개는 달랐다. 9회 모두 실행은 끝났고 입력은 같았다. 새 Test에서는 이전 출력이 정답 기준이 아니므로, 같은 출력을 강제하는 조건을 두지 않는다.

새 기술 점검은 **최장 Dev 1문서에서 압축 4회·판단 5회**를 실행해 입력 해시·입력 길이·greedy·메모리 정책·라벨/추출 ID 형식을 확인한다. 기존 출력과의 일치 여부는 따로 기록하며 통과 조건에 사용하지 않는다. Gold·정확도는 통과 조건이 아니고, 원본 Dev도 수정하지 않는다. 새 출력 전문은 `outputs.jsonl`에 보관한다.

기존 실험이 실행 중이면 `Ctrl+C`로 멈추고 최신 GitHub 코드를 업데이트한다. 특히 **`run_memory_safe.py`와 `memory_support/` 폴더 전체**를 함께 업데이트한다. `.venv`, 데이터·검수 자료·prepared·model_lock.json, 기존 run·protocol을 보존한다. 의존성 설치·모델 다운로드·중복 검수·전체 Dev 5,185건을 반복하는 절차가 아니다. 기본 Main 패키지와 config는 바꾸지 않았다.

다음 명령은 `runs/dev_greedy`, `protocol_greedy`까지 완료한 현재 실행을 기준으로 한다. 예전 검사 폴더와 구분하려고 새 검사 폴더를 쓴다.

```powershell
python -m unittest discover -s tests -v
python run_memory_safe.py check-dev --run runs/dev_greedy --out runs/memory_check_fresh
```

`Dev technical check: PASS`를 확인한다. `old outputs identical: .../9`는 참고 정보다. 형식·설정 검사 FAIL이나 OOM은 원인을 확인해야 하며 보고서를 임의 수정하지 않는다. 검사 중 중단되면 보고서를 보존하고 새 `--out`으로 다시 검사한다.

이어서 새 정책을 동결하고 **새 Test를 초기화**한다. `protocol_greedy`의 기존 Test 등록을 삭제하거나 수정하지 않는다.

```powershell
python run_memory_safe.py init-test --base-protocol protocol_greedy --protocol-out protocol_memory_safe --out runs/test_memory_safe
```

이 명령은 캐시된 토크나이저만 사용하며 모델 가중치를 로드하지 않는다. `protocol_memory_safe`에 기본 protocol 사본·보완 정책·소스·기술 점검을 보관하고, 새로운 Test run에는 원문 문맥만 만든다. 이전 예측·압축 결과를 0건 가져왔다는 안내를 확인한다. 기본 119문서를 사용한다면 총 판단은 10,115건이다. 다른 검사 폴더를 썼다면 이 초기화 명령에만 `--check-report 해당폴더/report.json`을 추가한다.

Original부터 **`1/2023`**으로 새로 평가하고 기준선을 분석한다.

```powershell
python run_memory_safe.py --debug judge --run runs/test_memory_safe --conditions original
python run_memory_safe.py analyze --run runs/test_memory_safe --conditions original
```

그다음 네 압축 조건을 새로 생성하고 판단·진단·전체 분석을 진행한다.

```powershell
python run_memory_safe.py --debug compress --run runs/test_memory_safe
python run_memory_safe.py --debug judge --run runs/test_memory_safe --resume
python run_memory_safe.py diagnose --run runs/test_memory_safe
python run_memory_safe.py analyze --run runs/test_memory_safe
```

Original은 이미 완료했으므로 마지막 판단의 `--resume`이 이를 건너뛰고 압축 4조건 8,092건을 실행한다. 중간에 중단되면 필요한 모델 단계에 `--resume`을 붙인다. 최종 결과는 `runs/test_memory_safe/results/`에 있다.

**통제와 기록:** 새 protocol의 모델 runtime에도 보완 정책 해시를 포함한다. 일반 `run_experiment.py`로 모델 실행을 시도하면 첫 생성 전에 runtime 불일치로 차단한다. 모든 모델 실행은 `run_memory_safe.py`를 쓴다. 원래 동결 소스 검사는 유지하며 추가 소스 사본·해시·검증·실행 정책은 `runtime_memory_policy.json`과 `memory_extension_source/`에 보관한다. 새 예측은 첫 행부터 정책이 있어야 한다. `report.md`와 `analysis.json`에도 처음부터 통일된 정책과 이전 모델 출력 재사용 0건을 기록한다.

짧은 입력/긴 입력의 실제 연산은 동일한 고정 규칙으로 분기하므로 모든 호출의 커널이나 시간이 같다는 뜻은 아니다. 실제 GPU에서 Test 최장 문서가 처리되는지는 로컬 실행으로 확인해야 한다. 에이전트는 모델 가중치 로드·GPU 모델 추론을 수행하지 않았다. D013의 혼합 실행 제안은 D014로 대체됐다.

## 추가 분석 1: 17개 가설 목록 제공

질문 없는 Main을 **Q0**, 같은 계약서에 고정 17개 가설 목록을 함께 제공한 조건을 **Q17**로 비교한다. 네 압축 방법 각각 문서당 한 번만 압축하고, 만든 문맥을 17개 가설 판단에 재사용한다. 판단 모델은 Main과 똑같이 문맥과 가설 **한 개씩** 받는다.

변경은 압축 user 입력에 공통 가설 안내와 17개 목록을 추가하는 것뿐이다. Main의 system·기본 작업 지시·보존 우선 문장, 모델 리비전, NF4 4bit·bfloat16, 명시적 greedy, thinking off, 안전 상한·정지 문자열, 추출 단위와 원문 순서 조합, 파싱·실패 처리, OOM 보완 정책을 유지한다. 길이 목표·최소 길이·재생성은 추가하지 않는다. 가설은 검증할 주장이라는 안내를 넣고 Gold Label·Gold Evidence는 제공하지 않는다.

**업데이트:** 최신 GitHub ZIP을 내려받아 코드·문서를 업데이트한다. 특히 `run_question_analysis.py`, **`question_support/` 전체**, `tests/test_questions.py`가 있어야 한다. `.venv`, 데이터·검수 자료·prepared·model_lock.json, 기존 run·protocol은 보존한다. 기존 Main 패키지와 OOM 보완 소스·config·의존성 버전은 이번 변경에서 수정하지 않았다. 설치·다운로드·중복 검수·전체 Dev 판단을 반복하지 않는다.

완료한 Main의 데이터·제외 목록·설정을 읽으므로 추가 분석용 config를 따로 편집하지 않는다. Main과 동일한 가상환경을 활성화한 IDE 터미널에서 아래 순서로 실행한다. 예시는 현재 완료한 `runs/test_memory_safe`와 `runs/dev_greedy`를 기준으로 한다. 자신의 run 이름이 다를 때만 해당 인자를 변경한다.

### 1. 최소 검사와 소규모 Dev 점검

```powershell
python -m unittest discover -s tests -v
python run_question_analysis.py --debug preflight --main-run runs/test_memory_safe --dev-run runs/dev_greedy --out runs/questions17_preflight --documents 1
```

Q17 입력이 가장 긴 Dev 1문서를 토크나이저로 고른다. 압축 4회와 다섯 조건 × 17가설 = 85개 판단을 실행해 입력 증가 후의 메모리·추출 ID·라벨 형식과 설정을 점검한다. **Dev 정확도는 계산하지 않는다.** `Q17 preflight: PASS`를 확인한 다음 진행한다. 길이·상한 도달·대화 플래그는 진단에 남기며 이를 이용해 유리한 출력을 골라내지 않는다. 실패하면 `diagnostics/technical_summary.json`과 오류를 먼저 확인한다.

사전 점검이 중단됐을 때는 같은 명령에 `--resume`을 붙인다. 완료된 출력과 실패 기록은 유지하며, 자동으로 재생성하지 않는다.

### 2. 별도 Test 초기화

```powershell
python run_question_analysis.py init --main-run runs/test_memory_safe --preflight runs/questions17_preflight --out runs/questions17
```

사전 점검의 프롬프트·실제 runtime·코드를 확인하고 `frozen_protocol.json`에 Q17 규칙을 동결한다. Main의 Original 문맥과 판단 결과만 정확히 복사한다. 질문 없는 네 압축 방법은 비교 기준으로 참조하며 Q17 출력으로 복사하지 않는다. 데이터 제외 목록과 문서·가설은 Main과 완전히 같다.

119문서라면 **새 압축 476회, 새 판단 8,092건, Original 재사용 2,023건**이다. 저장 결과에는 Original을 포함한 10,115건이 있으므로 새 모델 호출 수와 구분한다. 압축 실패는 기존 정책대로 오답 기록으로 남아 실제 모델 판단 호출 수가 줄 수 있다. 한 사전 점검으로 Test 하나만 초기화하며 기존 폴더를 덮어쓰지 않는다.

### 3. 압축 → 판단 → 진단 → 비교 분석

```powershell
python run_question_analysis.py --debug compress --run runs/questions17
python run_question_analysis.py --debug judge --run runs/questions17
python run_question_analysis.py diagnose --run runs/questions17
python run_question_analysis.py analyze --run runs/questions17
```

Original은 다시 판단하지 않는다. 진행 표시의 전체 저장 목표에는 이미 복사한 Original이 포함될 수 있다. Q17 모델 단계는 반드시 `run_question_analysis.py`로 실행한다. 다른 실행 파일로 Q17 run을 처리하면 프롬프트·소스 식별자가 달라 거부된다. 추가 분석을 시작한 뒤 Main run·코드·모델·환경을 바꾸지 않는다. 비교·재개 시에도 Main의 무결성을 검사하므로 참조 Main 폴더를 유지한다.

컴퓨터 종료 등으로 중단되면 **중단된 단계**만 다음처럼 재개한다. `--resume`은 완료한 성공·실패 행을 모두 건너뛰며 마지막 불완전 기록만 복구한다. 초기화 명령은 다시 실행하지 않는다.

```powershell
python run_question_analysis.py --debug compress --run runs/questions17 --resume
python run_question_analysis.py --debug judge --run runs/questions17 --resume
```

### 4. 확인할 결과와 프롬프트

결과 위치는 `runs/questions17/results/`다.

| 파일 | 내용 |
|---|---|
| `report.md` | Q0와 Q17의 방법별 성능·보존율·실패·추가 실행 시간 요약 |
| `question_effects.csv` | **질문 효과의 주 비교**: Q17−Q0 Accuracy·Macro-F1과 paired 문서 부트스트랩 95% CI, 보존율·실패·정답 전환 |
| `question_label_effects.csv` | 클래스별 Recall과 예측 라벨 수의 변화 |
| `question_times.csv` | Main 관측 시간과 새 실행 시간, Original 재사용 구분 |
| `question_analysis.json` | 비교 정의·설정·비용·효과 전체 |
| `metrics.csv`, `comparisons.csv` | Q17 다섯 조건의 성능 및 **Original−Q17** 비교. Q17−Q0와 구분 |

같은 문서의 17문항을 함께 복원추출하고 Main과 Q17에 같은 문서 표본을 적용한다. 10,000회·seed 42는 Main 설정을 그대로 읽고, 매 반복 Macro-F1을 재계산한다. CI는 다중 비교를 보정하지 않은 탐색적 결과다. 실패 문서를 분모에서 제거하지 않으며 형식 실패 변화도 성능 차이의 원인일 수 있다. 실제 길이가 달라질 수 있어 **길이와 무관한 질문 효과**라고 단정하지 않는다.

추가된 공통 안내 전문은 다음과 같다. 이어서 `H1`부터 `H17`까지 Main 스냅샷 순서의 가설 텍스트를 넣는다. H 번호는 원문 단위 ID와 구별하기 위한 표시이며 공식 가설 ID·순서는 스냅샷에 함께 남긴다.

```text
Use the following hypotheses to guide which contract information to retain. They are claims that will be evaluated, not facts to assume. Do not add information that is absent from the contract, and do not answer the hypotheses.

<evaluation_hypotheses>
H1: ...
...
H17: ...
</evaluation_hypotheses>
```

실제 17개 문장과 네 방법의 프롬프트 전문은 생성된 run의 `prompts.json`에서 확인한다. 실행 전 모델을 로드하지 않고 보려면 다음 명령을 쓴다.

```powershell
python run_question_analysis.py show-prompts --main-run runs/test_memory_safe --out runs/questions17_prompts.json
```

결론은 ‘평가할 고정 가설 목록을 미리 아는 효과’로 한정한다. 가설별 맞춤 재압축이나 정답 제공 실험이 아니다. 설계는 [설계서](design/EXPERIMENT_DESIGN.md), 변경 이유와 검증 범위는 [의사결정 기록](docs/DECISIONS.md)의 D015를 참고한다.

## 이후 추가 분석 — 아직 실행 기능 없음

| 분석 | 확정한 범위 |
|---|---|
| 추가 분석 2: 길이 제약 비교 | 질문 없음과 17개 가설 목록 제공에 같은 원문 기준 예산 적용. 목표 보존율·문서·방법·질문 유무별 문맥 하나를 17가설에 재사용. 수치·허용오차는 이후 결정. |
| 추가 분석 3: 근거·오류 분석 | 이미 생성된 공유 문맥을 각 가설의 Gold Evidence와 대조. 가설별 재압축 없음. |

## 파일 구성과 프롬프트

```text
run_experiment.py             CLI 진입점
run_memory_safe.py            처음부터 통일한 OOM 대비 Test CLI
memory_support/               전체 문맥 attention·기술 점검·새 Test 동결
run_question_analysis.py      추가 분석 1: 17개 가설 목록 제공 CLI
question_support/
  prompts.py                  Main 지시 + 공통 17가설 목록
  state.py                    Main 참조 검증·Q17 스냅샷·별도 식별자
  runner.py                   기술 점검·초기화·압축·판단·재개
  comparison.py               Q17−Q0 paired 비교·길이·실패·시간
config.toml                   메인 설정, 길이 목표 없음
requirements.txt              직접 의존성 고정
contract_experiment/
  data.py                     공식 스키마·중복 검토·원문 단위
  review.py                   오프라인 원문·라벨·Gold Evidence 비교 보고서
  prompts.py                  프롬프트 전문·ID/라벨 파싱
  runtime.py                  토크나이저·4bit 모델 어댑터
  experiment.py               단계 실행·재개·규칙 동결
  metrics.py                  메인 성능·문서 부트스트랩
  settings.py                 설정·해시·환경
  storage.py                  JSON/CSV·내구성 기록·실행 잠금
  cli.py                      명령어 정의
tests/                       Main·OOM·Q17 합성 워크플로와 CPU 최소 검증
design/EXPERIMENT_DESIGN.md   합의한 설계
design/DESIGN_COMPARISON.md   팀 공유용 비교 문서
docs/DECISIONS.md             의사결정과 추가 기록 양식
docs/IMPLEMENTATION.md        구현 규칙과 검증 범위
CHANGELOG.md                  변경 이력
```

Main 프롬프트 전문은 [contract_experiment/prompts.py](contract_experiment/prompts.py), Q17의 추가 블록은 [question_support/prompts.py](question_support/prompts.py)에 있다. 각 실행 파일의 `show-prompts`로 출력할 수 있고 전문·삽입 형식은 run의 `prompts.json`에 보존한다. Main 압축에는 가설이 없고 Q17 압축에는 전체 가설 목록이 있다. 어느 압축에도 Gold를 넣지 않으며, 판단에는 문맥과 가설 한 개가 들어간다. 추출형 ID는 판단 문맥에 들어가지 않는다.

구현에 확인한 공식 자료: [ContractNLI 스키마·배포](https://stanfordnlp.github.io/contract-nli/), [Qwen3 thinking off](https://huggingface.co/Qwen/Qwen3-8B#enable_thinkingfalse), [Transformers 4bit 양자화](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes), [bitsandbytes Windows 배포](https://pypi.org/project/bitsandbytes/0.48.1/).
