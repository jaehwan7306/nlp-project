# ContractNLI 문맥 압축 Main Experiment

Python **3.11** 프로젝트다. ContractNLI 원문과 질문 없는 압축 4조건을 같은 판단 모델로 평가한다. 이전 실험의 코드·결과·중복 제외 목록을 가져오지 않았다. 질문 제공, 목표 보존율 제약, Gold Evidence 오류 사례 분석은 이번 구현에 포함하지 않는다.

| 조건 ID | 조건 | 압축 출력 |
|---|---|---|
| `original` | Original | 원문 |
| `generative_general` | 생성형·일반 / Naive | 새 요약문 |
| `generative_priority` | 생성형·보존 우선 / Contextualized | 새 요약문 |
| `extractive_general` | 추출형·일반 | 원문 단위 ID 선택 |
| `extractive_priority` | 추출형·보존 우선 | 원문 단위 ID 선택 |

압축·선택은 모두 **Qwen2.5-7B-Instruct**, 판단은 **Qwen3-8B, thinking off**다. 두 모델 모두 같은 4bit NF4 설정을 사용한다. 같은 출력 방식에서는 보존 우선순위 문장 하나만 다르다. 네 조건 모두 질문·정답·Gold Evidence와 목표 보존율을 압축 입력에 넣지 않는다.

모델은 자연 종료하며 최소 생성 길이를 강제하지 않는다. `4096` 생성 토큰은 네 압축 작업에 공통으로 적용하는 **실행 안전 상한의 초기값**이다. 목표 보존율이나 원문 길이에 따른 예산이 아니다. 상한 도달·가짜 대화·빈 출력·원문 전체 복사 등을 기록하고 Dev에서 적절성을 확인한다. 결과가 같거나 실제 길이가 같다고 가정하지 않는다.

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

| 검토 결과 | `decision` | `notes` 예시 |
|---|---|---|
| 실질적으로 같은 계약서로 확인 | `duplicate` | 어떤 표기·서명 차이가 있고 같은 계약서라고 판단한 근거 |
| 비슷하지만 별개의 계약서 | `distinct` | 서로 다른 의무·기간·조항 등 차이 |

Train과의 중복은 보고만 하므로 `review_required=False` 행은 비워도 된다. 정확히 같은 원문 후보도 이번에는 사람이 확인한다. 검토를 마쳤으면 실행한다.

```powershell
python run_experiment.py prepare --audit data_audit/audit.json --review data_audit/review.csv --out prepared
```

미검토 후보가 있으면 `prepare`가 중단되는 것이 정상이다. 같은 계약서·같은 라벨인 그룹은 Test를 우선하고 같은 split에서는 작은 ID 하나를 유지한다. **같은 내용인데 라벨 충돌이 확인된 그룹은 전체 제외**하며 임의 재라벨링하지 않는다. 단순히 유사도와 라벨이 다르다는 이유만으로 제외하지 않는다. Dev/Test 교차 중복이면 Test를 유지하고 Dev에서 제외한다.

`prepared/exclusions.csv`의 제외 사유와 `prepared/manifest.json`의 최종 문서 수를 확인한다. 과거의 119·122 등 문서 수를 강제하지 않는다. Test가 N문서이면 메인은 **5 × 17N**건이다. 데이터나 검토를 바꾸면 새 audit/prepared 폴더를 사용한다.

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

```powershell
python run_experiment.py init-run --split test --protocol protocol --out runs/test
python run_experiment.py judge --run runs/test --conditions original
python run_experiment.py analyze --run runs/test --conditions original
```

Original 기준선은 `runs/test/results/baseline/`에서 확인한다. Test 결과를 보고 설정을 바꾸지 않는다. 규칙은 이미 동결한 상태다.

```powershell
python run_experiment.py compress --run runs/test
python run_experiment.py judge --run runs/test --resume
python run_experiment.py diagnose --run runs/test
python run_experiment.py analyze --run runs/test
```

하나의 protocol은 Test run 하나만 시작할 수 있다. 중단되면 같은 폴더에서 재개한다. 새 Test를 초기화해서 처음부터 다시 돌리는 흐름을 기본으로 제공하지 않는다.

## 8. 결과를 보는 곳

전체 메인 결과는 **`runs/test/results/`**에 생성된다. CSV 비율은 `0.30 = 30%`다.

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
- `runtime_compressor.json`, `runtime_judge.json`: 실제 GPU·전체 패키지 버전·채팅 템플릿·생성/양자화 설정.
- `source/`, `manifest.json`: 실행 시점 소스 사본과 무결성 해시.
- `events.jsonl`: 모델 로딩 시간 등 실행 이벤트.

성능 분모에는 모든 문항을 유지한다. 추출 파싱 실패는 해당 17문항을 압축 실패 오답으로 기록하고, 판단 파싱 실패도 오답으로 처리한다. 길이를 알 수 없는 압축 실패는 보존율을 0으로 꾸미지 않고 결측으로 남긴다. `[]`는 정상적인 빈 선택이므로 빈 문맥을 판단하고 보존율 0으로 측정한다.

문서 단위 클러스터 부트스트랩을 사용한다. 문서를 복원추출하고 그 문서의 17문항을 함께 포함하며, 같은 표본으로 방법을 비교하고 Macro-F1도 매 반복 재계산한다. 기본은 10,000회, seed 42, percentile 95% CI다. 메인의 길이는 통제하지 않았으므로 같은 길이에서의 우열을 의미하지 않는다.

## 9. 중단·재개와 변경 기록

완료한 레코드는 즉시 저장한다. 예를 들어 Test 압축 또는 판단 중 중단했다면 필요한 단계만 재개한다.

```powershell
python run_experiment.py compress --run runs/test --resume
python run_experiment.py judge --run runs/test --resume
```

기록된 실패 출력도 완료된 관측으로 유지하며 다시 생성하지 않는다. GPU OOM 같은 실행 오류는 그 항목을 완료로 저장하지 않고 중단한다. 동일한 설정으로 환경 자원을 확보한 뒤 재개한다. 소스·프롬프트·설정이 달라지면 같은 run에 섞어서 재개할 수 없다. 도중에 끊긴 JSONL 마지막 행만 원본 백업 후 복구하고, 중간 행 손상이나 중복 키는 오류로 처리한다.

향후 변경 때 다음을 남긴다.

1. [CHANGELOG.md](CHANGELOG.md): 무엇을 바꿨고 어떤 범위에 영향을 주는지.
2. [docs/DECISIONS.md](docs/DECISIONS.md): 문제, 선택지, 선택 이유, 변인통제 영향과 검증.
3. 동결 전 변경이라면 새 Dev run. 동결한 소스·설정·Test 결과는 덮어쓰지 않는다.

같은 하드웨어·버전에서 재현할 수 있도록 seed와 결정적 연산 설정을 기록한다. greedy라도 다른 GPU·라이브러리 사이의 비트 단위 일치를 보장하지는 않는다. 결정적 구현이 없는 연산 경고도 실행 로그와 함께 보관한다.

## 파일 구성과 프롬프트

```text
run_experiment.py             CLI 진입점
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
tests/test_core.py            합성 데이터 최소 검증
design/EXPERIMENT_DESIGN.md   합의한 설계
design/DESIGN_COMPARISON.md   팀 공유용 비교 문서
docs/DECISIONS.md             의사결정과 추가 기록 양식
docs/IMPLEMENTATION.md        구현 규칙과 검증 범위
CHANGELOG.md                  변경 이력
```

프롬프트 전문은 [contract_experiment/prompts.py](contract_experiment/prompts.py)에 있고 `show-prompts`로 출력할 수 있다. 실행 당시 전문과 삽입 형식은 각 run의 `prompts.json`에 보존한다. 압축 입력에는 가설·Gold가 없고, 판단 입력에는 문맥과 가설이 들어간다. 추출형 ID는 판단 문맥에 들어가지 않는다.

구현에 확인한 공식 자료: [ContractNLI 스키마·배포](https://stanfordnlp.github.io/contract-nli/), [Qwen3 thinking off](https://huggingface.co/Qwen/Qwen3-8B#enable_thinkingfalse), [Transformers 4bit 양자화](https://huggingface.co/docs/transformers/v4.57.1/en/quantization/bitsandbytes), [bitsandbytes Windows 배포](https://pypi.org/project/bitsandbytes/0.48.1/).
