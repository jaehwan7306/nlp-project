# 재현 및 터미널 실행 명령

[프로젝트 개요](../README.md) · [연구 질문별 결과 보고서](RESEARCH_QUESTION_RESULTS.md) · [공개 결과 파일](../results/README.md)

README가 있는 **프로젝트 폴더를 IDE에서 열고 그 IDE의 터미널**에서 실행한다. 모든 경로는 프로젝트 루트 기준이며 아래 명령은 Windows PowerShell용이다.

| 현재 상태 | 시작할 곳 |
|---|---|
| 처음부터 재현 | [A. 환경·전처리·Dev·Original·Q0](#from-scratch) |
| Original·Q0 완료 후 메인 비교 실행 | [B. Main의 Q17](#questions17) |
| Q0와 Q17을 완료하고 질문 8개를 실험 | [C. Analysis 1의 Q8](#questions8) |
| 실행 도중 중단 | [D. 중단 후 재개](#resume) |
| 저장된 결과를 확인 | [E. 결과와 재현 자료](#outputs) |
| 팀원이 실제 압축문과 Gold Evidence를 사용 | [F. 팀 분석에 필요한 파일](#team-data) |

이 가이드는 완료한 Main과 Analysis 1을 재현하는 절차다. [Analysis 2](../design/EXPERIMENT_DESIGN.md#analysis-2)와 [Analysis 3](../design/EXPERIMENT_DESIGN.md#analysis-3)의 예정 분석은 설계서에서 확인할 수 있다.

이미 완료한 현재 실험에 신규 초기화 명령을 반복할 필요는 없다. 코드를 업데이트할 때 `.venv`, 데이터·검수 자료·prepared·model_lock.json, 기존 run·protocol을 보존한다. 이전 오류 복구 절차는 [과거 실행 기록](archive/EXECUTION_HISTORY.md)에 분리했다.

<a id="from-scratch"></a>

## A. 메인 실험의 Original·Q0

### 1. Python 3.11 환경과 고정 의존성

기록된 실행은 Python **3.11.9**, PyTorch **2.8.0+cu128**, NVIDIA GeForce RTX 5070에서 수행했다. 동일 결과를 비교하려면 가능한 한 같은 Python·GPU·패키지 환경을 사용한다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
python -m pip freeze > environment-lock.txt
```

`Python 3.11.x`와 `CUDA: True`를 확인한다. PowerShell 활성화가 차단되면 해당 터미널에서 `Set-ExecutionPolicy -Scope Process Bypass` 후 활성화하거나 `python` 대신 `.\.venv\Scripts\python.exe`를 사용한다. 모델은 한 번에 하나만 GPU에 올리며 12GB 환경의 실제 처리 가능 여부는 사전 점검에서 확인한다.

모델 추론 없이 최소 검증을 실행한다. 설치 전에는 실제 Transformers·attention을 쓰는 일부 검사가 skip될 수 있으므로 설치 후 확인한다.

```powershell
python -m unittest discover -s tests -v
python run_experiment.py --help
```

### 2. 공식 데이터와 사람의 중복 검토

```powershell
python run_experiment.py fetch-data --dest data
python run_experiment.py audit-data --data data --out data_audit --threshold 0.95
Invoke-Item data_audit/review_report.html
```

유사도는 정규화한 연속 5단어 집합의 Jaccard다. 0.95는 **후보 탐색 기준**이며 자동 제외 기준이나 문자 일치율이 아니다. HTML에서 원문 차이·가설별 라벨·Gold Evidence를 보고 사람이 `data_audit/review.csv`의 `decision`과 `notes`를 채운다. 행이나 pair ID를 삭제하지 않는다.

| `decision` | 의미 |
|---|---|
| `duplicate` | 한 문서 유지. Test 우선, 같은 split에서는 작은 ID 유지 |
| `exclude_both` | 이 쌍의 두 문서를 모두 제외 |
| `distinct` | 이 쌍을 이유로 추가 제외하지 않음 |

`review_required=True`인 모든 행을 검토한 다음 진행한다. Gold 라벨 차이는 참고 자료이며 코드가 자동으로 제외 방식을 바꾸지 않는다. 현재 공개 결과의 결정은 [결과 문서](RESULTS.md#데이터와-실행-범위)에 기록했다. 같은 결과 집합을 재현하려면 같은 결정을 적용하고 검수 자료도 보관한다.

기존 audit 자료에서 HTML만 다시 만들 때는 다음 명령을 사용한다. 검수 CSV와 원본 데이터를 변경하지 않는다.

```powershell
python run_experiment.py review-report --audit data_audit/audit.json
```

검토가 끝났으면 prepared를 만든다.

```powershell
python run_experiment.py prepare --audit data_audit/audit.json --review data_audit/review.csv --out prepared
```

`prepared/exclusions.csv`와 `prepared/manifest.json`을 확인한다. 현재 결과는 Dev 61·Test 119문서지만 코드는 이 수를 강제하지 않는다. 결정을 바꿔 다시 준비할 때는 새 `--out`을 사용하고 이후 `--prepared`도 그 경로로 맞춘다. Train은 감사에만 사용한다.

### 3. 같은 모델 리비전 확보

보고된 결과의 모델 SHA는 [model_revisions.json](model_revisions.json)에 저장했다. 최신 리비전을 새로 조회하는 대신 이 파일을 새 프로젝트의 모델 잠금으로 사용한다.

```powershell
if (-not (Test-Path model_lock.json)) {
    Copy-Item docs/model_revisions.json model_lock.json
}
python run_experiment.py lock-models --config config.toml --out model_lock.json
python run_experiment.py download-models --config config.toml --models model_lock.json
```

`lock-models`는 이미 있는 유효한 잠금을 재사용한다. 기존 `model_lock.json`이 아래와 다르면 완료 run을 덮어쓰지 말고 별도 프로젝트에서 같은 리비전을 준비한다. 다운로드는 고정 SHA를 사용하고 추론은 기본적으로 로컬 캐시만 읽는다.

| 역할 | 모델 | 고정 SHA |
|---|---|---|
| 압축·선택 | Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| 판단 | Qwen/Qwen3-8B | `b968826d9c46dd6066d109eabc6255188de91218` |

### 4. 입력 길이와 소규모 Dev 점검

```powershell
python run_experiment.py inspect-inputs --config config.toml --prepared prepared --models model_lock.json --out runs/input_stats
python run_experiment.py init-run --split dev --documents 3 --out runs/preflight
python run_experiment.py --debug judge --run runs/preflight --conditions original
python run_experiment.py --debug compress --run runs/preflight
python run_experiment.py --debug judge --run runs/preflight --resume
python run_experiment.py diagnose --run runs/preflight
```

`inspect-inputs`는 토크나이저만 사용한다. `runs/input_stats/summary.json`의 `over_context_limit`과 preflight의 `diagnostics/technical_summary.json`, `lengths.csv`를 확인한다. preflight는 원문 문자 수가 가장 긴 Dev 3문서를 선택하고 255개 판단을 만든다.

점검 대상은 메모리·건수·종료·안전 상한·빈 출력·가짜 대화·추출 ID·라벨 파싱·실제 길이다. Dev 정확도를 계산해 유리한 프롬프트를 고르는 단계가 아니다. 변경이 필요하면 의사결정을 기록하고 새 Dev run에서 확인한다.

### 5. 전체 Dev와 규칙 동결

```powershell
python run_experiment.py init-run --split dev --out runs/dev
python run_experiment.py --debug judge --run runs/dev --conditions original
python run_experiment.py --debug compress --run runs/dev
python run_experiment.py --debug judge --run runs/dev --resume
python run_experiment.py diagnose --run runs/dev
```

진단을 확인한 후 동결한다. 소규모 preflight만으로는 동결할 수 없다.

```powershell
python run_experiment.py freeze --run runs/dev --out protocol
```

### 6. 전체 문맥 OOM 정책 점검과 Original·Q0 Test

긴 입력의 attention을 head별로 계산하는 정책을 Original·Q0 Test의 첫 모델 호출부터 적용한다. 전체 query/key 토큰을 유지한다.

```powershell
python run_memory_safe.py check-dev --run runs/dev --out runs/memory_check_fresh
```

`Dev technical check: PASS`를 확인한 다음 초기화한다. 최장 Dev 1문서의 압축 4회·판단 5회를 점검하며 이전 출력과의 일치 개수는 참고 정보다. 검사 중 중단됐다면 검사 보고서를 보존하고 새 `--out`으로 점검한다. 다른 검사 폴더를 쓰면 `init-test`에 `--check-report 해당폴더/report.json`을 지정한다.

```powershell
python run_memory_safe.py init-test --base-protocol protocol --protocol-out protocol_memory_safe --out runs/test_memory_safe
python run_memory_safe.py --debug judge --run runs/test_memory_safe --conditions original
python run_memory_safe.py analyze --run runs/test_memory_safe --conditions original
```

Original 기준선은 `runs/test_memory_safe/results/baseline/`에 생긴다. Test 결과를 보고 설정을 바꾸지 않고 네 압축 조건을 이어서 평가한다.

```powershell
python run_memory_safe.py --debug compress --run runs/test_memory_safe
python run_memory_safe.py --debug judge --run runs/test_memory_safe --resume
python run_memory_safe.py diagnose --run runs/test_memory_safe
python run_memory_safe.py analyze --run runs/test_memory_safe
```

`--resume`은 이미 완료한 Original을 건너뛴다. 한 새 protocol은 Test 하나만 초기화할 수 있다. Original과 Q0의 모델 단계는 `run_memory_safe.py`로 실행한다.

<a id="questions17"></a>

## B. 메인 실험의 Q17

완료한 Original·Q0와 같은 코드·환경·모델 잠금을 사용한다. `--main-run`에는 Original과 Q0가 담긴 `runs/test_memory_safe`를 지정한다. 아래 명령은 A에서 Dev를 `runs/dev`로 만든 경우다. 기존 Dev가 `runs/dev_greedy`라면 첫 명령의 `--dev-run`에 그 경로를 지정한다.

```powershell
python run_question_analysis.py --debug preflight --main-run runs/test_memory_safe --dev-run runs/dev --out runs/questions17_preflight --documents 1
```

`Q17 preflight: PASS`를 확인한 뒤 다음을 실행한다. Q17 입력이 가장 긴 Dev 1문서의 압축 4회·판단 85건을 확인하며 정확도는 계산하지 않는다.

```powershell
python run_question_analysis.py init --main-run runs/test_memory_safe --preflight runs/questions17_preflight --out runs/questions17
python run_question_analysis.py --debug compress --run runs/questions17
python run_question_analysis.py --debug judge --run runs/questions17
python run_question_analysis.py diagnose --run runs/questions17
python run_question_analysis.py analyze --run runs/questions17
```

119문서 기준 새 압축 476회와 새 판단 기록 8,092건을 만들고, Original 2,023건을 재사용한다. `analyze`는 같은 문서·가설에서 **Q17−Q0**를 계산한다. Q17의 모델 단계는 `run_question_analysis.py`를 사용한다. 압축에 실패한 문항은 오답으로 기록한다.

프롬프트 전문은 실행 전에도 모델 로딩 없이 볼 수 있다.

```powershell
python run_experiment.py show-prompts --out runs/main_prompts.json
python run_question_analysis.py show-prompts --main-run runs/test_memory_safe --out runs/questions17_prompts.json
```

실행 당시 전문·가설 순서·삽입 형식은 각 run의 `prompts.json`이 기준이다. Q17 압축 입력은 Q0의 기본 지시와 원문에 공통 안내·가설 목록을 더한 형태다. 판단 입력에는 원문 또는 압축 문맥과 가설 한 개를 제공한다. Gold 주석은 채점과 사후 분석에 사용한다.

<a id="questions8"></a>

## C. Analysis 1의 Q8

최신 코드를 기존 프로젝트에 가져오고, 같은 Python 3.11 가상환경을 활성화한 IDE 터미널에서 실행한다. Git으로 받은 프로젝트는 `git pull --ff-only`로 갱신할 수 있다. ZIP으로 받은 프로젝트는 최신 코드와 문서를 기존 프로젝트에 병합한다. 완료한 `runs/test_memory_safe`와 `runs/questions17` 폴더 전체가 필요하다. 다른 이름으로 저장했다면 첫 명령의 `--main-run`과 `--q17-run`을 실제 경로로 바꾼다.

```powershell
python run_question_subset.py init --main-run runs/test_memory_safe --q17-run runs/questions17 --out runs/questions8
```

초기화는 GPU 없이 두 참조 run의 문서·가설·Gold·설정·완료 기록을 검증한다. `runs/questions8/assignment.json`에 seed 42, 문서별 제공 ID와 가설별 제공 횟수가 저장된다. 119문서에서 각 가설은 56회 제공되고 63회 빠진다. 같은 문서의 네 압축 방법에는 같은 목록이 들어간다. Q17 프롬프트에서 미제공 가설 9줄을 제거하고 남은 가설의 원래 H 번호와 순서를 유지한다. `prompts.json`에 사용 형식과 예시가 들어 있다.

아래 명령은 앞 단계가 끝난 뒤 한 줄씩 실행한다.

```powershell
python run_question_subset.py --debug compress --run runs/questions8
python run_question_subset.py --debug judge --run runs/questions8
python run_question_subset.py diagnose --run runs/questions8
python run_question_subset.py analyze --run runs/questions8
```

119문서 기준 Q8 압축 476회와 판단 기록 8,092건을 새로 만든다. 모든 조건은 문서당 가설 17개를 평가한다. Original 2,023건은 복사하고 Q0·Q17 압축 결과와 판단 기록은 참조한다. 모델 호출이 필요한 단계는 `compress`와 `judge`다. 압축 실패 문항은 오답으로 기록하며 실제 판단 호출은 생략된다.

결과는 `runs/questions8/results/`에서 확인한다.

| 파일 | 내용 |
|---|---|
| `subset_report.md`, `subset_analysis.json` | 정확도 요약과 분석 설정 |
| `subset_effects.csv` | 전체 2,023문항, Q8 제공 952문항, 미제공 1,071문항의 방법별 정확도·Macro-F1. 같은 문항에서 Q8−Q0·Q8−Q17 차이와 paired 문서 부트스트랩 95% CI |
| `subset_hypothesis_effects.csv` | 각 가설에서 제공된 문서와 미제공 문서의 동일 문항 비교 |
| `paired_predictions.csv` | 문서·방법·가설별 Q8 제공 여부, Gold, 세 조건 예측과 상태 |
| `subset_resources.csv` | 보존율·실패·시간. Q0·Q17의 시간은 기존 실행 기록 |

제공·미제공별 결과를 볼 때 각 묶음 내에서 Q0·Q8·Q17을 비교한다. 세 조건 모두 동일한 문서·가설 조합을 쓴다. 원문과 평가용 Gold는 동일하게 유지하고 Gold는 압축 프롬프트에 들어가지 않는다. 길이와 가설 조합이 결과에 미친 영향도 해석에 포함한다.

<a id="resume"></a>

## D. 중단 후 재개

**중단된 단계만** 같은 run에서 재개한다. 초기화 명령을 반복하지 않는다.

```powershell
python run_memory_safe.py --debug judge --run runs/test_memory_safe --conditions original --resume
python run_memory_safe.py --debug compress --run runs/test_memory_safe --resume
python run_memory_safe.py --debug judge --run runs/test_memory_safe --resume
```

Q17 압축·판단도 필요한 단계에만 적용한다.

```powershell
python run_question_analysis.py --debug compress --run runs/questions17 --resume
python run_question_analysis.py --debug judge --run runs/questions17 --resume
```

Q17 preflight 도중 중단했다면 같은 preflight 명령에 `--resume`을 붙인다. Dev 단계에는 `run_experiment.py`의 해당 압축·판단 명령에 `--resume`을 붙인다.

Q8은 아래 중 중단된 단계만 이어서 실행한다.

```powershell
python run_question_subset.py --debug compress --run runs/questions8 --resume
python run_question_subset.py --debug judge --run runs/questions8 --resume
```

완료한 성공·실패 행은 모두 유지한다. OOM 같은 실행 오류로 미완료인 항목부터 재개하며 마지막 불완전 JSONL 행만 백업 후 복구한다. 소스·프롬프트·설정·모델·환경이 바뀌면 동일 run에 섞어 재개할 수 없다. 로그가 설명하는 원인을 먼저 확인하고 manifest·protocol 해시를 임의 수정하지 않는다. `--debug`는 항상 하위 명령 앞에 둔다.

<a id="outputs"></a>

## E. 결과와 재현 자료

완료한 결과만 확인하려면 [공개 결과 폴더](../results/README.md)의 CSV·JSON·보고서·그림을 열면 된다. 아래 명령은 원시 run을 보관한 로컬 프로젝트에서 집계를 다시 만들 때 사용한다.

이미 모델 실행이 끝났다면 다음은 저장된 결과를 읽는 진단·분석 명령이다.

```powershell
python run_memory_safe.py diagnose --run runs/test_memory_safe
python run_memory_safe.py analyze --run runs/test_memory_safe
python run_question_analysis.py diagnose --run runs/questions17
python run_question_analysis.py analyze --run runs/questions17
python run_question_subset.py diagnose --run runs/questions8
python run_question_subset.py analyze --run runs/questions8
```

| 파일·위치 | 내용 |
|---|---|
| `runs/test_memory_safe/results/` | Main의 Original·Q0 결과 |
| `runs/questions17/results/` | Main의 Q17 결과와 Q17−Q0 비교 |
| `runs/questions8/results/` | Analysis 1의 [Q8 집계](../results/questions8/report.md)와 동일 문항 Q0·Q8·Q17 비교 |
| `metrics.csv` | 정확도·Macro-F1·클래스 Recall·정답 유지율·실패율·보존율 |
| `comparisons.csv` | Original − 해당 압축 조건의 성능 차이·CI |
| `question_effects.csv` | **Q17 − Q0** 질문 효과·CI·실제 길이·실패·정답 전환 |
| `question_label_effects.csv`, `question_times.csv` | 라벨 변화·추가 실행 비용 |
| `report.md`, `analysis.json`, `question_analysis.json` | 요약과 집계 정의. `question_*`는 Q17에만 생성 |

CSV 비율은 `0.30 = 30%`다. 파싱·압축 실패 문항을 성능 분모에 포함한다. 길이를 알 수 없는 압축 실패는 결측이며 `[]`는 정상 빈 선택이라 보존율 0으로 측정한다.

팀 분석에 필요한 Test 결과, 문맥과 데이터 스냅샷은 공개 결과 폴더에서 내려받을 수 있다. 모델 가중치 캐시와 실행을 재개하는 데 쓰는 전체 `runs/`는 실행 환경에서 관리한다. [팀 분석용 공개 파일](#team-data)

재현에 필요한 로컬 기록은 다음과 같다.

- `data_audit/review.csv`, `prepared/exclusions.csv`, `prepared/manifest.json`: 사람의 결정·사유·제외·원본 해시.
- `model_lock.json`, `environment-lock.txt`: 고정 모델·환경. 실제 패키지는 run의 `environment.json`에도 기록.
- 각 run의 `dataset.json`, `prepared_manifest.json`, `prompts.json`, `config.json`, `models.json`, `manifest.json`, `frozen_protocol.json`.
- `contexts.jsonl`, `predictions.jsonl`, `runtime_compressor.json`, `runtime_judge.json`, `events.jsonl`: 원본 출력·시간·종료·실제 생성 설정·GPU·템플릿.
- `source/`, Original·Q0의 `memory_extension_source/`·`runtime_memory_policy.json`, Q8·Q17의 `lineage.json`, Q8의 `assignment.json`: 소스·정책·참조 run·배정표.

seed 42, 명시적 greedy, 패키지·리비전 고정은 재현을 돕지만 다른 GPU·라이브러리에서 모든 토큰이 비트 단위로 같음을 보장하지 않는다. Q8·Q17은 참조 run의 해시를 검사하므로 상대적 폴더 관계를 유지해 함께 보관한다. 변경 이유는 [DECISIONS](DECISIONS.md), 수정 내역은 [CHANGELOG](../CHANGELOG.md)에 남긴다.

<a id="team-data"></a>

## F. 팀 분석에 필요한 파일

**Gold Evidence가 압축 후에 남아 있는지 보려면** 원문의 근거 주석과 모델이 실제로 읽은 문맥을 대조한다. `contexts.jsonl`의 `text`는 판단 단계에 사용한 원문 또는 압축문이다. 추출형은 선택한 원문 위치를 `source_ranges`에도 기록한다. 아래 자료는 모두 저장소의 `results/`에서 받을 수 있다.

| 필요한 자료 | 파일 | 용도 |
|---|---|---|
| 공통 원문·Gold Evidence | [dataset.json](../results/dataset.json) | 실제 평가한 119문서의 원문·17개 가설·정답·근거 위치 |
| Main의 Original·Q0 문맥 | [Original·Q0 contexts.jsonl](../results/main/contexts.jsonl) | 원문 기준선과 질문 없이 만든 압축문 |
| Main의 Q17 문맥 | [Q17 contexts.jsonl](../results/questions17/contexts.jsonl) | 질문 17개를 주고 만든 압축문 |
| Analysis 1의 Q8 문맥 | [Q8 contexts.jsonl](../results/questions8/contexts.jsonl) | Q8 근거까지 비교할 때 사용하는 압축문 |
| Main 문항별 정답·예측 | [Original·Q0 CSV](../results/main/predictions.csv) · [Q17 CSV](../results/questions17/predictions.csv) | 오답에서 정답으로 바뀐 문항과 반대 사례 확인 |
| Analysis 1의 문항별 비교 | [paired_predictions.csv](../results/questions8/paired_predictions.csv) | Q8에 제공한 가설인지와 Q0·Q8·Q17 예측 확인 |

Q0와 Q17을 분석할 때는 공통 `dataset.json` 하나와 두 실행의 `contexts.jsonl`을 구분해 가져오면 된다. 세 실행의 데이터 스냅샷은 같은 119문서를 담고 있다. 가설별 근거 ID는 `annotations[가설 ID].spans`에서 읽고, 문서의 `spans` 목록에서 원문의 시작·끝 문자 위치를 찾는다. 문맥은 같은 실행 안에서 `document`와 `condition`으로, 판단 기록은 여기에 `hypothesis`까지 붙여 연결한다. 압축에 실패한 문맥은 `status`로 확인한다. 각 문맥·예측 파일의 로컬 출처와 해시는 [결과 manifest](../results/manifest.json)에 있다.

**Analysis 2에서 Original 입력의 근거 위치만 볼 때는** 공통 `dataset.json`의 원문과 공개 CSV의 `condition=original` 행을 연결하면 된다. 공식 `data/test.json`을 사용할 때는 같은 공개 예측 행의 문서 ID로 119문서를 고른다. 전체 입력 토큰 위치를 계산할 때는 현재 가설을 포함한 판단 프롬프트와 고정 Qwen3 토크나이저를 사용한다. [판단 프롬프트](../contract_experiment/prompts.py) · [모델 리비전](model_revisions.json)
