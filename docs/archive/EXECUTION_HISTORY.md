# 이전 실행의 오류 복구 기록

이 문서는 2026-11-14~15의 디코딩 수정·OOM 보완 도입 절차를 보존한다. 현재 신규 재현은 [재현 및 실행 명령](../REPRODUCIBILITY.md)을 따른다. 이미 완료한 현재 Main·Q17 결과를 다시 실행하는 절차가 아니다.

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

