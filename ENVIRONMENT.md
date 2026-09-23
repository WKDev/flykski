# flykski 개발 환경

## 0) 다른 PC로 옮기기 (2026-09-23 패키징)
GitHub: https://github.com/WKDev/flykski (flybody는 서브모듈, `d015e9b` 고정). 절차는 [README.md](README.md) "설치" 절:
`git clone --recursive` → `conda env create -f environment.yml` → `pip install -e ./flybody` → `python setup/verify.py`.
- 아래 1)~2)의 경로(`C:\Users\chson\...`)는 원래 개발 PC 기준 기록이다. 새 PC에선 conda env 이름(`flybody`, `malecns`)만 같으면 된다.
- 시크릿: `.env`/`.Renviron`은 레포에 없다(.gitignore). `.env.example` 참고. 커넥톰 CSV는 레포에 포함돼 있어 R/토큰 없이도 시뮬레이션·학습은 전부 가능.
- 하드웨어: 지금 워크로드(MuJoCo CPU 물리 + 소형 numpy LIF + 순차 ES)는 GPU 이득이 거의 없고 **CPU 코어 수**가 중요(README "하드웨어" 절). 원 개발 PC: 16코어, NVIDIA GPU 없음.
- 실시간 창: `python -m flyski_sim.live_view` (macOS는 `mjpython`). 헤드리스 Linux 렌더링은 `MUJOCO_GL=egl`.

## 요약
- OS: Windows 11, 패키지 매니저: winget으로 Miniconda3 설치.
- 두 개의 독립 conda 환경을 사용한다 (flybody는 Python 3.10 고정 요구사항, malecns는 R — 하나로 합치지 않음).

## 1) flybody (Python, MuJoCo)
```
conda 환경: flybody  (Python 3.10.21)
경로: C:\Users\chson\miniconda3\envs\flybody
소스: C:\Users\chson\Desktop\flykski\flybody  (TuragaLab/flybody, git clone)
설치: pip install -e .   (core만 설치, [tf]/[ray] extra는 RL 학습 시 추가 설치 필요)
```
실행 예:
```bash
"C:/Users/chson/miniconda3/envs/flybody/python.exe" -c "
from flybody.fly_envs import walk_imitation
env = walk_imitation()
"
```
**검증 완료**: `pytest tests/test_flywalker.py tests/test_walking_env.py tests/test_core.py` 14개 전부 통과 (2026-09-23).

**참고**: `flybody/download_data.py`로 사전학습 체크포인트/모방학습 데이터셋(walking, flight)을 추가로 받을 수 있음 — 업무3~5에서 사전학습 컨트롤러 재사용/모방학습 워밍업 시 필요.

## 2) malecns (R, natverse)
Windows용 R 공식 GUI 인스톨러가 이 실행 환경(샌드박스)에서 응답 없이 멈추는 문제가 있어(추정: UAC/데스크톱 세션 접근 제한), **conda-forge의 r-base로 우회 설치**했다.
```
conda 환경: malecns  (R 4.5.3, conda-forge r-base)
경로: C:\Users\chson\miniconda3\envs\malecns
```
R 실행 시 conda 환경의 DLL 경로를 PATH에 추가해야 한다(안 하면 "shared object file" 오류):
```bash
export PATH="/c/Users/chson/miniconda3/envs/malecns:/c/Users/chson/miniconda3/envs/malecns/Library/mingw-w64/bin:/c/Users/chson/miniconda3/envs/malecns/Library/usr/bin:/c/Users/chson/miniconda3/envs/malecns/Library/bin:/c/Users/chson/miniconda3/envs/malecns/Scripts:/c/Users/chson/miniconda3/envs/malecns/bin:$PATH"
Rscript -e '...'
```
**natmanager 사용 불가**: `natmanager`가 내부적으로 쓰는 `pak` 패키지가 conda-forge R 빌드와 바이너리 비호환("Wrong OS or architecture, pak is probably dysfunctional")이라 로드 실패. 대신 `remotes::install_github("natverse/malecns", dependencies=TRUE)`로 설치.

**검증 완료 (2026-09-23)**: `library(malecns)` 로드 성공, 기본 데이터셋 `male-cns:v1.0` 확인. `dr_malecns()` 실행 결과 패키지/네트워크 설정은 정상이며, 예상대로 인증 단계에서만 막힘:
```
Error: You must supply an authorisation token for neuprint.janelia.org
Error in clio_auth(): Clio/Google auth failure.
```

**neuprint 인증 완료 (2026-09-23)**: 토큰을 `C:\Users\chson\Desktop\flykski\.env`(NEUPRINT_TOKEN=...)에 발급받아, **`C:\Users\chson\.Renviron`에도 동일 값을 등록**해 모든 R 세션에서 자동으로 로드되게 했다(malecns README가 권장하는 `usethis::edit_r_environ()`와 동일한 효과, `usethis` 없이 직접 파일 생성). `.env`는 flykski 프로젝트 전용(git 저장소 아님, 커밋 위험 없음), `.Renviron`은 R 전역 설정 — 둘 다 시크릿이므로 공유/커밋 금지.

**실제 조회 검증 완료**:
```bash
export PATH="/c/Users/chson/miniconda3/envs/malecns:/c/Users/chson/miniconda3/envs/malecns/Library/mingw-w64/bin:/c/Users/chson/miniconda3/envs/malecns/Library/usr/bin:/c/Users/chson/miniconda3/envs/malecns/Library/bin:/c/Users/chson/miniconda3/envs/malecns/Scripts:/c/Users/chson/miniconda3/envs/malecns/bin:$PATH"
Rscript -e 'library(malecns); mcns_neuprint_meta("/.+_[adl]+PN")'
```
- PN(투사뉴런) 쿼리: 219개 뉴런 조회 성공 (예: bodyid 12653, type "DA1_lPN").
- **업무4 핵심 신경 확인**: `DNg100` 2개 bodyId, `DNb08` 4개 bodyId가 male-cns:v1.0에 타입명 그대로 존재함을 확인 (04_connectome_interface.md에 반영 완료).
- Clio 쓰기 접근(공동연구자 전용)은 별도이며 아직 미설정 — 읽기 전용 neuprint 조회에는 불필요.

## 알려진 이슈 로그
- winget으로 R 공식 인스톨러 설치 시도 2회 모두 무한 대기(멈춤) → 강제 종료 후 conda-forge 우회로 해결.
- `natmanager::install()` → pak 바이너리 비호환 오류 → `remotes::install_github()`로 우회.
- `flybody/download_data.py`로 사전학습 체크포인트/모방학습 데이터셋 다운로드 시도 → figshare가 AWS WAF 봇 차단 챌린지(`x-amzn-waf-action: challenge`, 202 응답에 본문 없음)를 걸어 `requests` 라이브러리로는 다운로드 불가. **브라우저로 직접 다운로드해야 함**: https://doi.org/10.25378/janelia.25309105 (janelia figshare, "controller-reuse-checkpoints"/"walking-imitation-dataset" 등) 접속 후 zip을 받아 `flybody/flybody-data/`에 압축 해제. 업무3~5에서 사전학습 컨트롤러/모방학습 워밍업이 필요할 때 진행.
- `remotes::install_github("natverse/malecns", dependencies=TRUE)` 자체는 정상 동작하나, natverse 의존성 체인(nat → neuprintr → nat.templatebrains → catmaid → fafbseg → malevnc → nat.h5reg → nat.jrcbrains → nat.flybrains → coconat → malecns)이 길어 **총 소요 시간 약 35~40분**. Rtools 미설치 경고가 각 단계마다 뜨지만 실제로는 CRAN 사전빌드 바이너리(.zip)를 받아 쓰므로 문제없이 완료됨.

## 다음에 할 일
- [x] neuprint.janelia.org에서 토큰 발급 후 `NEUPRINT_TOKEN` 설정, `mcns_neuprint_meta()` 실제 조회 테스트 (2026-09-23 완료).
- [ ] flybody `download_data.py`로 보행 사전학습 체크포인트/모방학습 데이터셋 받기 (브라우저로 수동 다운로드 필요, 위 이슈 로그 참고).
- [x] 업무4 R 추출 스크립트 완료 (2026-09-23, `flyski_sim/connectome/extract_leg_cpg_circuit.R`): DNg100/DNb08 → 인터뉴런(IN09A002/IN12B003/IN01A015/IN17A001/dPR1) → **실제 다리 운동뉴런(73종 360개)**까지 이어지는 전체 경로를 male-cns:v1.0에서 실증. CSV 4종을 `flyski_sim/connectome/`에 저장. 원 논문과의 정확한 대조는 아직(정황증거만 강함) — 04번 문서 참고.
- [x] Python `ConnectomeController` 구현 완료 (2026-09-23, `flyski_sim/connectome_controller.py`). tanh rate-model은 리듬 안 생김(RESEARCH_NOTES 18번) → **LIF 모델(`connectome_controller_lif.py`)로 교체하니 실제 리듬 발생 확인**(19번). real/shuffled가 random보다 10배 이상 강하게 진동함, real vs shuffled 우열은 아직 불명확(업무5에서 통계검정 필요). 테스트: `python -m flyski_sim.connectome_controller_test`, `python -m flyski_sim.connectome_controller_lif_test`.
- [x] LIF 리듬 → flybody 관절 통합 완료 (2026-09-23, `flyski_sim/motor_decoder.py` + `connectome_to_flybody_test.py`). `femur_T1_left` 관절이 실제로 진동함을 확인(std=0.079). 근육-관절 매핑은 명시적 근사(RESEARCH_NOTES 20번), 좌우 side 구분 없음.
- [x] 협응된 전진 보행인지 확인함 (2026-09-23) — **아직 아님**. 다리 간 위상 상관이 전부 거의 0(삼각보행이면 |0.7~0.9| 기대), thorax는 초반에 낮은 자세로 주저앉은 뒤 거기서 다리만 흔듦. 예상된 결과(학습 전 원본 파라미터라 그럼) — RESEARCH_NOTES.md 21번.
- [x] 업무5 첫 학습 루프(2026-09-23, `flyski_sim/train_connectome_params.py`) — **성공적인 첫 결과**. 처음엔 진화전략이 정체됐다가(elitism 없음) elitism+sigma decay로 고치니 fitness -0.27→2.14로 개선, 실측 위상 상관 T1L-T3L=+0.816(강한 양의 상관, 삼각보행 기대와 일치)까지 확인. RESEARCH_NOTES.md 22~23번 필독.
- [x] real/shuffled/random 첫 비교 (2026-09-23, n=1 시드): **real=2.14 > shuffled=1.29 > random=0.64**(random은 8세대 내내 거의 정체) — 가설과 일치하는 방향이지만 시드 1개뿐이라 통계적 결론 아님(5회 이상 반복 필요, RESEARCH_NOTES 24번).
- [x] 후각 신경군 추출 완료 (2026-09-23, `flyski_sim/connectome/extract_olfactory_pathway.R`). **DA1(페로몬) PN → 하행뉴런(DNpe052 등) → 다리 CPG 명령 뉴런(DNg100/DNb08)까지 단 2홉으로 직접 연결됨을 확인** — 놀랍도록 짧은 "냄새→걷기명령" 경로. 좌우(side) 정보가 없어서 좌우 비대칭 조향 가능 여부는 아직 미확인(RESEARCH_NOTES 25번, 이게 세 번째 반복된 실수).
- [x] real/shuffled/random 5회 반복 통계검정 완료 (2026-09-23, `topology_comparison_results.csv`). **중요한 음성 결과: 통계적으로 유의미한 차이 없음**(real=1.08±0.69, shuffled=0.92±0.46, random=0.92±0.62, Mann-Whitney p=0.69~1.00 전부). n=1 예비 결과(real이 제일 좋아 보였던 것)는 노이즈였음 — RESEARCH_NOTES.md 26번 필독(이 세션에서 가장 중요한 발견 중 하나).
- [x] 역할군(4그룹, 10차원) 파라미터 재검증 완료 (2026-09-23) — **가설을 뒷받침 못했고 순위가 반대로 나옴**: random(1.84) > real(1.15) > shuffled(0.91), 전부 통계적으로 비유의(p>0.06). real이 "침묵 네트워크" 실패 모드에 가장 자주(60%) 빠짐. RESEARCH_NOTES.md 27번 — 다음 사람/세션이 판단할 열린 질문으로 남김.
- [x] 좌우(somaSide) 정보 보강 완료 (2026-09-23) — 14/20/25번에서 세 번 지적됐던 문제. 기존 CSV 4개에 `somaSide`/`rootSide` 컬럼 추가, `motor_decoder.py`가 이제 실제로 좌/우 액추에이터를 구분해서 구동함(검증 완료). RESEARCH_NOTES.md 28번.
- [x] 29번 원인 재분석 → ① 발화율 보정 + 침묵 페널티 재비교 완료(2026-09-23, `run_topology_comparison_calibrated.py`): 침묵 실패 0%로 제거, 하지만 shuffled 1.60 > real 1.24 > random 1.13, 전부 비유의(RESEARCH_NOTES 31번).
- [x] 경사 지형 치명적 버그 3종 수정(파묻힌 스폰, hfield 높이 규약, 항력이 토크로 적용) + 뷰어 검은 화면 수정(RESEARCH_NOTES 30번). 이전 경사 지형 결과는 무효.
- [x] 스키 재설계: 좌/우 휘는 플레이트 N=2(`ski_plate.py`, 기본), 스키 전용 마찰(`SnowParams.ski_friction`, priority=1)로 실제 활강 확인(RESEARCH_NOTES 32번).
- [ ] 29번 ② 적합도 개선(전진거리/진폭 하한/다중 시드/held-out 재평가), ES 평가 multiprocessing 병렬화.
- [ ] 컨트롤러를 스키 사면(side_plate)에서 학습 — 지금까지 학습은 전부 평지.
- [ ] 완전 뉴런별(392차원) 파라미터 확장, 게이트 기반 세션 프레임워크는 아직 미착수. 이 시점에서 real vs shuffled/random 비교는 "차이 없음(오히려 반대 방향도 관찰)"이 현재까지의 결론(2번의 독립 실험 모두) — 실험설계 재점검 필요.
- [x] 업무1(지형) 1단계 프로토타입을 flybody MuJoCo 월드에 붙이는 작업 (2026-09-23 완료, `flyski_sim/` — 실행: `python -m flyski_sim.smoke_test` in flykski 루트).
- [x] `max_slope_deg` 직접 제어 (2026-09-23 완료 — `calibrate_mogul_height_for_max_slope`, 이분탐색으로 mogul_height 자동 보정).
- [x] 설질(마찰) 존 구현 (2026-09-23 완료 — `flyski_sim/snow.py` 5종 프리셋 + `SlopedMoguls`의 x축 구간 배정 + 컨트롤 스텝마다 마찰 런타임 갱신. 여러 설질 동시 물리 표현은 미지원, RESEARCH_NOTES.md 11번 참고).
- [x] 냄새 게이트 필드 (2026-09-23 완료 — `flyski_sim/odor.py`(`OdorField`), 순차 점화, 순번/교대/기울기/완주 전부 검증). 업무1 나머지는 2단계(무한 스트리밍)뿐 — 의도적으로 뒤로 미룸(마스터 프롬프트 권장 순서: 1단계 먼저 끝내고 업무2~3으로 베이스라인부터 완성).
- [x] 업무2(스키 프로파일 5종)/업무3(N=6 compliant 부착 + 파생 텔레메트리) 핵심 부분 완료 (2026-09-23 — `flyski_sim/ski_profiles.py`, `ski_attachment.py`, `telemetry.py`, 테스트: `python -m flyski_sim.ski_test`, `python -m flyski_sim.telemetry_test`). **중요**: 스키가 실제로 땅에 안 닿던 버그와 엣지각이 다리마다 잘못 계산되던 버그를 둘 다 발견·수정함(RESEARCH_NOTES.md 14~15번, 꼭 읽어볼 것). `xfrc_applied` 침투 보조력도 완료(2026-09-23, `python -m flyski_sim.drag_test`). 남은 건 실제 보행 컨트롤러로 걷기 검증(사전학습 체크포인트 필요, figshare 봇차단으로 아직 못 받음).

**시각 검증 방법 (2026-09-23 추가)**: `python -m flyski_sim.render_demo`로 PNG/GIF를 `renders/`에 저장. Unity/Unreal 안 씀(MuJoCo 자체 렌더러 재사용, GUI 창 없이 헤드리스로 이미지 저장 — 이 환경에서 오프스크린 렌더링은 정상 동작 확인됨). `renders/connectome_driven_legs.gif` — 업무4 커넥톰 리듬으로 실제 다리가 움직이는 모습.
