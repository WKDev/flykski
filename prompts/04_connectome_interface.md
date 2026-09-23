# 업무 4 — MaleCNS 커넥톰: 다리 CPG + 후각 스티어링 컨트롤러 (v2)

> 상위 컨텍스트: [00_master_prompt.md](00_master_prompt.md). 업무3의 `Observation`/`FlySkiEnv.step(action_dim)` 계약을 그대로 사용한다(102 하드코딩 금지, 03 참고).

## 목표
[natverse/malecns](https://github.com/natverse/malecns)로 접근하는 MaleCNS 커넥톰(neuprint `male-cns:v1.0`, [male-cns.janelia.org](https://male-cns.janelia.org/))의 **실제 위상을 고정**한 채, (1) 다리 운동을 만드는 CPG 경로와 (2) 냄새 기울기를 따라 좌우 조향을 만드는 후각 경로 두 개를 하나의 컨트롤러로 구성해, 업무3의 `action_dim` 관절 액추에이션을 출력한다. FlyGM("Whole-Brain Connectomic Graph Model Enables Whole-Body Locomotion Control in Fruit Fly", Jin et al., arXiv 2602.17997)의 설계 원칙(위상 고정 + 뉴런별 소수 학습 파라미터 + encoder/decoder, 그래프 신경망을 RL로 학습)을 참고 설계로 채택한다.

## 배경·근거 (교차검증으로 정정됨 — 아래 두 논문의 결과를 이 프로젝트가 자동으로 승계한다고 가정하지 않는다)
- **FlyGM의 실제 조건**: 이 논문은 **MaleCNS가 아니라 FlyWire(뇌만, VNC 없음) 커넥톰**을 사용했고, flybody의 사전학습 보행/비행 컨트롤러가 만든 시연 궤적으로 **모방학습(IL) 워밍업 후 PPO로 미세조정**하는 2단계 파이프라인이었다. 대조군은 degree-preserving 재배선 그래프, Erdős–Rényi 랜덤 그래프, MLP였고, 실제 위상이 세 대조군보다 학습 손실/각도 오차가 낮았다. **본 프로젝트는 다른 커넥톰(MaleCNS 부분그래프), 다른 과제(스키/후각 스티어링, 시연 데이터 없음)를 쓰므로 "실제 위상이 더 낫다"는 결과가 재현될지는 검증 대상 가설이지, 이미 보장된 결과가 아니다.** 업무5에서 이를 가설로 명시하고 통계적으로 검정한다.
- **CPG 경로의 실제 근거**: bioRxiv 10.1101/2025.09.12.675944 (Pugliese, Chou, Abe, Turcu, Lancaster, Tuthill, Brunton, "Connectome simulations identify a central pattern generator circuit for fly walking")는 초파리 VNC 커넥톰 시뮬레이션에서 하행뉴런 **DNg100**(걷기 명령 뉴런)이 리듬 생성의 최상위 드라이버임을 찾았고, **억제성 1개 + 흥분성 2개로 구성된 최소 CPG 회로**가 6개 다리 전체의 운동 리듬에 필요·충분함을 보였으며, 별도의 하행경로 **DNb08**도 리듬성 다리 운동을 유발함을 광유전학으로 확인했다. **단, 이 결과는 지속/강직 입력에 대한 개방루프(open-loop) 리듬 생성을 검증한 것이며, 자세 되먹임이 있는 폐루프 보행 제어나 관절 토크로의 직접 디코딩까지 검증한 것은 아니다.** 또한 DNb08 활성화가 논문에서 (걷기와 구별되는) 비접촉 다리 흔들림(flailing)과 함께 보고된 맥락이 있으므로, 이 회로를 실제 보행/스키 제어에 쓰려면 **폐루프 연결과 안정성 검증을 이 프로젝트에서 새로 수행해야 한다** — CPG 논문이 "이미 검증된 걷기 컨트롤러"를 제공하는 것은 아니다.
- **검증 완료(2026-09-23, 실제 neuprint 조회)**: `mcns_neuprint_meta()`로 male-cns:v1.0에서 `DNg100`(2개 bodyId, 좌우 한 쌍으로 추정) 및 `DNb08`(4개 bodyId)이 타입명 그대로 직접 조회됨을 확인했다. 즉 CPG 논문의 핵심 하행뉴런은 male-cns:v1.0에 존재하며, `mancType` 크로스레퍼런스 폴백은 이 두 타입에는 불필요하다(억제성 1개+흥분성 2개 인터뉴런 3종은 아직 미확인 — 신경군 추출 단계에서 실제로 조회해 타입명을 확정할 것).
- 후각 경로: NeuroMechFly/FlyGym 계열이 후각 플룸 추적(취주) 내비게이션을 지원한다는 선례가 있다 — 후각수용체뉴런(ORN) → 촉각엽(antennal lobe) → 버섯체/측각(mushroom body/lateral horn) → 하행뉴런(descending neuron) 경로가 MaleCNS에도 식별 가능해야 하며, 이 경로의 출력이 좌우 CPG 구동 비대칭을 만들어 조향을 일으킨다. 업무1과 마찬가지로 **냄새원은 한 번에 하나만 활성화**된다는 전제로 이 경로를 설계한다(동시 다중 소스는 기울기 간섭을 일으킴).
  - **후각 신경군 추출 완료(2026-09-23, `flyski_sim/connectome/extract_olfactory_pathway.R`)**: DA1 글로머룰루스(male CNS라 남성 특이적 페로몬 cVA 처리 채널을 대표 후각 입력으로 채택) PN 15개에서 출발, 1홉 하류에서 하행뉴런 8개(`DNpe052`, `DNb05`, `DNp32`, `DNc01`, `DNp29`)를 확인했다. **핵심 발견: 이 하행뉴런 중 `DNpe052`/`DNp29`/`DNp32`가 우리가 이미 찾은 다리 CPG 명령 뉴런(`DNg100`, `DNb08`)에 직접 연결된다**(가중치 최대 11, `DNpe052`→`DNg100`) — 즉 "냄새 → 조향"과 "냄새 → 걷기 명령"이 뉴런 수준에서 거의 같은 병목(DNg100/DNb08)을 공유하는 짧은 경로가 실제로 존재한다. **아직 확인 못한 것**: 좌우(side) 정보가 없어서 이게 "좌우 비대칭 조향"에 실제로 쓰일 수 있는 구조인지(양쪽 안테나가 대칭적으로 같은 뉴런에 모이는 건지, 좌우 분리된 경로가 따로 있는지)는 모른다 — 다음 추출 시 `somaSide` 컬럼을 포함해서 재확인 필요.

## 상세 요구사항

### 1) 신경군 추출 (R / natverse, malecns)
- `mcns_neuprint_meta()`와 연결성 쿼리로 다음 세 그룹을 확보:
  - **CPG/다리 운동군**: 위 bioRxiv 논문에서 언급된 DNg100, DNb08, 그리고 이들이 구동하는 억제성/흥분성 인터뉴런 3종을 neuprint에서 타입명으로 조회(원문 표기 그대로 검색, 없으면 `mancType` 크로스레퍼런스로 MaleCNS 대응 뉴런 탐색) + 이와 시냅스로 연결된 다리 전운동뉴런/운동뉴런(efferent, 관절 액추에이터에 직접 대응).
  - **후각 입력군**: ORN/투사뉴런(PN) 계열, 냄새 농도 대리 입력을 받는 afferent.
  - **후각-조향 연결군**: PN → 버섯체/측각 → 하행뉴런 경로 상의 intrinsic/efferent 뉴런.
- 세 그룹과 그 사이 실제 시냅스 가중치(카운트 기반)를 `neurons.parquet`/`connectivity.parquet`로 추출, 데이터셋 버전(`male-cns:v1.0`) 기록.
- 신경군 선정 근거(조회 쿼리, 타입명, 참고 논문, `mancType` 매핑 결과)를 별도 문서로 남긴다 — 추정이 아니라 실제 조회 결과여야 한다. DNg100/DNb08는 male-cns:v1.0에서 직접 확인됐으므로(위 배경 참고), **남은 것은 억제성 1개+흥분성 2개로 구성된 최소 CPG 인터뉴런 3종**이다 — 이들은 원 논문에서 명시적 타입명이 아니라 회로 내 역할로 기술되었을 가능성이 있으므로, DNg100/DNb08의 시냅스 하류(downstream) 연결을 실제로 따라가며 후보를 찾는 탐색적 쿼리 절차를 문서화한다(원 논문의 방법/보충자료에서 구체적 타입명이 있는지 먼저 확인).
  - **신경군 추출 완료(2026-09-23, `flyski_sim/connectome/extract_leg_cpg_circuit.R`)**: DNg100 하류를 `superclass=='vnc_intrinsic'`으로 필터링해 억제성 `IN09A002`/`IN12B003`(gaba), 흥분성 `IN01A015`/`IN17A001`/`dPR1`(acetylcholine) 후보를 확보(32 bodyId, T1/T2/T3 전부에 반복 등장 — dPR1만 T1 한정). 후보군 내부 연결(137 edges)을 조회한 결과:
    - **`IN17A001`(흥분성) → `IN09A002`(억제성)** 연결이 T1/T2/T3 전 분절에서 압도적으로 강함(196/261/431, 후방으로 갈수록 강해짐) — 매 분절 반복되는 핵심 모티프.
    - `IN01A015`(흥분성)도 `IN09A002`로 보조적 흥분 입력(T1=43, T2=41)을 줌 — "흥분성 2개(IN17A001, IN01A015)가 억제성 1개(IN09A002)에 수렴"하는 구조가 T1/T2에서 일관되게 나타남(T3는 약하게 혼재).
    - `DNg100`이 5개 인터뉴런 후보 전부에 강하게 직접 연결됨(가중치 146~456, 전 분절) — "최상위 드라이버"라는 논문 기술과 부합.
    - **`IN09A002`가 실제 이름 붙은 다리 운동뉴런(`superclass=='vnc_motor'`)에 직접 연결됨을 확인**: `MNhl29`, `MNml29`, `Fe reductor MN`(femur), `Acc. ti flexor MN`(tibia), `Sternal anterior/posterior rotator MN`, `Pleural remotor/abductor MN` 등 — coxa/femur/tibia/tarsus 전 관절에 걸친 73종, 360개 모터뉴런, 627개 연결(`03_leg_motor_neurons.csv`/`04_interneuron_to_motor_connectivity.csv`). **DNg100(명령) → 흥분/억제 인터뉴런 → 실제 다리 근육 모터뉴런까지 끊기지 않고 이어지는 경로를 실증**.
    - **여전히 확정 아님**: 이게 원 논문이 기술한 "정확히 그 3-뉴런 회로"인지는 원문 대조를 못 했다(정황증거가 강할 뿐).

### 2) 신경 제어망 구성 (Python, FlyGM 레시피) — 구현 완료, 중요한 부정적 결과 있음
`flyski_sim/connectome_controller.py`(`ConnectomeController`)로 구현·검증 완료 (2026-09-23, `python -m flyski_sim.connectome_controller_test`).
- 위상(어떤 뉴런이 어떤 뉴런과 흥분/억제로 연결되는지)은 고정. 학습 대상은 뉴런별 소수 파라미터(decay/gain/bias, 총 1176 = 392뉴런×3)뿐 — FlyGM과 동일 원칙.
- afferent(DNg100/DNb08, 6개)/intrinsic(인터뉴런)/efferent(다리 운동뉴런, 360개) 3분할 구조 명시적으로 유지. 전체 392개 뉴런, 764개 연결.
- 대조군 세 조건(`topology_mode='real'|'shuffled'|'random'`)을 동일 인터페이스로 지원, 노드/엣지 수는 세 조건 모두 동일(392/764), 실제 위상 행렬만 다름을 확인.
- `get_learnable_params()`/`set_learnable_params()` 왕복 검증 통과.
- **뉴런 모델 관련 중요 발견(2단계 결과)**:
  1. **1차 시도(tanh rate-model)는 실패**: DNg100/DNb08에 지속 구동을 줬을 때, 단순 tanh rate-model은 gain(1~15)·decay(균일/이질적 0.05~0.95)·bias를 넓게 스윕해도 리듬 없이 고정점으로 수렴했다(`real`/`shuffled`/`random` 전부).
  2. **2차 시도(LIF로 교체) 성공**: `flyski_sim/connectome_controller_lif.py`(`LIFConnectomeController`, 불응기 있는 leaky integrate-and-fire)로 같은 위상을 재구동하니 **뚜렷한 리듬(약 4~5스텝 주기 반복 패턴)이 실제로 나타났다**(`weight_scale=30, v_thresh=0.5, tau_mem=8.0`, `python -m flyski_sim.connectome_controller_lif_test`). 뉴런 모델(연속 rate vs 스파이킹+불응기) 선택이 실제로 결정적이었다는 뜻.
  3. **real vs shuffled vs random 비교(5시드)**: `real`(std=0.0140, 활성도 0.113)과 `shuffled`(std=0.0151, 활성도 0.057)는 둘 다 진동하고, `random`(std=0.0013, 활성도 0.002)은 10배 이상 약하다 — "진짜 배선이든 degree만 비슷한 셔플이든 어떤 구조화된 연결이 있으면 발진하지만, 완전 무작위는 거의 발진 못 한다"는 결과. **다만 이 시점에서 `real`이 `shuffled`보다 명확히 우월하다고는 아직 말할 수 없다**(오히려 std는 shuffled이 근소하게 높음, 대신 활성도/zero-crossing 패턴은 다름) — 단일 파라미터 지점·5시드 결과일 뿐이라 업무5에서 제대로 된 통계 검정이 필요하다.
  - **(c) flybody 통합까지 완료(2026-09-23)**: `flyski_sim/motor_decoder.py`로 다리 운동뉴런 타입명(고전 곤충 다리근 명명법: promotor/remotor, reductor, flexor/extensor, levator/depressor)을 flybody의 관절 액추에이터(coxa_twist/coxa_abduct/coxa/femur/tibia × T1~T3 × 좌우)로 키워드 매칭해 디코딩하고, `flyski_sim/connectome_to_flybody_test.py`로 실제 flybody 물리 시뮬레이션에 주입했다. 결과: 400스텝 안정, `femur_T1_left` 관절각이 실제로 진동(std=0.079, range 0.12~0.49, 다수의 zero-crossing) — **커넥톰에서 나온 리듬이 실제 다리 관절을 움직이는 것까지 처음으로 확인됨**. 단, 이 매핑은 명시적 근사다: (i) `superclass=='vnc_motor'`로 뽑은 360개 중 이름으로 알아볼 수 있는 18종 타입만 매핑했고 나머지(MNad*/MNhl*/DLMn/DVMn 등, 비행근으로 추정)는 제외, (ii) 근육-관절 대응은 male-cns로 직접 검증하지 않고 표준 곤충 다리근 명명 관례로 추정, (iii) 좌우(side) 정보가 CSV에 없어서 좌우 액추에이터에 동일 신호를 적용(진짜 비대칭 걸음은 아직 불가능).
  - **협응된 보행인지 확인(2026-09-23) — 아직 아니다**: 몸통(thorax) z높이가 시작 50스텝 안에 0.298→0.044로 한 번 떨어진 뒤 나머지 450스텝 동안 그 낮은 높이에서 거의 평평하게 유지됨(계속 쓰러지는 건 아니고, 다른 낮은 자세로 정착). 다리 간 위상 상관도 약함(T1_left-T2_left = -0.045, T1_left-T3_left = 0.104, T1_left-T1_right = -0.042) — 삼각보행이면 기대되는 |상관| 0.7~0.9 근처와는 거리가 멀다. **결론: 지금 상태는 "다리가 흔들린다"는 확인되지만 "협응된 전진 보행"은 아니다.** 이건 놀라운 결과가 아니라 오히려 예상된 결과에 가깝다 — 18번에서 확인했듯 이 프로젝트는 학습(파라미터 튜닝) 전 원본 위상만으로는 정교한 행동이 안 나온다는 걸 반복해서 확인하고 있다. 업무5의 학습 루프가 실제로 필요한 지점.
  - **다음 후보**: (a) 명시적 감각 되먹임 루프 추가로 리듬이 더 안정/뚜렷해지는지, (b) real vs shuffled 차이를 더 많은 파라미터 지점·시드에서 통계적으로 검증(업무5), (c) 좌우 side 정보를 다시 추출해서 진짜 좌우 비대칭 조향 기반 마련, (d) 업무5 학습 루프로 실제 삼각보행이 나오는 파라미터를 탐색(진화전략/RL 대상 확정).

### 3) 센서 → 신경 입력 매핑
업무3 `Observation`의 필드를 다음과 같이 라우팅한다:
- `odor.concentration`, `odor.gradient_xyz` (양안/양측 antenna 위치 차이로 좌우 농도차 계산) → 후각 입력군(afferent).
- `proprioception.joint_angles/velocities`, `ski_telemetry.plate_load_share` → CPG 경로의 되먹임 입력(고유수용감각 대리).
- `terrain_local.slope_deg` → 필요 시 속도 조절용 보조 입력(경사가 급할수록 감속 경향 등, 선택적 라우팅).
- 정규화 규칙(범위, 클리핑)을 `connectome_mapping.yaml`로 버전 관리.

### 4) 신경 출력 → 관절 액추에이션 디코딩
- CPG/운동군의 efferent 활성을 flybody의 `action_dim`차원 관절 액추에이션으로 디코딩하는 decoder(FlyGM처럼 조건부 소형 MLP 허용, 단 입력은 반드시 efferent 뉴런 활성이어야 함). CPG 논문의 회로는 개방루프 리듬 생성만 검증됐으므로, 이 decoder와 아래 5)의 되먹임 연결이 실제로 안정적인 폐루프 보행을 만드는지는 **이 프로젝트에서 처음 검증하는 부분**임을 전제로 설계·테스트한다.
- 후각-조향 경로의 출력은 **좌우 CPG 구동 비대칭**(예: 좌측 CPG 게인을 올리고 우측을 내리는 식의 변조 신호)으로 주입해, 별도의 "조향 액추에이터"를 새로 만들지 않고 걷기/스키 동작 자체를 좌우로 휘게 만든다.

### 5) 실행 주파수
- 신경망 갱신 주기는 물리 스텝보다 낮은 고정 주파수(예: 50~100Hz)로 두고, 그 사이는 zero-order hold. flybody 기본 컨트롤러 주기와 맞춘다.

## 인터페이스 계약
```
ConnectomeController.load(neurons_path, connectivity_path, mapping_config, topology_mode: "real"|"shuffled"|"random")
ConnectomeController.step(observation: Observation, dt: float) -> joint_actuation: np.ndarray[action_dim]  # 업무3 FlySkiEnv.action_dim과 동일
ConnectomeController.get_learnable_params() -> np.ndarray   # 업무5 학습 대상 (뉴런별 소수 파라미터 + encoder/decoder)
ConnectomeController.set_learnable_params(params: np.ndarray)
```

## 산출물
- [x] R 추출 스크립트(`flyski_sim/connectome/extract_leg_cpg_circuit.R`) + CSV 4종(`01_candidate_neurons.csv`, `02_internal_connectivity.csv`, `03_leg_motor_neurons.csv`, `04_interneuron_to_motor_connectivity.csv`, 데이터셋 버전 male-cns:v1.0 기록). parquet 변환은 안 했음(CSV로 충분, 후속 과제).
- [x] 신경군 선정 근거 — 이 문서의 "배경·근거"/"상세 요구사항 1)" 절 자체가 근거 문서 역할.
- [x] 후각 입력군/후각-조향 연결군 추출 완료(`extract_olfactory_pathway.R`) — DA1 PN → 하행뉴런(8개) → 다리 CPG 명령 뉴런(DNg100/DNb08) 직접 연결 확인. 좌우(side) 구분은 아직 없음.
- [x] Python `ConnectomeController` (real/shuffled/random 3조건 지원) — 완료. `connectome_mapping.yaml`(센서 입력 라우팅 규칙)은 아직 — 후각 경로가 없어서 아직 매핑할 입력이 DNg100/DNb08 구동값 하나뿐.
- [ ] flybody 사전학습 베이스라인과 비교 스모크 테스트 — 사전학습 체크포인트가 아직 없어서(figshare 봇차단) 보류.

## 수용 기준
- [x] CPG군 신경 타입이 실제 neuprint 조회 결과이며 참고 논문/쿼리 근거가 문서화됨(위 배경 참고). 후각군은 아직.
- [x] DNg100(2 bodyId)/DNb08(4 bodyId)가 male-cns:v1.0에서 타입명으로 직접 조회됨(2026-09-23 검증 완료).
- [x] 최소 CPG 인터뉴런 후보(억제성 IN09A002/IN12B003, 흥분성 IN01A015/IN17A001/dPR1)를 male-cns:v1.0에서 실제로 특정하는 탐색 절차가 코드로 실행됨(`extract_leg_cpg_circuit.R`), 실제 다리 운동뉴런까지 연결 확인됨.
- [x] `topology_mode="real"`과 `"shuffled"/"random"`이 동일 뉴런 수/연결 수(392/764)를 가지면서 위상만 다름(단위테스트로 확인).
- [ ] 냄새 기울기를 좌/우로 인위 주입했을 때(단위테스트), 디코딩된 좌우 관절 토크에 비대칭이 발생함 — 후각 경로 자체가 아직 없어서 불가능.
- [x] `ConnectomeController`(LIF 버전)가 flybody와 연결되어 크래시 없이 400스텝 완주함(2026-09-23, `connectome_to_flybody_test.py`). 단 스키 없는 평지 태스크로 확인했고(`SlopeSmokeTask`), 스키 장착 `FlySkiEnv` 조합은 아직 안 해봄 — 다음 단계.
- [x] `get/set_learnable_params`가 왕복 검증됨(코드 테스트).
- [x] **(신규, 2026-09-23)** DNg100/DNb08에 지속 구동 시 다리 운동뉴런 층에서 리듬(진동)이 발생함 — **LIF 모델로 교체 후 충족**(`LIFConnectomeController`). tanh rate-model에서는 미충족이었다가, 뉴런 모델을 스파이킹+불응기로 바꾸니 해결됨(위 2절 참고). real/shuffled가 random보다 10배 이상 강하게 진동하는 것까지 확인했으나, real vs shuffled 우열은 아직 불명확 — 업무5에서 통계적으로 재검증할 것.

## 확장 아이디어 (선택)
- 후각 경로 대신(또는 추가로) 시각(광류) 경로를 붙여 두 감각의 기여도 비교.
- CPG 신경군을 다르게 선정한 대안 버전과의 성능 비교.
