# 업무 2 — flybody 다리 부착형 스키 프로파일 (v2)

> 상위 컨텍스트: [00_master_prompt.md](00_master_prompt.md). [flybody](https://github.com/TuragaLab/flybody/)의 다리 세그먼트(coxa-femur-tibia-tarsus, 좌우 3쌍)에 부착되는 강체로 설계한다.

## 목표
flybody 다리 말단(tarsus)에 부착 가능한 스키 프로파일을 3~5종 정의한다. 새 물리 엔진을 만들지 않고, **MuJoCo 바디/지오메트리로 flybody 모델에 추가**하는 형태로 설계한다.

## 배경·근거
flybody는 6개 다리, 각 다리 tarsus가 지면 접촉의 실제 지점이다. **초기 설계(구버전)는 "삼각보행 그룹을 좌/우 스키 하나에 통째로 rigid weld"하는 안이었으나, 교차검증(codex) 결과 물리적으로 과구속(overconstraint)임이 확인되어 폐기한다.** 이유: (1) 실제 트라이포드는 좌/우가 아니라 대각선 조합(예: 좌전+우중+좌후 vs 우전+좌중+우후)이라 "좌측 스키/우측 스키"라는 지리적 이름 자체가 성립하지 않는다. (2) 보행 중 세 다리는 서로 다른 위상(지지기/유각기)으로 독립적으로 움직이는데, 이를 강체 하나에 용접하면 서로 다른 목표 위치를 동시에 요구하는 폐루프 구속이 생겨 구속력이 발산하거나 관절이 왜곡된다.

## 상세 요구사항

### 1) 다리-플레이트 결합 방식 (기본값 N=6, 다리당 1개, compliant 연결)
- **기본안(N=6)**: 다리 6개 각각의 tarsus에 독립적인 미니 스키를 부착한다. 각 스키는 자신이 붙은 다리 하나만 따라 움직이므로 과구속이 생기지 않는다. 결합은 처음부터 **약한 스프링-댐퍼(compliant joint)**로 하며, 완전 강체 `weld`는 사용하지 않는다(강체 결합도 다리 자체 관절 가동범위와 충돌할 수 있음 — 스프링 상수를 0에 가깝게 낮춰가며 안정성을 검증한다).
- **확장 옵션(N=2, "사람처럼 두 스키")**: 사람 스키 형태를 원하면, 다리 그룹 전체를 용접하지 말고 **그룹당 기준 다리(reference leg) 하나만** 스키에 결합하고 나머지 다리는 스키 표면 위를 미끄러지듯 접촉만 하는(별도 슬라이딩 콜라이더) 구성으로 구현한다. 이 옵션은 N=6 기본안이 안정적으로 동작한 뒤에만 시도한다.
- 어느 쪽이든 결합부는 다리의 자연스러운 상하 움직임이 스키에 하중 변화로 전달되도록 스프링-댐퍼 특성을 튜닝한다.

### 2) 프로파일 파라미터 스키마 (v1과 대부분 동일, 재확인)
- `length_m`, `width_profile`(tip/waist/tail), `sidecut_radius_m`, `flex_stiffness`(구간별), `torsional_stiffness`, `mass_kg`, `base_surface_area_m2`, `rocker_camber_profile`.
- 신규: `attachment_compliance`(다리-스키 결합부 스프링-댐퍼 상수) — flybody 관절에 가해지는 반작용을 결정하므로 필수 파라미터로 승격.
- 질량/치수는 flybody 스케일(초파리 실측 크기 기준 MuJoCo 단위)에 맞게 조정한다 — 사람 스키를 그대로 축소 비례하면 관성이 비현실적일 수 있으므로, "스키 대 몸통 질량비"를 사람 스키어 대비 유사한 비율로 스케일링한 근거를 문서화한다.

### 3) 프로파일 3~5종 (v1 후보군 유지, 결합 방식만 갱신)
| 프로파일 | 길이 | 특징 | 주 용도 |
|---|---|---|---|
| 슬라롬(Slalom) | 짧음 | 작은 사이드컷 반경, 높은 비틀림 강성 | 급회전, 좁은 게이트 간격 |
| 자이언트슬라롬(GS) | 중~장 | 중간 사이드컷 반경 | 넓은 게이트 간격, 고속 |
| 올마운틴(All-mountain) | 중간 | 균형 파라미터 | 범용 |
| 파우더(Powder) | 장, 폭 넓음 | 넓은 허리 폭, 강한 로커 | 파우더 설질 |
| 모글 스키(Mogul) | 짧~중, 소프트 | 낮은 종방향 강성 | 모글 지형 |

### 4) 물리 거동 매핑
- 사이드컷 반경 + (업무3에서 파생 계산되는) 유효 엣지각 → 카빙 회전 반경.
- 다리별 하중 차이가 곧 스키별(N=6 기본안에서는 다리당 1개) 하중 차이가 되며, 이는 flybody의 CPG 보행 위상(어느 다리가 지지기, 어느 다리가 유각기인지)에 실제로 영향을 받는다 — 즉 프로파일의 "느낌"이 걸음걸이 위상과 상호작용함을 설계에서 인지하고 로깅한다.

## 인터페이스 계약
- `SkiProfile` 정의(json/yaml) + `SkiProfileRegistry.get(name)`.
- `AttachmentSpec(leg_id: "L1".."L3"|"R1".."R3", ski_profile, attachment_compliance) -> MuJoCo XML 조각(body/geom/equality 또는 spring joint)` 생성 함수 — flybody MJCF에 병합 가능한 형태로 산출. N=2 확장 옵션에서는 `AttachmentSpec(reference_leg_id, secondary_leg_ids: [...], ...)` 형태로 그룹을 표현하되 secondary 다리는 강체 결합하지 않는다.
- `SkiPhysics.compute_turn_radius(profile, effective_edge_angle_deg, speed_mps) -> radius_m` (엣지각은 업무3에서 파생값으로 주입).

## 산출물
- 3~5개 `SkiProfile` 정의 파일.
- flybody MJCF에 스키를 부착하는 XML 생성/병합 스크립트. **N=6(다리당 1개, compliant)을 기본 구현으로 완성한 뒤**, N=2(기준 다리+슬라이딩 콜라이더) 확장 옵션을 추가하는 순서로 진행한다.
- 프로파일별 회전반경/침투깊이 벤치마크 스크립트.

## 구현 현황
`flyski_sim/ski_profiles.py`(5개 프로파일 + `compute_turn_radius`/`compute_penetration_cm`) + `ski_attachment.py`(N=6, claw마다 compliant ball 조인트로 미니 스키 부착) + `ski_test.py`로 구현·검증 완료 (2026-09-23, `python -m flyski_sim.ski_test`). Rigid weld가 아니라 `stiffness`/`damping`이 있는 MuJoCo `ball` 조인트로 붙였고, `attachment_compliance` 프로파일 값이 그 강성 배율로 들어간다.

**중요한 정정**: 처음 구현에서는 스키가 실제로 땅에 닿지 않는 상태로 "안정적"이라는 잘못된 결과가 나왔었다(부착 위치를 claw 원점 기준 로컬 -Z로 잘못 잡았고, 새 geom의 `contype`/`conaffinity`도 명시 안 해서 충돌 자체가 꺼져 있었음 — RESEARCH_NOTES.md 14번). claw 내부 collision capsule의 실제 발끝(`fromto` 끝점)을 읽어서 그 연장선에 붙이고 `contype=1,conaffinity=1`을 명시한 뒤 재검증했고, 아래 수용 기준은 **이 수정 이후의 결과**다(`physics.data.contact`에 `ski_geom_*`가 실제로 잡히는 것까지 확인함).

- [x] N=6 기본안에서 스키 부착 후 평지(zero-action, 사전학습 컨트롤러는 아직 없어 대체 검증)에서 200스텝 동안 크래시 없이 안정. (slalom/all_mountain/powder 3개 프로파일 모두 확인)
- [x] 좌/우 대칭 위치(T1~T3 left/right)의 스키 질량이 좌우 대칭으로 정확히 일치함을 확인.
- [x] 5개 프로파일이 정의되고 동일 엣지각(30°)·속도에서 회전반경이 전부 서로 다름(0.39~0.95cm).
- [x] 파우더 프로파일의 접지 면적이 최대이며, 동일 '물렁함' 조건에서 파우더가 슬라롬보다 덜 침투함(151.5 vs 505.1, 임의 단위) — 프로파일 간 트레이드오프가 실제로 방향성 있게 나타남.
- [x] 스키 부착 + 업무1 경사/모글 지형을 결합해도 150스텝 안정적으로 시뮬레이션됨.
- [ ] `attachment_compliance`를 강체에 가깝게 낮출 때 보행 불안정이 실제로 나타나는지 회귀 테스트(현재는 compliance=1.0/0.6만 테스트했고 극단값은 아직 안 해봄).
- [ ] N=2 확장 옵션(사람처럼 두 스키)은 구현하지 않음 — N=6가 기본안으로 충분히 동작하므로 우선순위 하향.
- [ ] flybody 사전학습 보행 컨트롤러(체크포인트, figshare 다운로드 필요— RESEARCH_NOTES.md 6번)로 실제 "걷기"까지는 아직 검증 못함, 지금은 zero-action 정적 안정성만 확인.

## 확장 아이디어 (선택)
- N=2("사람처럼 두 스키") 구성에서의 안정성/조종성 비교.
- 스키-스키 간 충돌(인접 다리의 미니 스키가 서로 부딪히는 상황) 처리.
