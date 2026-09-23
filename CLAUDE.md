# flykski 프로젝트 노트

초파리 male CNS 커넥톰(natverse/malecns) + flybody(MuJoCo)로 "초파리 스키 시뮬레이터"를 만드는 프로젝트.

## 작업 전 반드시 확인할 문서
- **[RESEARCH_NOTES.md](RESEARCH_NOTES.md)**: 지금까지 겪은 삽질/트러블슈팅 전체 기록(설치 문제, 잘못된 가정, 교차검증으로 잡아낸 설계 결함 등). **같은 문제를 다시 겪기 전에 먼저 여기서 검색해볼 것.**
- **[ENVIRONMENT.md](ENVIRONMENT.md)**: flybody/malecns 개발 환경 세팅 방법, 실행 커맨드, 알려진 이슈, "다음에 할 일" 체크리스트.
- **[prompts/](prompts/)**: 업무 단위별 설계 프롬프트(00 마스터 ~ 05 학습/평가). v2로 flybody 기반 재설계 완료, codex 교차검증 반영 완료.

## 핵심 결정 사항 (요약, 근거는 prompts/ 및 RESEARCH_NOTES.md 참고)
- 몸체는 처음부터 만들지 않고 [flybody](https://github.com/TuragaLab/flybody/)(DeepMind×Janelia, MuJoCo)를 재사용한다.
- 스키는 **좌/우 휘는 플레이트 2개(N=2, `ski_plate.py`)**가 기본안이다(2026-09-23 사용자 승인으로 변경) — 한쪽 다리 3개를 *탄성*(T2 스프링 ball + T1/T3 soft connect)으로 묶는다. 과거에 폐기된 건 rigid weld(과구속)였다. 다리당 미니스키 N=6은 `ski_layout='per_leg'` 레거시. 스키-눈 마찰은 `SnowParams.ski_friction`(priority=1)로 따로 둔다(RESEARCH_NOTES 32번).
- 액션 공간은 flybody 네이티브 관절 액추에이터이며 차원을 하드코딩하지 않는다(태스크마다 다름, 예: 보행=59).
- 목표 궤적은 사람이 정의하지 않고, 슬로프에 순차 점화되는 냄새 게이트(슬라롬 게이트 역할)를 배치해 초파리의 실제 후각 내비게이션이 S자를 만들게 한다.
- MaleCNS 위상은 고정하고 뉴런별 소수 파라미터만 학습한다(FlyGM 논문 레시피 참고, 단 그 논문의 결과가 이 프로젝트에도 재현된다고 가정하지 않고 검증 대상으로 취급).

## 환경
- GitHub: https://github.com/WKDev/flykski (공개). flybody는 git 서브모듈(`d015e9b`). 새 PC 설치는 README.md "설치" 절, 검증은 `python setup/verify.py`.
- flybody: conda env `flybody` (Python 3.10), `flybody/`에 소스 클론됨.
- malecns: conda env `malecns` (R 4.5.3, conda-forge). `NEUPRINT_TOKEN`은 `.env`와 `C:\Users\chson\.Renviron`에 설정 완료.
- 정확한 실행 커맨드(PATH 설정 등)는 ENVIRONMENT.md 참고.

## 진행 방식
1. flybody 사전학습 컨트롤러 평지 보행 확인 — 완료.
2. malecns neuprint 인증 및 핵심 뉴런(DNg100/DNb08) 존재 확인 — 완료.
3. 업무1(지형) — **거의 완료**(경사+모글/알파인, max_slope_deg 자동보정, 설질 마찰 구간, 냄새 게이트까지 전부 구현·검증됨). 남은 건 2단계(무한 스트리밍)뿐이고 의도적으로 미룸.
4. 업무2(스키 프로파일 5종)/업무3(N=6 compliant 부착 + 파생 텔레메트리) — **핵심 부분 완료**. 도중에 "스키가 실은 땅에 안 닿고 있었다"(RESEARCH_NOTES 14번)와 "엣지각이 다리 로컬 프레임 문제로 44°씩 잘못 나왔다"(15번) 두 개의 진짜 버그를 발견·수정함 — **크래시 안 남 ≠ 제대로 동작함**을 몸소 겪음, 새 부착물 만들 때는 꼭 `physics.data.contact`로 직접 확인할 것.
5. 업무4 R 추출 스크립트 완료(`flyski_sim/connectome/extract_leg_cpg_circuit.R`): DNg100/DNb08 → 인터뉴런(억제성 IN09A002/IN12B003, 흥분성 IN01A015/IN17A001/dPR1) → **실제 다리 운동뉴런 73종 360개**까지 이어지는 전체 경로를 male-cns:v1.0에서 실증(연결 데이터는 `flyski_sim/connectome/*.csv`). 원 논문과 정확한 대조는 아직 미완이지만 정황증거는 강함.
6. `xfrc_applied` 침투 보조력(설질별 속도비례 항력) — **완료**. 계수 스케일을 처음에 잘못 잡아 바로 발산했다가 재조정(RESEARCH_NOTES 16번).
7. 시각 검증: `flyski_sim/render_demo.py`로 PNG/GIF 렌더링 확인됨(`renders/` 폴더). Unity/Unreal 안 쓰고 MuJoCo 자체 오프스크린 렌더러 사용 — 이 환경에서 정상 동작.
8. `ConnectomeController`(tanh rate-model) 구현 — 지속구동 시 리듬 없음(고정점 수렴, RESEARCH_NOTES 18번). → **`LIFConnectomeController`(스파이킹+불응기)로 교체하니 실제 리듬 발생 확인**(19번, `weight_scale=30, v_thresh=0.5, tau_mem=8.0`). real/shuffled가 random보다 10배 이상 강하게 진동 — 단 real vs shuffled 우열은 아직 불명확, 업무5에서 통계검정 필요.
9. **커넥톰 리듬 → flybody 관절 통합 완료**: `motor_decoder.py`(다리근 이름→액추에이터 키워드 매핑) + `connectome_to_flybody_test.py`. `femur_T1_left` 관절이 실제로 진동(std=0.079) — 커넥톰에서 나온 스파이킹 리듬이 진짜로 다리를 움직이는 것까지 확인. 중요 한계: `vnc_motor`엔 비행근도 섞여있어 이름으로 다리근 18종만 골라 씀, 좌우(side) 정보 없어서 좌우 대칭 신호(RESEARCH_NOTES 20번).
10. **협응된 보행인지 검증함 — 아직 아니다.** 다리 간 위상 상관이 전부 거의 0(삼각보행 기대치 |0.7~0.9|와 거리가 멂), thorax는 초반에 낮은 자세로 주저앉은 뒤 그 자세에서 다리만 흔듦(계속 쓰러지는 건 아님). 예상된 결과(원본 위상+학습 안 된 파라미터, RESEARCH_NOTES 21번) — **업무5 학습 루프(파라미터 최적화)가 다음 최우선순위**.
11. **업무5 첫 학습 루프 — 실제로 삼각보행 위상을 만들어냄.** `train_connectome_params.py`(전역 스칼라 4개, (1+λ) ES + elitism). 1차 시도(elitism 없음)는 fitness가 1세대 이후 정체(22번) → elitism+sigma decay로 고치니 **fitness -0.27→2.14로 꾸준히 개선**, 실측 위상 상관 `T1L-T3L=+0.816`(강한 양의 상관, 삼각보행 기대와 정확히 일치), `T1L-T2L=-0.341`, `T1L-T1R=-0.17`(둘 다 기대 방향)까지 확인(23번). 이 프로젝트 핵심 가설("위상 고정+소수 파라미터 학습으로 쓸만한 행동이 나오는가")에 대한 가장 직접적인 긍정적 증거.
12. **real vs shuffled vs random 첫 비교(n=1 시드)**: 완전히 동일한 ES 절차로 위상만 바꿔 3번 실행 — **real=2.14 > shuffled=1.29 > random=0.64**(random은 8세대 내내 값이 고정, "침묵 네트워크"에 elitism이 갇힘). 이 프로젝트 핵심 가설과 일치하는 방향! **단 시드 1개뿐이라 통계적 결론은 아니다** — 05번 문서가 요구하는 5회 이상 반복+통계검정이 필요(RESEARCH_NOTES 24번, 확인편향 경계).
13. **후각 신경군 추출 완료**: DA1(페로몬) PN → 하행뉴런(DNpe052 등) → 다리 CPG 명령 뉴런(DNg100/DNb08)까지 단 2홉 직접 연결 확인(`extract_olfactory_pathway.R`). "냄새→조향"과 "냄새→걷기명령"이 뉴런 수준에서 거의 같은 병목 공유. 좌우(side) 정보 없어서 좌우 비대칭 조향 가능 여부는 미확인 — **좌우 정보 누락이 세 번째(14/20/25번) 반복된 실수**, 다음 추출부턴 `somaSide` 기본 포함할 것.
14. **real/shuffled/random 5회 반복 통계검정 완료 — 중요한 음성 결과.** real=1.08±0.69, shuffled=0.92±0.46, random=0.92±0.62(평균±표준편차), Mann-Whitney p=0.69~1.00 전부 → **통계적으로 유의미한 차이 없음**. n=1 예비 결과(real이 제일 좋아 보였던 것, real=2.14>shuffled=1.29>random=0.64)는 노이즈였다 — 24번에서 미리 경계해뒀던 확인편향 함정이 실제로 일어났고, 제대로 검증해서 피했다. **이 세션에서 가장 중요한 발견 중 하나**(RESEARCH_NOTES 26번). 단, 전역 스칼라 4개짜리 좁은 실험이라 "위상이 안 중요하다"고 성급히 결론 내리면 안 됨 — 뉴런별 파라미터로 재검증 필요.
15. **역할군 파라미터(10차원) 재검증 완료 — 가설을 뒷받침 못했고 순위가 반대로 나옴.** random(1.84) > real(1.15) > shuffled(0.91), 전부 통계적으로 비유의(p>0.06). real이 "침묵 네트워크"(발화 안 함) 실패 모드에 가장 자주(60%) 빠짐 — random(20%)보다 오히려 나쁨. **두 번의 독립 실험(전역스칼라 26번, 역할군 27번) 모두 "real이 유리하다"는 증거를 못 찾았다**(오히려 이번엔 방향이 반대). 초기값이 조건별로 공정하지 않았을 가능성 등 실험 설계 자체를 의심해볼 필요가 있음 — 다음 사람/세션이 판단할 열린 질문(RESEARCH_NOTES 27번).
16. **좌우(side) 정보 보강 완료** — 14/20/25번에서 세 번 지적만 하고 미뤘던 걸 드디어 고침. 기존 CSV 4개에 `somaSide`/`rootSide` 추가, `motor_decoder.py`가 이제 실제로 좌/우 액추에이터를 구분해서 구동(검증됨: 왼쪽에만 활성 주입 → 액션 [1,0] 정확히 나옴). 통합 테스트 재확인(400스텝 안정, 관절 움직임 유지). RESEARCH_NOTES 28번.
17. 다음(같은 실험을 더 변형해서 반복하기보다, 다른 축으로): (a) 완전 뉴런별 파라미터 확장 또는 실험 설계 재점검(조건별 별도 초기화 등 — 판단 필요, 사용자에게 알림 보냄), (b) 좌우 side가 이제 있으니 후각 좌우 비대칭 조향 실제로 시도, (c) 냄새 게이트/세션 프레임워크. 업무1~3은 사실상 다 됨.
18. **(2026-09-23 후반) 경사 지형 치명적 버그 3종 발견·수정**(RESEARCH_NOTES 30번): 초파리가 지형 속 10cm 아래에 파묻혀 스폰(composer가 task 훅을 arena보다 먼저 부름), hfield를 elevation_z=1에 cm 높이로 써서 z>1 충돌 불가, 설질 항력이 토크 칸에 들어감. **이전 경사 지형 결과는 전부 무효**(평지 커넥톰 학습은 영향 없음). 실시간 뷰어(`live_view.py`) 검은 화면 = `physics.contexts` 접근이 GL 컨텍스트를 만들어 충돌.
19. **발화율 보정 + 침묵 페널티 재비교**(31번): 침묵 실패 0%로 제거했지만 shuffled 1.60 > real 1.24 > random 1.13, 전부 비유의 — 세 번째 독립 음성 결과. 다음은 29번 ② 적합도 개선.
20. **스키 재설계 N=2 휘는 플레이트 + 스키가 사면에서 안 미끄러지던 문제 수정**(32번): 이제 20° 사면에서 스키만 접지한 채 2초에 ~30cm 활강. 알파인은 선 채로, 모글은 zero action이면 전복(컨트롤러 과제).
21. 새 세션에서 이어갈 때 실행 순서(flykski 루트에서, python.exe는 `C:/Users/chson/miniconda3/envs/flybody/python.exe`):
   - `python setup/verify.py` — 전체 검사 한 번에(지형/스키/텔레메트리/항력/커넥톰 통합).
   - `python -m flyski_sim.live_view` — 실시간 3D 창(모글, zero action 활강).
   - 비교 실험 결과: `topology_comparison_results.csv`(전역스칼라, 26번), `topology_comparison_role_results.csv`(역할군, 27번), `topology_comparison_calibrated_results.csv`(발화율 보정, 31번) — 전부 유의차 없음.
   - `python -m flyski_sim.smoke_test` — 업무1(지형/설질/냄새) 검증.
   - `python -m flyski_sim.ski_test` — 업무2/3(스키 프로파일/부착) 검증.
   - `python -m flyski_sim.telemetry_test` — 업무3(파생 텔레메트리) 검증.
   - `python -m flyski_sim.drag_test` — 업무3(설질 항력) 검증.
   - `python -m flyski_sim.render_demo` — 시각 검증용 PNG/GIF 생성(`renders/`).
   - `python -m flyski_sim.connectome_controller_lif_test` — 업무4(LIF 리듬) 검증.
   - `python -m flyski_sim.connectome_to_flybody_test` — 업무4(커넥톰→flybody 관절) 통합 검증.
   - `python -m flyski_sim.train_connectome_params` — 업무5(진화전략 학습 루프) 실행.
   - `python -m flyski_sim.connectome_controller_test` — 업무4(ConnectomeController) 검증.
