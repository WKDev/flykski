# 속도 명령과 몸 요각으로 쐐기(플루크) 자세를 고르는 스크립트 전문가: 모방학습 시범/기준선용
"""플루크 전문가(RESEARCH_NOTES 38번).

- 속도 명령 -> 쐐기 크기(WEDGE_BY_CMD), 안쪽 엣지는 항상 EDGE_DEG. "빠르게"도 판을 완전히
  평평하게 두지 않는다(쐐기 2°, 엣지 12°). 평평하면 판이 진행 방향으로 정렬되려는 힘이
  약해서 몸이 돌며 코스를 벗어났다.
- 몸 요각 -> 좌우 쐐기 차(P 제어, KP도/rad). 왼쪽으로 돌았으면 왼쪽 판을 더 모아 오른쪽으로.
- IK(ski_stance.solve_plate_pose)는 느려서 (명령, 쐐기 차) 격자로 미리 풀어 두고 보간한다.
SpeedControlEnv에서 3판 모두 완주, 속도 오차 1.4~1.7cm/s(KP=20).
"""
from __future__ import annotations

import numpy as np

from flyski_sim.ski_stance import solve_plate_pose

WEDGE_BY_CMD = {0.: 18., 5.: 12., 10.: 7., 15.: 2.}
EDGE_DEG = 12.
DIFFS = np.linspace(-8., 8., 9)
KP = 20.


def build_table(env) -> dict:
    """{(명령, 쐐기 차): 정책 액션(42)}. env는 SkiCourseEnv 계열(액션 = 스탠스 오프셋/scale)."""
    table = {}
    for cmd, wedge in WEDGE_BY_CMD.items():
        for d in DIFFS:
            pose, _ = solve_plate_pose(env.task.walker, {'left': -(wedge + d), 'right': wedge - d},
                                       {'left': EDGE_DEG, 'right': -EDGE_DEG})
            off = np.array([pose[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)])
            table[(cmd, float(d))] = np.clip(off / env._action_scale, -1., 1.)
    return table


class SnowplowExpert:
    def __init__(self, env, table: dict | None = None, kp: float = KP):
        self.env = env
        self.table = table if table is not None else build_table(env)
        self.kp = kp

    def __call__(self, obs=None) -> np.ndarray:
        p = self.env.env.physics
        R = p.data.xmat[self.env._th].reshape(3, 3)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        d = float(np.clip(self.kp * yaw, DIFFS[0], DIFFS[-1]))
        i = int(np.clip(np.searchsorted(DIFFS, d), 1, len(DIFFS) - 1))
        a0, a1 = float(DIFFS[i - 1]), float(DIFFS[i])
        t = (d - a0) / (a1 - a0)
        cmd = min(WEDGE_BY_CMD, key=lambda c: abs(c - self.env._cmd))
        return (1 - t) * self.table[(cmd, a0)] + t * self.table[(cmd, a1)]


# ---- 조향(S자 추종) 전문가 ----
# 조향 입력 u(-1~+1, + = 왼쪽으로 돌기). 판은 11자(쐐기 0) 그대로 두고, 바깥 판(u>0이면
# 오른쪽)만 안쪽 엣지를 걸고 안쪽 판 엣지는 풀며, 무게중심을 바깥 판 쪽으로 옮긴다
# (스키딩 패럴렐 턴에 가까움). 쐐기 0에서 이렇게 하면 사람 스키와 같은 방향으로 돈다(요
# +-42~53°). 바깥 판을 벌리는(슈템) 동작을 더하면 벌린 판이 브레이크처럼 몸을 그쪽으로
# 돌려 방향이 뒤집혔다: 판이 몸에 단단히 묶여 있고 요 관성이 아주 작은 초파리 스케일
# 특성(RESEARCH_NOTES 39번).
STEER_BASE_WEDGE = 0.
STEER_COM_CM = 0.06            # 몸을 바깥 판 쪽으로 옮기는 양(cm, 판 사이 반폭 ~0.13cm).
STEER_EDGE_DEG = 12.           # 바깥 판 안쪽 엣지각(직진일 땐 양쪽 다 이 값).
STEER_US = np.linspace(-1., 1., 9)
STEER_SMOOTH = 0.2              # 1차 저역 필터 계수(스텝당).
LOOKAHEAD_CM = 4.               # 선행 주시 거리(경로 호 길이, cm). 0이면 가장 가까운 점.
K_PSI, K_Y, K_R = 2.5, 0.3, 0.3  # u = K_PSI * 방향 오차(rad) + K_Y * 옆 오차(cm) - K_R * 요레이트(rad/s)


def build_steer_table(env) -> dict:
    table = {}
    for u in STEER_US:
        e_r = STEER_EDGE_DEG * (1. if u >= 0 else 1. + u)  # u>0: 오른쪽 판이 바깥 -> 엣지 유지.
        e_l = STEER_EDGE_DEG * (1. if u <= 0 else 1. - u)
        com_y = -STEER_COM_CM * u                       # u>0: 몸을 오른쪽(-y) 판 위로.
        shift = -np.array([0., com_y, 0.])              # 판을 -d로 = 몸을 +d로.
        pose, _ = solve_plate_pose(env.task.walker, {'left': -STEER_BASE_WEDGE, 'right': STEER_BASE_WEDGE},
                                   {'left': e_l, 'right': -e_r}, shift_cm={'left': shift, 'right': shift})
        off = np.array([pose[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)])
        table[float(u)] = np.clip(off / env._action_scale, -1., 1.)
    return table


class SteerExpert:
    """S자 목표 궤적 추종: 방향/옆 오차 -> 조향 입력 u -> (엣지, 무게중심) 자세 보간."""

    def __init__(self, env, table: dict | None = None):
        self.env = env
        self.table = table if table is not None else build_steer_table(env)
        self.last_u = 0.

    def __call__(self, obs=None) -> np.ndarray:
        e_psi, e_y = self.env._errors(self.env.env.physics)
        ref = getattr(self.env, 'reference', None)
        if LOOKAHEAD_CM > 0 and hasattr(ref, 'psi'):
            # 선행 주시: 몇 cm 앞 경로의 진행 방향을 목표로(대회전에서 턴을 늦게 끊어 오르막으로
            # 100° 넘게 돌다 멈추던 오버슈트 방지, 42번).
            i = min(ref._i + int(LOOKAHEAD_CM / 0.05), len(ref.psi) - 1)
            p = self.env.env.physics
            v = p.data.qvel[:2]
            head = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 3. else self.env._yaw(p)
            e_psi = float(np.angle(np.exp(1j * (ref.psi[i] - head))))
        p = self.env.env.physics
        yaw_rate = float(p.data.cvel[self.env._th][2])
        u = float(np.clip(K_PSI * e_psi + K_Y * e_y - K_R * yaw_rate, -1., 1.))
        if self.env._stats['steps'] < 20:         # 착지 0.2초 동안은 조향하지 않음.
            u = 0.
        u = self.last_u + STEER_SMOOTH * (u - self.last_u)   # 조향 떨림 저역 필터.
        self.last_u = u
        i = int(np.clip(np.searchsorted(STEER_US, u), 1, len(STEER_US) - 1))
        a0, a1 = float(STEER_US[i - 1]), float(STEER_US[i])
        t = (u - a0) / (a1 - a0)
        return (1 - t) * self.table[a0] + t * self.table[a1]


# ---- 패럴렐 턴 전문가 ----
# 두 판을 11자로 두고 같은 쪽 엣지로 동시에 세운다(바깥 판은 안쪽 엣지, 안쪽 판은 바깥
# 엣지). 무게중심은 바깥 판 쪽으로. 쐐기 0에서 두 판을 오른쪽 엣지로 세우면 우회전
# (요 -35°), 왼쪽 엣지면 좌회전(+54°), 사람과 같은 방향(RESEARCH_NOTES 40번).
PARALLEL_EDGE_DEG = 15.
PARALLEL_COM_CM = 0.04


def build_parallel_table(env) -> dict:
    table = {}
    for u in STEER_US:
        roll = -PARALLEL_EDGE_DEG * u                   # u>0(좌회전): 두 판 왼쪽 엣지(롤 -).
        shift = -np.array([0., -PARALLEL_COM_CM * u, 0.])   # u>0: 몸을 바깥(오른쪽) 판 쪽으로.
        pose, _ = solve_plate_pose(env.task.walker, {'left': 0., 'right': 0.},
                                   {'left': roll, 'right': roll}, shift_cm={'left': shift, 'right': shift})
        off = np.array([pose[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)])
        table[float(u)] = np.clip(off / env._action_scale, -1., 1.)
    return table


# 멈춤 복구(RESEARCH_NOTES 43번): 폴라인과 직각으로 산쪽 엣지에 걸려 멈추면, 사람처럼 엣지를
# 풀고 판을 폴라인 쪽으로 틀고(피벗) 몸을 골짜기 쪽으로 던진다. recovery_test에서 이 조합만
# 30cm/s까지 다시 출발했다(엣지만 풀면 그대로 멈춤).
RECOVER_SPEED = 2.              # 이 속력 아래 + 몸 방향이 폴라인에서 RECOVER_YAW_DEG 넘게 벗어나면 복구.
RECOVER_YAW_DEG = 60.
RECOVER_EXIT_SPEED = 6.
RECOVER_PIVOT_DEG, RECOVER_EDGE_DEG, RECOVER_COM_CM = 20., 5., 0.04


def build_recover_actions(env) -> dict:
    """{+1: 몸이 왼쪽(+요)을 향해 멈췄을 때, -1: 오른쪽}. 폴라인은 +1이면 몸의 오른쪽."""
    out = {}
    for side in (+1., -1.):
        yaw = -side * RECOVER_PIVOT_DEG                 # 팁을 폴라인(골짜기) 쪽으로.
        roll = side * RECOVER_EDGE_DEG                  # 골짜기 쪽 엣지.
        shift = -np.array([0., -side * RECOVER_COM_CM, 0.])   # 몸을 골짜기 쪽으로.
        pose, _ = solve_plate_pose(env.task.walker, {'left': yaw, 'right': yaw}, {'left': roll, 'right': roll},
                                   shift_cm={'left': shift, 'right': shift})
        off = np.array([pose[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)])
        out[side] = np.clip(off / env._action_scale, -1., 1.)
    return out


class ParallelExpert(SteerExpert):
    """SteerExpert와 같은 되먹임, 자세 표만 패럴렐(두 판 같은 쪽 엣지). 멈추면 복구 모드."""

    def __init__(self, env, table: dict | None = None, recover: dict | None = None):
        super().__init__(env, table if table is not None else build_parallel_table(env))
        self.recover = recover if recover is not None else build_recover_actions(env)
        self.recovering = 0.

    def __call__(self, obs=None) -> np.ndarray:
        p = self.env.env.physics
        speed = float(np.linalg.norm(p.data.qvel[:2]))
        yaw = self.env._yaw(p)
        if self.recovering:
            if speed > RECOVER_EXIT_SPEED or abs(yaw) < np.deg2rad(30.):
                self.recovering = 0.
        elif (self.env._stats['steps'] > 20 and speed < RECOVER_SPEED
              and abs(yaw) > np.deg2rad(RECOVER_YAW_DEG)):
            self.recovering = float(np.sign(yaw))
        if self.recovering:
            return self.recover[self.recovering]
        return super().__call__(obs)


def expert_for(stage: str):
    """단계별 (표 생성 함수, 전문가 클래스)."""
    return {'speed': (build_table, SnowplowExpert), 'turn': (build_steer_table, SteerExpert),
            'parallel': (build_parallel_table, ParallelExpert)}[stage]
