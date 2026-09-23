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
