# 명령한 판 엣지각(롤)이 실제로 나오는지, 주행 중에도 유지되는지, 엣지 방향대로 도는지 측정하는 검사
"""엣지 제어 검사(RESEARCH_NOTES 42번).

1) 평지 정지 / 15° 사면 주행(초기 8cm/s)에서 두 판에 같은 롤 r(IK, 스탠스 대비)을 명령하고
   실제 판 롤(지면 법선 기준, + = 오른쪽 엣지)의 평균과 요 변화를 잰다.
2) 패럴렐 전문가로 대회전 경로를 달리며 명령 롤과 실제 판 롤의 상관/비율/지연을 잰다.

    python -m flyski_sim.edge_tracking_test
"""
from __future__ import annotations

import warnings

import numpy as np

from flyski_sim import ski_stance
from flyski_sim.rl_task import ParallelTrackEnv
from flyski_sim.ski_stance import solve_plate_pose

ROLLS = (-20., -10., 0., 10., 20.)


def plate_rolls(env):
    p = env.env.physics
    n = env.task._slope_n
    out = []
    for b in env._roots:
        R = p.data.xmat[b].reshape(3, 3)
        out.append(float(np.degrees(np.arctan2(R[:, 1] @ n, R[:, 2] @ n))))
    return out


def open_loop(env, roll, moving):
    pose, _ = solve_plate_pose(env.task.walker, {'left': 0., 'right': 0.}, {'left': roll, 'right': roll})
    a = np.clip(np.array([pose[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)]), -1, 1)
    env.reset(seed=0)
    p = env.env.physics
    if not moving:
        p.data.qvel[:] = 0.
    rolls, yaws = [], []
    for k in range(80):
        full = np.zeros(env._nu)
        full[env._act_idx] = env._action_scale * a * min(k / 15, 1.)
        env.env.step(full)
        if not moving:
            p.data.qvel[:3] = 0.                        # 정지 상태 유지(미끄러짐 제거).
        if k >= 40:
            rolls.append(plate_rolls(env))
        R = p.data.xmat[env._th].reshape(3, 3)
        yaws.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    r = np.mean(rolls, axis=0)
    return r, yaws[-1] - yaws[15]


def main():
    warnings.filterwarnings('ignore')
    env = ParallelTrackEnv(seed=0)
    print(f'stance width x{ski_stance.STANCE_WIDTH_SCALE}')
    for moving in (False, True):
        print('moving on slope' if moving else 'held still on slope')
        for roll in ROLLS:
            (l, r), dyaw = open_loop(env, roll, moving)
            print(f'  command roll {roll:+5.0f}: actual L {l:+6.1f} R {r:+6.1f}'
                  + (f', yaw change {dyaw:+6.1f} (+ = left)' if moving else ''), flush=True)
    from flyski_sim.experts import PARALLEL_EDGE_DEG, ParallelExpert
    pol = ParallelExpert(env)
    obs, _ = env.reset(seed=1)
    cmd, act = [], []
    for k in range(600):
        obs, r, term, trunc, info = env.step(pol(obs))
        cmd.append(-PARALLEL_EDGE_DEG * pol.last_u)
        act.append(np.mean(plate_rolls(env)))
        if term or trunc:
            break
    cmd, act = np.array(cmd), np.array(act)
    lags = range(0, 30)
    cor = [np.corrcoef(cmd[:len(cmd) - L], act[L:])[0, 1] for L in lags]
    best = int(np.argmax(cor))
    slope = np.polyfit(cmd[:len(cmd) - best], act[best:], 1)[0]
    print(f'parallel expert closed loop ({len(cmd) * 0.01:.1f}s, end {info.get("reason", "running")}): command roll range '
          f'{cmd.min():+.0f}..{cmd.max():+.0f}, actual {act.min():+.1f}..{act.max():+.1f}, best corr {cor[best]:.2f} '
          f'at lag {best * 10}ms, actual/command gain {slope:.2f}')


if __name__ == '__main__':
    main()
