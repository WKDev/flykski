# 폴라인과 직각으로 멈춘 상태에서 엣지 해제/판 피벗/하중 이동으로 다시 출발해 폴라인으로 돌아오는지 측정
"""멈춤 복구 검사(RESEARCH_NOTES 43번).

몸 방향 +90°(왼쪽을 향해 사면을 가로지름)로 세워 산쪽 엣지(오른쪽 엣지, 롤 +)를 건 채
멈춘 상태를 만든 뒤, 후보 복구 자세를 걸고 2초 동안 속력과 몸 방향(0 = 폴라인)을 본다.
몸이 +y를 향하면 폴라인(+x, 골짜기)은 몸의 오른쪽이다. 판 요 -는 팁을 오른쪽(골짜기)으로.

    python -m flyski_sim.recovery_test
"""
from __future__ import annotations

import warnings

import numpy as np

from flyski_sim.rl_task import ParallelTrackEnv
from flyski_sim.ski_stance import solve_plate_pose

YAW0 = np.deg2rad(90.)


def pose(env, yaw=(0., 0.), roll=(0., 0.), shift=(0., 0., 0.)):
    sh = np.array(shift)
    q, _ = solve_plate_pose(env.task.walker, {'left': yaw[0], 'right': yaw[1]},
                            {'left': roll[0], 'right': roll[1]}, shift_cm={'left': -sh, 'right': -sh})
    return np.clip(np.array([q[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)]), -1, 1)


def stuck_then(env, recover, hold=0.5, seconds=2.0):
    """산쪽 엣지로 멈춘 상태를 만든 뒤 recover 자세를 건다."""
    p = env.env.physics
    env.reset(seed=0)
    pos, _ = env.task.walker.get_pose(p)
    slope = np.deg2rad(20.)
    # 사면 기울기 + 몸 요 90°.
    qz = np.array([np.cos(YAW0 / 2), 0, 0, np.sin(YAW0 / 2)])
    qy = np.array([np.cos(slope / 2), 0, np.sin(slope / 2), 0])
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qz
    q = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                  w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
    env.task.walker.set_pose(p, position=pos, quaternion=q)
    p.data.qvel[:] = 0.
    stuck = pose(env, roll=(12., 12.))               # 산쪽(오른쪽... 몸 기준 왼쪽이 산) 엣지.
    stuck = pose(env, roll=(-12., -12.))             # 몸이 +y를 향하면 산쪽 = 몸의 왼쪽 = 롤 -.
    yaws, speeds = [], []
    for k in range(int((hold + seconds) / 0.01)):
        a = stuck if k * 0.01 < hold else recover
        full = np.zeros(env._nu)
        full[env._act_idx] = env._action_scale * a
        env.env.step(full)
        R = p.data.xmat[env._th].reshape(3, 3)
        yaws.append(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        speeds.append(float(np.linalg.norm(p.data.qvel[:2])))
    h = int(hold / 0.01)
    return speeds[h - 1], yaws[h - 1], max(speeds[h:]), yaws[-1], speeds[-1]


def main():
    warnings.filterwarnings('ignore')
    env = ParallelTrackEnv(seed=0)
    cands = {
        'keep uphill edge (stuck)': pose(env, roll=(-12., -12.)),
        'release edges (flat)': pose(env),
        'release + pivot plates 20deg to fall line': pose(env, yaw=(-20., -20.)),
        'release + pivot 30': pose(env, yaw=(-30., -30.)),
        'downhill edge 8 + COM downhill 0.04': pose(env, roll=(8., 8.), shift=(0., -0.04, 0.)),
        'pivot 20 + downhill edge 5 + COM downhill': pose(env, yaw=(-20., -20.), roll=(5., 5.), shift=(0., -0.04, 0.)),
        'pivot 20 + COM forward 0.03': pose(env, yaw=(-20., -20.), shift=(0.03, 0., 0.)),
    }
    for name, a in cands.items():
        v0, y0, vmax, y1, v1 = stuck_then(env, a)
        print(f'{name:44s}: stuck speed {v0:4.1f} yaw {y0:+5.0f} -> max speed {vmax:5.1f}, end yaw {y1:+6.1f} '
              f'(0 = fall line), end speed {v1:5.1f}', flush=True)


if __name__ == '__main__':
    main()
