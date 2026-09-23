# 쐐기 자세에서 무게중심을 한쪽 판으로 옮기면 반대쪽으로 도는지(플루크 보겐) 측정하는 물리 검사
"""플루크 보겐 물리 검사(RESEARCH_NOTES 38번).

쐐기(snowplow_pose) + 몸을 좌/우로 옮긴 자세를 고정 액션으로 걸고 15° 사면에서 요
변화를 잰다. 왼쪽 턴은 바깥(오른쪽) 스키에 하중 = 몸을 오른쪽(-y)으로. 컨트롤러 없음.
모델에 좌향 드리프트가 있어서(38번) 이동 0과의 차이로 판정한다.

    python -m flyski_sim.pflugbogen_test
"""
from __future__ import annotations

import mujoco
import numpy as np

from flyski_sim import snowplow_test
from flyski_sim.ski_stance import snowplow_pose

WEDGE, EDGE = 15., 12.


def run(com_y_cm, seconds=2.0, ramp=0.2):
    env, task = snowplow_test.build()
    env.reset()
    p = env.physics
    names = env.action_spec().name.split('\t')
    pose, err = snowplow_pose(task.walker, WEDGE, EDGE, (0., com_y_cm, 0.))
    off = np.array([pose[n] - task._stance[n] if n in pose else 0. for n in names])
    th = p.model.name2id('walker/thorax', 'body')
    ski_side = {p.model.name2id(g.full_identifier, 'geom'): s
                for s, u in task.skis.items() for g in u.geoms}
    f6 = np.zeros(6)
    yaws, share, ys = [], [], []
    dt = env.control_timestep()
    for k in range(int(seconds / dt)):
        env.step(off * min(k * dt / ramp, 1.))
        R = p.data.xmat[th].reshape(3, 3)
        yaws.append(np.arctan2(R[1, 0], R[0, 0]))
        ys.append(float(p.data.xpos[th][1]))
        if k * dt > ramp:
            # 판별 하중 = 그 판 조각들이 설면에서 받는 수직력 합(발톱은 판과 등식 구속이라
            # cfrc_ext로는 판 하중을 못 잰다).
            f = {'left': 0., 'right': 0.}
            for i in range(p.data.ncon):
                c = p.data.contact[i]
                g = c.geom1 if c.geom1 in ski_side else c.geom2 if c.geom2 in ski_side else None
                if g is None:
                    continue
                mujoco.mj_contactForce(p.model.ptr, p.data.ptr, i, f6)
                f[ski_side[g]] += max(f6[0], 0.)
            if f['left'] + f['right'] > 0:
                share.append(f['right'] / (f['left'] + f['right']))
    yaws = np.degrees(np.unwrap(yaws))
    v = p.data.qvel[:2]
    return dict(com_y_cm=com_y_cm, ik_tip_err_max=round(max(v for k, v in err.items() if not k.endswith('ori')), 4),
                yaw_change_deg=round(float(yaws[-1] - yaws[int(ramp / dt)]), 1),
                lateral_cm=round(ys[-1] - ys[int(ramp / dt)], 2),          # + = 왼쪽으로 이동.
                vel_heading_deg=round(float(np.degrees(np.arctan2(v[1], v[0]))), 1),
                right_ski_load=round(float(np.mean(share)), 2),
                speed_end=round(float(np.linalg.norm(p.data.qvel[:2])), 1),
                upright=bool(p.data.xmat[th].reshape(3, 3)[2, 2] > 0.5))


def main():
    rows = [run(y) for y in (0., -0.02, -0.04, 0.02, 0.04)]
    for r in rows:
        print(r, flush=True)
    # 판정은 몸 방향(요)이 아니라 실제 경로(옆 이동)로. 한쪽 판에 하중이 실리면 그 판의
    # 그립이 경로를 판이 향한 쪽으로 휘게 하는 동시에 그쪽을 브레이크처럼 잡아 몸을 반대로
    # 돌리는 모멘트도 만들어서, 요만 보면 판정이 뒤집힐 수 있다.
    base = rows[0]['lateral_cm']
    left = [round(r['lateral_cm'] - base, 2) for r in rows if r['com_y_cm'] < 0]   # 몸 오른쪽 -> 왼쪽(+).
    right = [round(r['lateral_cm'] - base, 2) for r in rows if r['com_y_cm'] > 0]
    ok = all(v > 0.3 for v in left) and all(v < -0.3 for v in right)
    print(f'이동 0 대비 옆 이동(cm): 몸 오른쪽 {left}, 몸 왼쪽 {right} -> 바깥 스키 하중으로 반대쪽 턴? {ok}')
    return ok


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
