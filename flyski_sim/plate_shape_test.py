# 스키 플레이트(힌지 체인)의 굽힘 모양이 볼록한지(지그재그 없는지)와 하중에 따른 휨을 측정하는 검사
"""판 모양 검사(RESEARCH_NOTES 37번).

판은 7조각 + 굽힘 힌지 6개 체인이라 힌지가 제각각 꺾이면 실제 스키에는 없는 지그재그
(비볼록) 모양이 나올 수 있다. 굽힘 힌지 각을 캠버 기준각(springref) 대비로 보고,
- zigzag_frac: |각| > 0.05°인 힌지들 사이에서 부호가 바뀌는 스텝의 비율(0이 이상적),
- sag_um: 팁-테일 기준선 대비 판 중앙 처짐(um, + = 가운데가 판 아래쪽으로 = 역캠버),
- bend_curv: 굽힘 힌지 각 합 / 판 길이(1/cm).
을 세 상황(평지 스탠스, 15° 사면 플루크 20°/15°, 15° 사면 부츠 캔팅 엣지 20°)에서 잰다.

    python -m flyski_sim.plate_shape_test
"""
from __future__ import annotations

import numpy as np

from flyski_sim import carving_test, snowplow_test
from flyski_sim.ski_stance import snowplow_pose
from flyski_sim.ski_test import build_flat_ski_env

THRESH = np.deg2rad(0.05)


def _hinges(p, task, side):
    names = sorted((j for b in task.skis[side].bodies for j in b.get_children('joint')
                    if j.name.startswith('ski_bend_')), key=lambda j: int(j.name.split('_s')[-1]))
    return [p.model.name2id(j.full_identifier, 'joint') for j in names]


def _shape(p, task, side, jids):
    m = p.model
    dev = np.array([p.data.qpos[m.jnt_qposadr[j]] - m.qpos_spring[m.jnt_qposadr[j]] for j in jids])
    # 판 길이 방향 순서(테일->팁)로 정렬돼 있다. 테일 쪽 힌지는 부호가 반대로 정의돼
    # 있어서(체인이 가운데서 양쪽으로 뻗음) 같은 모양이면 같은 부호가 되게 맞춘다.
    n = len(dev)
    signed = np.where(np.arange(n) < n // 2, -dev, dev)
    big = signed[np.abs(signed) > THRESH]
    zig = bool(len(big) > 1 and np.any(np.sign(big[1:]) != np.sign(big[:-1])))
    bodies = sorted(task.skis[side].bodies, key=lambda b: int(b.name.split('_s')[-1]))
    pts = np.array([p.bind(b).xpos for b in bodies])
    R = p.bind(task.skis[side].root_body).xmat.reshape(3, 3)
    z = pts @ R[:, 2]
    sag = 0.5 * (z[0] + z[-1]) - z[len(z) // 2]
    length = 2 * task.skis[side].half_length_cm
    return zig, sag * 1e4, float(np.sum(signed)) / length


def measure(env, task, action_fn, steps, skip):
    p = env.physics
    jids = {s: _hinges(p, task, s) for s in task.skis}
    zig, sag, curv = [], [], []
    for k in range(steps):
        env.step(action_fn(k))
        if k >= skip:
            for s in task.skis:
                z, sg, cv = _shape(p, task, s, jids[s])
                zig.append(z)
                sag.append(sg)
                curv.append(cv)
    return dict(zigzag_frac=round(float(np.mean(zig)), 2), sag_um=round(float(np.mean(sag)), 2),
                sag_um_range=(round(float(np.min(sag)), 1), round(float(np.max(sag)), 1)),
                bend_curv=round(float(np.mean(curv)), 4))


def main():
    out = {}
    env, task = build_flat_ski_env('all_mountain', 5)
    env.reset()
    nu = env.action_spec().shape[0]
    out['flat_stance'] = measure(env, task, lambda k: np.zeros(nu), 300, 100)

    env, task = snowplow_test.build()
    env.reset()
    names = env.action_spec().name.split('\t')
    pose, _ = snowplow_pose(task.walker, 20., 15.)
    off = np.array([pose[n] - task._stance[n] if n in pose else 0. for n in names])
    out['slope_snowplow_20_15'] = measure(env, task, lambda k: off * min(k / 20, 1.), 150, 50)

    env, task = carving_test.build()
    env.reset()
    p = env.physics
    rolls = [p.model.name2id(f'walker/ski_bind_{s}_roll', 'joint') for s in task.skis]

    def cant(k):
        if k * env.control_timestep() > 0.05:
            p.model.qpos_spring[p.model.jnt_qposadr[rolls]] = np.deg2rad(20.)
        return np.zeros(len(names))
    out['slope_edge20'] = measure(env, task, cant, 500, 150)
    for k, v in out.items():
        print(k, v)
    return out


if __name__ == '__main__':
    main()
