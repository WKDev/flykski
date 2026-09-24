# S자 목표 궤적 추종(TurnTrackEnv)을 조향 전문가나 학습 정책으로 돌려 궤적/다리 관절/하중 이동을 그래프로 남기는 평가 스크립트
"""S자 추종 평가(RESEARCH_NOTES 39번).

    python -m flyski_sim.turn_eval                         # 조향 전문가
    python -m flyski_sim.turn_eval --model runs/x/model.zip

renders/turn/<이름>.png: (1) 위에서 본 목표(초록) 대 실제(빨강) 궤적, (2) 조향 입력과 방향,
(3) 다리 관절각(스탠스 대비) 몇 개, (4) 무게중심 좌우 위치(두 판 사이 -1~+1)와 오른쪽 판
하중 비율. "다리가 실제로 움직이고 하중이 옮겨지는가"를 보는 용도.
"""
from __future__ import annotations

import argparse
import os
import warnings

import mujoco
import numpy as np

from flyski_sim.rl_task import TurnTrackEnv

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'renders', 'turn')
JOINTS = ('coxa_abduct_T2_left', 'femur_T2_left', 'tibia_T2_left',
          'coxa_abduct_T2_right', 'femur_T2_right', 'tibia_T2_right')


def run(policy, env, seed=0):
    obs, _ = env.reset(seed=seed)
    p = env.env.physics
    m = p.model
    side = {m.name2id(g.full_identifier, 'geom'): s for s, u in env.task.skis.items() for g in u.geoms}
    jq = {j: m.jnt_qposadr[m.name2id(f'walker/{j}', 'joint')] for j in JOINTS}
    f6 = np.zeros(6)
    log = {k: [] for k in ('x', 'y', 'yref', 'cross', 'u', 'yaw', 'com', 'share', 'reason')}
    log.update({j: [] for j in JOINTS})
    total = 0.
    while True:
        a = policy(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        pos = p.data.xpos[env._th]
        log['x'].append(pos[0])
        log['y'].append(pos[1])
        log['yref'].append(env.reference.y(pos[0]))
        log['cross'].append(env._errors(p)[1])             # 경로까지 수직 거리(+ = 경로가 왼쪽).
        log['u'].append(getattr(policy, 'last_u', np.nan))
        log['yaw'].append(np.degrees(env._yaw(p)))
        for j in JOINTS:
            log[j].append(p.data.qpos[jq[j]] - env.task._stance[j])
        roots = [p.data.xpos[b] for b in env._roots]
        R = p.data.xmat[env._th].reshape(3, 3)
        half = 0.5 * float((roots[0] - roots[1]) @ R[:, 1])
        mid = 0.5 * (roots[0] + roots[1])
        log['com'].append(float((p.data.subtree_com[env._th] - mid) @ R[:, 1]) / max(half, 1e-6))
        F = {'left': 0., 'right': 0.}
        for i in range(p.data.ncon):
            c = p.data.contact[i]
            g = c.geom1 if c.geom1 in side else c.geom2 if c.geom2 in side else None
            if g is not None:
                mujoco.mj_contactForce(m.ptr, p.data.ptr, i, f6)
                F[side[g]] += max(f6[0], 0.)
        log['share'].append(F['right'] / max(F['left'] + F['right'], 1e-9) if F['left'] + F['right'] > 0 else np.nan)
        if term or trunc:
            log['reason'] = info['reason']
            break
    log['return'] = total
    log['ref'] = env.reference
    return log


def plot(log, name, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)
    x = np.array(log['x']) - log['x'][0]
    t = np.arange(len(x)) * 0.01
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    ref = log['ref']
    keep = ref.x <= log['x'][-1] + 10.
    ax[0].plot(ref.yy[keep] - log['y'][0], ref.x[keep] - log['x'][0], 'g', label='desired')
    ax[0].plot(np.array(log['y']) - log['y'][0], x, 'r', label='actual')
    ax[0].invert_yaxis()
    ax[0].set_xlabel('y (cm, + = left)')
    ax[0].set_ylabel('downhill (cm)')
    ax[0].set_title('top view')
    ax[0].legend()
    ax[1].plot(t, log['u'], 'k', label='steer u (+ = left)')
    ax[1].plot(t, np.array(log['yaw']) / 45., 'b', label='yaw / 45 deg')
    ax[1].set_title('steering')
    ax[1].legend()
    for j in JOINTS:
        ax[2].plot(t, np.degrees(log[j]), label=j.replace('_T2', ''))
    ax[2].set_title('T2 leg joints - stance (deg)')
    ax[2].legend(fontsize=7)
    ax[3].plot(t, log['com'], 'm', label='COM lateral (-1 R plate, +1 L plate)')
    ax[3].plot(t, log['share'], 'c', label='right plate load share')
    ax[3].set_title('weight shift')
    ax[3].legend(fontsize=8)
    for a in ax[1:]:
        a.set_xlabel('time (s)')
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(f"{title}: return {log['return']:.0f}, end {log['reason']}, {len(x) * 0.01:.1f}s")
    fig.tight_layout()
    path = os.path.join(OUT, f'{name}.png')
    fig.savefig(path, dpi=100)
    return path


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    env = TurnTrackEnv(seed=args.seed)
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device='cpu')
        policy = lambda o: model.predict(o, deterministic=True)[0]
        name, title = 'policy', f'policy {args.model}'
    else:
        from flyski_sim.experts import SteerExpert
        policy = SteerExpert(env)
        name, title = 'steer_expert', 'steer expert'
    log = run(policy, env, args.seed)
    rms = float(np.sqrt(np.mean(np.square(log['cross']))))
    print(f'{title}: return {log["return"]:.1f}, {len(log["x"])} steps, end {log["reason"]}, '
          f'cross-track rms {rms:.2f}cm, downhill {log["x"][-1] - log["x"][0]:.1f}cm, '
          f'joint range(deg) {[round(float(np.degrees(np.ptp(log[j]))), 1) for j in JOINTS]}, '
          f'COM range {np.nanmin(log["com"]):+.2f}..{np.nanmax(log["com"]):+.2f}, '
          f'right load {np.nanmin(log["share"]):.2f}..{np.nanmax(log["share"]):.2f}')
    print('saved', plot(log, name, title))


if __name__ == '__main__':
    main()
