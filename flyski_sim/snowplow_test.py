# 플루크(쐐기) 자세를 고정 액션으로 걸었을 때 경사에서 감속/정지하는지 측정하는 제동 물리 테스트
"""플루크 제동 검사(RESEARCH_NOTES 37번).

`ski_stance.snowplow_pose`로 구한 쐐기 자세(팁을 모으고 안쪽 엣지를 세움)를 다리
액션(스탠스 대비 오프셋, position servo 목표각)으로 걸고, 15° 알파인 평면에서
스탠스(직활강) 대비 속도가 얼마나 줄어드는지 본다. 컨트롤러 없음, 외부 토크 없음.

    python -m flyski_sim.snowplow_test          # 표 + 판정
    python -m flyski_sim.snowplow_test --gif    # renders/snowplow/ 에 GIF/그래프

측정값
- speed_end: 마지막 0.2초 평균 속력(cm/s). v_plateau가 스탠스보다 작으면 제동.
- wedge_actual / edge_actual: 판 루트의 실제 요(팁 모임, 좌우 평균)와 안쪽 엣지각.
- skid: 접지 중 옆미끄럼 비율(플루크는 스키딩으로 제동하므로 커야 정상).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from dm_control import composer

from flybody.fruitfly import fruitfly
from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.ski_stance import snowplow_pose
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls

SLOPE_DEG = 15.
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'renders', 'snowplow')


def build(seed=3):
    arena = SlopedMoguls(dim=(30., 15.), mean_slope_deg=SLOPE_DEG, terrain_type='alpine',
                         mogul_height=0., grid_density=15)
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES['all_mountain'], walker=fruitfly.FruitFly,
                          arena=arena, time_limit=1e9, joint_filter=0., claw_friction=1.0,
                          spawn_xy=(-25., 0.))
    task.set_timesteps(control_timestep=0.01, physics_timestep=2e-4)
    env = composer.Environment(time_limit=1e9, task=task, random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True,
                               recompile_mjcf_every_episode=False)
    return env, task


def run(wedge_deg, edge_deg, seconds=1.5, ramp=0.2, frames=None):
    env, task = build()
    env.reset()
    p = env.physics
    names = env.action_spec().name.split('\t')
    pose, _ = snowplow_pose(task.walker, wedge_deg, edge_deg)
    offset = np.zeros(len(names))
    for i, n in enumerate(names):
        if n in pose:
            offset[i] = pose[n] - task._stance[n]
    th = p.model.name2id('walker/thorax', 'body')
    roots = {s: p.model.name2id(task.skis[s].root_body.full_identifier, 'body') for s in ('left', 'right')}
    n = np.array([np.sin(np.deg2rad(SLOPE_DEG)), 0., np.cos(np.deg2rad(SLOPE_DEG))])
    dt = env.control_timestep()
    speeds, skid, wedge, edge, xs = [], [], [], [], []
    for k in range(int(seconds / dt)):
        a = offset * min(k * dt / ramp, 1.)        # 착지 직후 0.2초에 걸쳐 자세로.
        env.step(a)
        m = task.ski_metrics(p)
        v = float(np.linalg.norm(p.data.qvel[:3]))
        speeds.append(v)
        xs.append(float(p.data.xpos[th][0]))
        if k * dt > ramp + 0.1:
            if m['skid_ratio'] is not None:
                skid.append(m['skid_ratio'])
            yaw = {s: np.arctan2(p.data.xmat[b][3], p.data.xmat[b][0]) for s, b in roots.items()}
            body_yaw = np.arctan2(p.data.xmat[th][3], p.data.xmat[th][0])
            # 팁 모임 = 오른쪽 판 요 - 왼쪽 판 요의 절반(몸 요 기준).
            wedge.append(np.degrees(0.5 * (np.angle(np.exp(1j * (yaw['right'] - body_yaw))) -
                                           np.angle(np.exp(1j * (yaw['left'] - body_yaw))))))
            rl = {s: np.degrees(np.arctan2(p.data.xmat[b].reshape(3, 3)[:, 1] @ n,
                                           p.data.xmat[b].reshape(3, 3)[:, 2] @ n))
                  for s, b in roots.items()}
            edge.append(0.5 * (rl['left'] - rl['right']))   # 안쪽 엣지 +.
        if frames is not None and k % 3 == 0:
            frames.append(_frame(p, th))
    up = float(p.data.xmat[th].reshape(3, 3)[:, 2] @ n)
    tail = int(0.2 / dt)
    return dict(wedge_deg=wedge_deg, edge_deg=edge_deg,
                wedge_actual=round(float(np.mean(wedge)), 1), edge_actual=round(float(np.mean(edge)), 1),
                speed_end=round(float(np.mean(speeds[-tail:])), 1), speed_max=round(max(speeds), 1),
                distance_cm=round(xs[-1] - xs[0], 1),
                skid=round(float(np.mean(skid)), 2) if skid else None, upright=up > 0.5,
                _speeds=np.array(speeds))


def _frame(p, th):
    from dm_control.mujoco.engine import MovableCamera
    cam = MovableCamera(p, height=360, width=480)
    cam.set_pose(p.data.xpos[th], 1.1, 200, -40)
    return cam.render()


def _plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in rows:
        t = np.arange(len(r['_speeds'])) * 0.01
        ax.plot(t, r['_speeds'], label=f"wedge {r['wedge_deg']:.0f}, edge {r['edge_deg']:.0f}")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (cm/s)')
    ax.set_title(f'snowplow braking on {SLOPE_DEG:.0f} deg slope (no controller)')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.savefig(os.path.join(OUT_DIR, 'speed_by_wedge.png'), dpi=120, bbox_inches='tight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gif', action='store_true')
    args = ap.parse_args()
    if args.gif:
        os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for wedge, edge in ((0., 0.), (10., 10.), (15., 12.), (20., 15.), (25., 15.)):
        frames = [] if args.gif and (wedge, edge) in ((0., 0.), (15., 12.)) else None
        r = run(wedge, edge, frames=frames)
        rows.append(r)
        print({k: v for k, v in r.items() if not k.startswith('_')}, flush=True)
        if frames:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(os.path.join(OUT_DIR, f'wedge{int(wedge)}_edge{int(edge)}.gif'),
                         save_all=True, append_images=imgs[1:], duration=50, loop=0)
    if args.gif:
        _plot(rows)
    base = rows[0]['speed_end']
    braking = [r for r in rows[1:] if r['upright'] and r['speed_end'] < 0.5 * base]
    print(f'스탠스 끝속도 {base}cm/s 대비 절반 이하로 줄인 쐐기:',
          [(r['wedge_deg'], r['edge_deg'], r['speed_end']) for r in braking])
    return bool(braking)


if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
