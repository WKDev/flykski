# 두 판을 고정 엣지각으로 세운 채 활강시켜 턴 반경, 판이 실제로 휜 호 반경, 그립 소산 에너지를 재는 카빙 물리 검사
"""카빙 물리 검사 2판(RESEARCH_NOTES 48번).

정책 없이 IK 자세(두 판 같은 쪽 엣지 phi, 11자)를 고정하고 폴라인 방향 초기 속도로 1.5초
활강한다. 그립 모델 조합(edge_grip.ARC_SOURCE x GRIP_MODEL)별로 잰다.

- turn_r: 진행 방향(속도 벡터) 회전 반경 = 속력 / |진행각 변화율|의 중앙값(cm).
- shape_r: 바깥 판(엣지 쪽) 조각 위치를 설면에 투영해 원을 맞춘 반경(cm) = 판이 실제로 휜 호.
- formula_r: R_sidecut x cos(phi), 35번 가정의 호 반경.
- loss: 그립 소산 에너지 / 이동 거리(몸무게 x cm 단위로 나눈 무차원, 0이면 옆미끄럼 손실 없음).

    python -m flyski_sim.carve_physics_test                  # 4조합 전부
    python -m flyski_sim.carve_physics_test shape platform   # 한 조합만(병렬 실행용)
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

import flyski_sim.edge_grip as eg
from flyski_sim.rl_task import ENVS
from flyski_sim.ski_stance import solve_plate_pose

EDGES_DEG = (5., 10., 15., 20., 30.)
SEEDS = (0, 1)
SECONDS = 1.5


def _circle_radius(xy: np.ndarray) -> float:
    """점들에 원을 대수적으로 맞춘 반경. 거의 일직선이면 큰 값."""
    x, y = xy[:, 0] - xy[:, 0].mean(), xy[:, 1] - xy[:, 1].mean()
    A = np.c_[x, y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    r2 = c + (cx / 2) ** 2 + (cy / 2) ** 2
    return float(np.sqrt(r2)) if r2 > 0 else float('inf')


def run(edge_deg: float, seed: int = 0) -> dict:
    env = ENVS['parallel'](seed=seed)
    env.reset(seed=seed)
    p = env.env.physics
    roll = -edge_deg                                     # 두 판 왼쪽 엣지(좌회전 쪽, experts와 같은 부호).
    pose, _ = solve_plate_pose(env.task.walker, {'left': 0., 'right': 0.}, {'left': roll, 'right': roll})
    act = np.clip(np.array([pose[n] - env._stance_q[i] for i, n in enumerate(env._leg_names)])
                  / env._action_scale, -1., 1.)
    grip = env.task._grip
    n = env.task._slope_n
    e1 = np.cross([0., 1., 0.], n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    bodies = {k: sorted({int(b) for g, b in enumerate(grip._geom_body) if grip._side_of[g] == k})
              for k in range(len(grip._sides))}
    grip.dissipated = 0.
    x0 = p.data.xpos[env._th].copy()
    heads, speeds, shape_r, edges = [], [], [], []
    for k in range(int(SECONDS / 0.01)):
        _, _, term, _, info = env.step(act)
        if info.get('reason') in ('fallen', 'body_touch'):
            break
        v = p.data.qvel[:2]
        heads.append(np.arctan2(v[1], v[0]))
        speeds.append(float(np.linalg.norm(v)))
        _, (e_l, e_r) = env._plate_state(p)
        edges.append(np.rad2deg(abs(e_l) + abs(e_r)) / 2)
        if k % 5 == 0 and k > 20:
            for side in bodies.values():
                pts = p.data.xpos[side]
                shape_r.append(_circle_radius(np.c_[pts @ e1, pts @ e2]))
    if len(heads) < 25:                                  # 착지 직후 넘어짐.
        return dict(edge_cmd=edge_deg, edge=np.nan, steps=len(heads), turn_r=np.inf, turn_deg=0.,
                    shape_r=np.inf, formula_r=np.nan, loss=np.nan, speed=np.nan)
    heads = np.unwrap(np.array(heads))
    speeds = np.array(speeds)
    rate = np.abs(np.gradient(heads, 0.01))
    ok = (speeds > 3.) & (rate > 1e-3)
    ok[:20] = False
    dist = float(np.linalg.norm((p.data.xpos[env._th] - x0)[:2]))
    weight = float(p.model.body_subtreemass[env._th] * np.linalg.norm(p.model.opt.gravity))
    return dict(edge_cmd=edge_deg, edge=float(np.median(edges)) if edges else np.nan,
                steps=len(speeds), turn_r=float(np.median(speeds[ok] / rate[ok])) if ok.any() else np.inf,
                turn_deg=float(np.rad2deg(heads[-1] - heads[20])) if len(heads) > 21 else 0.,
                shape_r=float(np.median(shape_r)) if shape_r else np.inf,
                formula_r=float(np.mean(grip._rsc) * np.cos(np.deg2rad(np.median(edges) if edges else 0.))),
                loss=grip.dissipated / max(weight * dist, 1e-9), speed=float(np.median(speeds)))


def main():
    import sys
    warnings.filterwarnings('ignore')
    combos = ([tuple(sys.argv[1:3])] if len(sys.argv) > 2 else
              itertools.product(('formula', 'shape'), ('ramp', 'platform')))
    for arc, model in combos:
        eg.ARC_SOURCE, eg.GRIP_MODEL = arc, model
        for e, seed in itertools.product(EDGES_DEG, SEEDS):
            r = run(e, seed)
            print(f'{arc:7s} {model:8s} seed {seed} edge cmd {e:4.0f} actual {r["edge"]:5.1f} | steps {r["steps"]:3d} '
                  f'turn {r["turn_deg"]:+6.1f}deg radius {r["turn_r"]:6.1f}cm | plate shape R {r["shape_r"]:7.1f}cm '
                  f'formula R {r["formula_r"]:5.1f}cm | loss {r["loss"]:.3f} speed {r["speed"]:4.1f}', flush=True)


if __name__ == '__main__':
    main()
