# 일반인 상급 카빙 기준(폴라인 +-70도 코스 완주율, 턴 반경 대 판 휨 호, 그립 손실, 엣지 일관성)으로 정책을 채점하는 평가 도구
"""카빙 성공 판정(RESEARCH_NOTES 49번, 사용자와 합의한 기준).

매번 다른 무작위 코스(주기 50~80cm, 첫 턴 방향 무작위)에서 최대 진행각을 PSI_DEG로 고정해
결정적 정책으로 달린다.

1. 완주율: 12초 버팀(통로 이탈/넘어짐 없이) >= 80%.
2. 호 따라 돌기: 턴 구간에서 실제 턴 반경(속력 / 진행각 변화율) / 판이 휜 호 반경(바깥 판 조각
   위치에 원 맞춤, 설면 투영)의 중앙값이 0.7~1.4.
3. 덜 미끄러짐: 그립이 옆미끄럼으로 소산한 에너지 / 위치에너지 감소. 기준값은 비교용(판을
   틀어 도는 턴과 비교), 합격선 없이 보고.
4. 엣지 일관성: 턴 구간에서 두 판 엣지가 같은 쪽이고 그 쪽으로 도는 스텝 비율 >= 80%.

    FLYSKI_SLOPE_DEG=20 python -m flyski_sim.carve_eval runs/race6/model_best.zip
    FLYSKI_SLOPE_DEG=25 python -m flyski_sim.carve_eval runs/race25a/model_best.zip --courses 20
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from flyski_sim.carve_physics_test import _circle_radius
from flyski_sim.rl_task import COURSE, CONTROL_DT, ENVS, PARALLEL_TURN_HEADING_DEG, TurnPath

PSI_DEG = 70.
PASS_COMPLETE, PASS_ARC, PASS_EDGE = 0.8, (0.7, 1.4), 0.8


def evaluate(model_path: str, courses: int = 20, seed: int = 1234, psi_deg: float = PSI_DEG) -> dict:
    from flyski_sim.record import make_policy
    env = ENVS['race'](seed=seed)
    p = env.env.physics
    m, d = p.model, p.data
    policy = make_policy(env, 'model', model_path)
    grip = env.task._grip
    n = env.task._slope_n
    e1 = np.cross([0., 1., 0.], n)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    bodies = [sorted({int(b) for g, b in enumerate(grip._geom_body) if grip._side_of[g] == k})
              for k in range(len(grip._sides))]
    weight = float(m.body_subtreemass[env._th] * np.linalg.norm(m.opt.gravity))
    done, ratios, edge_ok, edge_n, losses = [], [], 0, 0, []
    for c in range(courses):
        env.reset()
        pos = d.xpos[env._th]
        sign = np.sign(env.reference.psi_max)
        env.reference = TurnPath(float(pos[0]), float(pos[1]), sign * np.deg2rad(psi_deg), env.reference.period)
        obs = env._obs(env.task.ski_metrics(p), env._odor(p), 0.)
        grip = env.task._grip                    # EdgeGrip은 에피소드마다 새로 만들어진다(tasks.initialize_episode).
        grip.dissipated = 0.
        h0 = float(d.xpos[env._th] @ n)
        z0 = float(d.xpos[env._th][2])
        prev_head, k = None, 0
        while True:
            obs, _, term, trunc, info = env.step(policy(obs))
            k += 1
            v = d.qvel[:2]
            speed = float(np.linalg.norm(v))
            head = float(np.arctan2(v[1], v[0]))
            psi_path = abs(env.reference.psi[env.reference._i])
            if prev_head is not None and speed > 5. and psi_path > np.deg2rad(PARALLEL_TURN_HEADING_DEG):
                rate = float(np.angle(np.exp(1j * (head - prev_head)))) / CONTROL_DT
                _, (e_l, e_r) = env._plate_state(p)
                if abs(rate) > 0.3:
                    edge_n += 1
                    edge_ok += int(np.sign(e_l) == np.sign(e_r) and np.sign(rate) == -np.sign(e_l))
                    if k % 5 == 0:
                        outer = 1 if rate > 0 else 0          # 좌회전(+)이면 오른쪽 판이 바깥.
                        pts = d.xpos[bodies[outer]]
                        r_plate = _circle_radius(np.c_[pts @ e1, pts @ e2])
                        if np.isfinite(r_plate) and r_plate < 1e3:
                            ratios.append((speed / abs(rate)) / r_plate)
            prev_head = head
            if term or trunc:
                break
        done.append(info['reason'] in ('time', 'finish'))
        drop = z0 - float(d.xpos[env._th][2])
        losses.append(grip.dissipated / max(weight * drop, 1e-9))
        print(f'  course {c:2d} period {env.reference.period:4.0f}cm first turn {"L" if sign > 0 else "R"}: '
              f'{info["reason"]:8s} {k * CONTROL_DT:5.2f}s grip loss / PE drop {losses[-1]:.3f}', flush=True)
    ratios = np.array(ratios)
    out = dict(complete=float(np.mean(done)),
               arc_ratio=float(np.median(ratios)) if ratios.size else np.nan,
               arc_ratio_iqr=tuple(np.percentile(ratios, [25, 75]).round(2)) if ratios.size else (np.nan, np.nan),
               loss=float(np.median(losses)), edge=edge_ok / max(edge_n, 1))
    out['pass'] = dict(complete=out['complete'] >= PASS_COMPLETE,
                       arc=PASS_ARC[0] <= out['arc_ratio'] <= PASS_ARC[1], edge=out['edge'] >= PASS_EDGE)
    return out


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--courses', type=int, default=20)
    ap.add_argument('--psi', type=float, default=PSI_DEG)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()
    print(f'{args.model} slope {COURSE["slope_deg"]:.0f}deg, psi +-{args.psi:.0f}deg, {args.courses} courses')
    r = evaluate(args.model, args.courses, args.seed, args.psi)
    ok = lambda b: 'PASS' if b else 'fail'
    print(f'1 complete {r["complete"]:.0%} [{ok(r["pass"]["complete"])}]  '
          f'2 turn R / plate arc R median {r["arc_ratio"]:.2f} IQR {r["arc_ratio_iqr"]} [{ok(r["pass"]["arc"])}]  '
          f'3 grip loss / PE drop {r["loss"]:.3f}  4 edge consistency {r["edge"]:.0%} [{ok(r["pass"]["edge"])}]')


if __name__ == '__main__':
    main()
