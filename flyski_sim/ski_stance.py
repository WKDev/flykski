# 좌/우 T1/T2/T3 발끝이 일직선(T2가 가운데)이 되는 "스키 스탠스" 다리 관절각을 IK로 구한다
"""스키 스탠스(RESEARCH_NOTES 34번, A안).

flybody 기본 자세(qpos0)에서는 가운데 다리(T2)가 앞/뒷다리보다 옆으로 ~0.08cm 더
벌어져 있어서, T1/T3 발끝 중점을 중심으로 만든 플레이트에서 T2가 판 밖에 있었다.
여기서는 한쪽 다리 3개의 발끝을
- 옆(y): 세 발끝 y의 평균,
- 앞뒤(x): T2 = 몸 전체 무게중심 x, T1/T3 = 그로부터 ±(기본 자세 T1~T3 거리의 절반),
- 높이(z): 기본 자세 발끝 높이
로 옮기는 관절각을 최소제곱 IK로 구한다(주 관절 7개/다리, tarsus2~5 힘줄 관절은 0 유지).
결과는 thorax 기준 좌표계에서 정의되므로 몸의 위치/자세와 무관하다.
"""
from __future__ import annotations

import numpy as np
from dm_control import mjcf
from scipy.optimize import least_squares

from flyski_sim.ski_attachment import _find_claw_tip_offset

SIDES = ('left', 'right')
LEGS = ('T1', 'T2', 'T3')
_MAIN_JOINTS = ('coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus')
_REG = 0.003           # 관절각 크기 정규화(해가 여럿일 때 기본 자세에 가까운 쪽).
_MAX_TIP_ERR_CM = 0.005

_cache: dict[int, dict[str, float]] = {}


def _tips_th(physics, walker):
    th = physics.model.name2id('thorax', 'body')
    R_th = physics.data.xmat[th].reshape(3, 3)
    x_th = physics.data.xpos[th]
    out = {}
    for side in SIDES:
        for leg in LEGS:
            name = f'claw_{leg}_{side}'
            bid = physics.model.name2id(name, 'body')
            off = _find_claw_tip_offset(walker.mjcf_model.find('body', name))
            tip = physics.data.xpos[bid] + physics.data.xmat[bid].reshape(3, 3) @ off
            out[f'{leg}_{side}'] = R_th.T @ (tip - x_th)
    return out


def stance_targets(walker) -> dict[str, np.ndarray]:
    """기본 자세에서 계산한 발끝 목표 위치(thorax 좌표계)."""
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    tips = _tips_th(physics, walker)
    th = physics.model.name2id('thorax', 'body')
    R_th = physics.data.xmat[th].reshape(3, 3)
    com_x = float((physics.data.subtree_com[th] - physics.data.xpos[th]) @ R_th[:, 0])
    targets = {}
    for side in SIDES:
        t1, t2, t3 = (tips[f'{l}_{side}'] for l in LEGS)
        y = float(np.mean([t1[1], t2[1], t3[1]]))
        z = float(np.mean([t1[2], t2[2], t3[2]]))
        half = 0.5 * float(t1[0] - t3[0])
        for leg, dx in (('T1', half), ('T2', 0.), ('T3', -half)):
            targets[f'{leg}_{side}'] = np.array([com_x + dx, y, z])
    return targets


def compute_ski_stance(walker) -> dict[str, float]:
    """{관절 이름(접두어 없음): 각도 rad}. 같은 walker 객체면 캐시."""
    key = id(walker)
    if key in _cache:
        return _cache[key]
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    targets = stance_targets(walker)
    stance = {}
    for side in SIDES:
        for leg in LEGS:
            names = [f'{j}_{leg}_{side}' for j in _MAIN_JOINTS]
            jids = [physics.model.name2id(n, 'joint') for n in names]
            qadr = physics.model.jnt_qposadr[jids]
            lo, hi = physics.model.jnt_range[jids].T
            target = targets[f'{leg}_{side}']

            def residual(q):
                physics.data.qpos[:] = 0.
                physics.data.qpos[qadr] = q
                physics.forward()
                err = _tips_th(physics, walker)[f'{leg}_{side}'] - target
                return np.concatenate([err, _REG * q])

            q0 = np.clip(np.zeros(len(jids)), lo + 1e-6, hi - 1e-6)
            sol = least_squares(residual, q0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
            tip_err = float(np.linalg.norm(residual(sol.x)[:3]))
            if tip_err > _MAX_TIP_ERR_CM:
                raise RuntimeError(f'ski stance IK failed for {leg}_{side}: '
                                   f'tip error {tip_err:.4f}cm')
            stance.update({n: float(v) for n, v in zip(names, sol.x)})
    _cache[key] = stance
    return stance
