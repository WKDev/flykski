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


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def solve_plate_pose(walker, yaw_deg: dict[str, float], roll_deg: dict[str, float],
                     ori_weight: float = 0.05, shift_cm: dict[str, np.ndarray] | None = None,
                     pitch_deg: dict[str, float] | None = None
                     ) -> tuple[dict[str, float], dict[str, float]]:
    """스탠스에서 판을 T2 발끝 기준으로 요(수직축)/롤(판 축)만큼 돌린 다리 자세를 IK로 구한다.

    yaw_deg/roll_deg: {'left': 각, 'right': 각}. 요 +는 반시계(팁이 왼쪽으로), 롤 +는
    오른쪽 엣지가 내려감. 플루크(쐐기)는 left 요 -, right 요 +(팁을 모음), left 롤 +,
    right 롤 -(안쪽 엣지). T1/T3는 발끝 위치만, T2는 발끝 위치 + 발(부츠) 방향까지
    맞춘다(부츠 요/롤 힌지가 단단해서 T2 발이 판과 같이 안 돌면 스프링과 싸움).
    shift_cm: {'left': (dx, dy, dz)} thorax 좌표계에서 판(세 발끝)을 평행이동. 판을 -d로
    옮기면 몸(무게중심)이 판 위에서 +d로 옮겨 간 것과 같다(하중 이동, 38번).
    반환: ({관절: 각}, {다리: 발끝 위치 오차 cm, T2는 방향 오차 rad도 'T2_side_ori'}).
    """
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    stance = compute_ski_stance(walker)
    for name, angle in stance.items():
        physics.named.data.qpos[name] = angle
    physics.forward()
    tips0 = _tips_th(physics, walker)
    th = physics.model.name2id('thorax', 'body')
    R_th = physics.data.xmat[th].reshape(3, 3).copy()
    claw_R0 = {}
    for side in SIDES:
        bid = physics.model.name2id(f'claw_T2_{side}', 'body')
        claw_R0[side] = R_th.T @ physics.data.xmat[bid].reshape(3, 3)
    pose, errors = dict(stance), {}
    for side in SIDES:
        pitch = np.deg2rad((pitch_deg or {}).get(side, 0.))     # + = 팁이 내려감(y축 둘레).
        R_delta = (_rot('z', np.deg2rad(yaw_deg[side])) @ _rot('x', np.deg2rad(roll_deg[side]))
                   @ _rot('y', pitch))
        pivot = tips0[f'T2_{side}']
        shift = np.asarray((shift_cm or {}).get(side, (0., 0., 0.)), dtype=float)
        for leg in LEGS:
            names = [f'{j}_{leg}_{side}' for j in _MAIN_JOINTS]
            jids = [physics.model.name2id(n, 'joint') for n in names]
            qadr = physics.model.jnt_qposadr[jids]
            lo, hi = physics.model.jnt_range[jids].T
            target = pivot + shift + R_delta @ (tips0[f'{leg}_{side}'] - pivot)
            bid = physics.model.name2id(f'claw_{leg}_{side}', 'body')
            R_goal = R_delta @ claw_R0[side] if leg == 'T2' else None

            def residual(q):
                physics.data.qpos[qadr] = q
                physics.forward()
                err = _tips_th(physics, walker)[f'{leg}_{side}'] - target
                out = [err]
                if R_goal is not None:
                    R = R_th.T @ physics.data.xmat[bid].reshape(3, 3)
                    E = R_goal.T @ R
                    out.append(ori_weight * 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0],
                                                           E[1, 0] - E[0, 1]]))
                out.append(_REG * (q - np.array([stance[n] for n in names])))
                return np.concatenate(out)

            q0 = np.array([stance[n] for n in names])
            sol = least_squares(residual, q0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
            r = residual(sol.x)
            errors[f'{leg}_{side}'] = float(np.linalg.norm(r[:3]))
            if R_goal is not None:
                errors[f'T2_{side}_ori'] = float(np.linalg.norm(r[3:6]) / ori_weight)
            pose.update({n: float(v) for n, v in zip(names, sol.x)})
            physics.data.qpos[qadr] = sol.x
    return pose, errors


def snowplow_pose(walker, wedge_deg: float, edge_deg: float,
                  com_shift_cm: tuple[float, float, float] = (0., 0., 0.)):
    """플루크(쐐기): 팁을 모으고(각 판 wedge_deg) 안쪽 엣지를 edge_deg만큼 세운 자세.

    com_shift_cm: 몸(무게중심)을 판 위에서 옮길 양(thorax 좌표, +y = 왼쪽). 플루크 보겐은
    바깥 스키 쪽으로 옮긴다(왼쪽 턴 = 오른쪽 스키 쪽 = -y).
    """
    shift = -np.asarray(com_shift_cm, dtype=float)
    return solve_plate_pose(walker, {'left': -wedge_deg, 'right': wedge_deg},
                            {'left': edge_deg, 'right': -edge_deg},
                            shift_cm={'left': shift, 'right': shift})
