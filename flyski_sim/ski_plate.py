"""업무2/3 v2: 좌/우 스키 플레이트(N=2) — 한쪽 다리 3개를 탄성 연결한, 휘는 스키.

RESEARCH_NOTES.md 32번. 기존 N=6(다리당 미니스키, `ski_attachment.py`)은 스키마다
한 점(스프링 ball 조인트)으로만 자세를 잡아 롤에 약했고, 스키가 강체 box 하나라
프로파일의 flex/torsion/camber 값이 물리에 전혀 반영되지 않았다(장식용 값이었음).

구조(한쪽 기준):
- 판 = 긴 축 방향 K개 조각의 체인. 가운데(root) 조각이 T2 발끝(claw tip)에 스프링
  ball 조인트로 붙는다(피벗 = 발끝).
- T1/T3 발끝은 가장 가까운 조각에 **soft `connect` 등식 구속**(점 연결, 회전 자유)으로
  붙는다. 세 점이 판의 평면을 정하되 rigid weld가 아니라서 과구속이 아니다
  (RESEARCH_NOTES 7번에서 폐기된 건 "3다리 rigid weld"였다).
- 조각 사이에는 굽힘(판 로컬 y축) + 비틀림(판 로컬 x축) 스프링 힌지. 강성은
  "체중 절반이 걸렸을 때 몇 도 휘는가"로 역산한다 — 초파리 스케일이라 실제 스키
  재료 강성을 그대로 쓰면 사실상 안 휜다. 캠버/로커는 굽힘 힌지의 springref로,
  사이드컷은 조각별 폭으로 표현한다.
- 판 축은 rest pose(qpos0)에서 T3→T1 발끝을 잇는 선 방향, 판 윗면은 발끝 높이.
"""
from __future__ import annotations

import mujoco
import numpy as np
from dm_control import mjcf

from flyski_sim.ski_attachment import SkiUnit, _find_claw_tip_offset
from flyski_sim.ski_profiles import SkiProfile

SIDES = ('left', 'right')
N_SEGMENTS = 7            # 홀수 — 가운데 조각이 root.
HALF_THICKNESS_CM = 0.004
BINDING_HEIGHT_CM = 0.01
_THETA_MAX_EXTRA_DEG = 8.0  # stiffness=0일 때 추가로 허용하는 휨각.
_CAMBER_DEG_PER_HINGE = 1.5
_TARGET_OMEGA = 1000.     # 힌지 고유진동수 상한(rad/s) — physics dt(2e-4s)에서 안정.
_HINGE_RANGE_RAD = 0.35   # 굽힘/비틀림 힌지 가동범위 ±20°.
_BIND_RANGE_RAD = 0.8     # T2 바인딩 ball 조인트 최대 회전 ~46°.
_LEG_BODY_PREFIXES = ('coxa', 'femur', 'tibia', 'tarsus', 'claw')


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(float))
    return q


def _rest_pose_info(walker):
    """qpos0에서 thorax 기준 발끝 위치, claw 월드 포즈를 독립 컴파일로 구한다."""
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    th = physics.model.name2id('thorax', 'body')
    R_th = physics.data.xmat[th].reshape(3, 3).copy()
    x_th = physics.data.xpos[th].copy()
    info = {}
    for side in SIDES:
        for seg in ('T1', 'T2', 'T3'):
            leg = f'{seg}_{side}'
            claw = walker.mjcf_model.find('body', f'claw_{leg}')
            bid = physics.model.name2id(f'claw_{leg}', 'body')
            R_c = physics.data.xmat[bid].reshape(3, 3).copy()
            x_c = physics.data.xpos[bid].copy()
            tip_off = _find_claw_tip_offset(claw)
            tip_world = x_c + R_c @ tip_off
            info[leg] = dict(claw=claw, R=R_c, x=x_c, tip_off=tip_off,
                             tip_th=R_th.T @ (tip_world - x_th))
    total_mass = float(physics.model.body_subtreemass[th])
    gravity = float(abs(physics.model.opt.gravity[2]))
    return info, R_th, x_th, total_mass * gravity


def _width_at(profile: SkiProfile, s: float) -> float:
    """s∈[-1(테일), +1(팁)]에서의 스키 폭 — 사이드컷을 2차 보간."""
    if s >= 0:
        return profile.width_waist_cm + (profile.width_tip_cm - profile.width_waist_cm) * s * s
    return profile.width_waist_cm + (profile.width_tail_cm - profile.width_waist_cm) * s * s


def attach_side_plates(walker, profile: SkiProfile,
                       n_segments: int = N_SEGMENTS) -> dict[str, SkiUnit]:
    """walker의 좌/우 다리 3개씩에 휘는 스키 플레이트를 하나씩 붙인다."""
    info, R_th, x_th, weight = _rest_pose_info(walker)
    model = walker.mjcf_model
    units = {}
    for side in SIDES:
        t1, t2, t3 = (info[f'{s}_{side}'] for s in ('T1', 'T2', 'T3'))
        # 판 프레임(thorax 기준): x = T3→T1 방향(수평 투영), z = thorax 위.
        axis = t1['tip_th'] - t3['tip_th']
        axis[2] = 0.
        span = float(np.linalg.norm(axis))
        ex = axis / span
        ez = np.array([0., 0., 1.])
        ey = np.cross(ez, ex)
        R_plate_th = np.stack([ex, ey, ez], axis=1)         # plate -> thorax
        tip_z = np.mean([t['tip_th'][2] for t in (t1, t2, t3)])
        mid = 0.5 * (t1['tip_th'] + t3['tip_th'])
        # 판 윗면을 발끝보다 BINDING_HEIGHT_CM 아래에 둔다(부츠/바인딩 높이). 발톱-판
        # 충돌은 exclude라서, 바인딩이 조금만 처져도 발톱이 판을 통과해 눈(마찰 1.0)에
        # 닿아 브레이크가 걸렸다(RESEARCH_NOTES 32번).
        center_th = np.array([mid[0], mid[1],
                              tip_z - BINDING_HEIGHT_CM - HALF_THICKNESS_CM])

        length = span + profile.length_cm
        seg_len = length / n_segments
        m = n_segments // 2
        half_w_load = 0.5 * weight                           # 판 하나가 받는 하중.
        flex_theta = np.deg2rad(1. + _THETA_MAX_EXTRA_DEG * (1 - profile.flex_stiffness))
        tors_theta = np.deg2rad(1. + _THETA_MAX_EXTRA_DEG * (1 - profile.torsional_stiffness))
        # 모멘트 팔 L/16: 처음 L/4로 잡았더니 실측 휨이 목표의 1/5~1/8이었다(보정값).
        k_bend = half_w_load * (length / 16) / flex_theta
        k_tors = half_w_load * (profile.width_waist_cm / 2) / tors_theta
        seg_mass = 3 * profile.mass_g / n_segments           # 기존 N=6과 총 질량 동일.

        def hinge_kw(k, rng=(-_HINGE_RANGE_RAD, _HINGE_RANGE_RAD)):
            # fruitfly.xml 기본 joint 클래스가 limited="true"라 range를 명시해야 한다.
            arm = max(k / _TARGET_OMEGA ** 2, 1e-8)
            return dict(stiffness=k, armature=arm, range=rng,
                        damping=2 * 0.7 * np.sqrt(k * arm))

        # root 조각: claw_T2의 자식. 월드 포즈 -> claw 로컬로 변환.
        R_plate_w = R_th @ R_plate_th
        center_w = x_th + R_th @ center_th
        R_c, x_c = t2['R'], t2['x']
        root_pos = R_c.T @ (center_w - x_c)
        root_quat = _mat_to_quat(R_c.T @ R_plate_w)
        bodies, geoms = [], []

        def add_segment(parent, idx, pos, quat=None):
            kw = dict(name=f'ski_{side}_s{idx}', pos=tuple(pos))
            if quat is not None:
                kw['quat'] = tuple(quat)
            body = parent.add('body', **kw)
            s = (idx - m) / m                                  # -1(테일)~+1(팁).
            g = body.add('geom', name=f'ski_geom_{side}_s{idx}', type='box',
                         size=(seg_len / 2, _width_at(profile, s) / 2, HALF_THICKNESS_CM),
                         mass=seg_mass, contype=1, conaffinity=1, condim=3,
                         # priority=1: 스키-지면 접촉은 지면이 아니라 이 geom의
                         # 마찰(설질별 ski_friction, 매 스텝 갱신)을 쓴다.
                         priority=1, friction=(0.05, 0.005, 0.0001),
                         rgba=(0.7, 0.85, 1.0, 0.9))
            bodies.append(body)
            geoms.append(g)
            return body

        root = add_segment(t2['claw'], m, root_pos, root_quat)
        # T2 바인딩: 발끝을 피벗으로 한 스프링 ball 조인트.
        t2_tip_plate = R_plate_w.T @ (x_c + R_c @ t2['tip_off'] - center_w)
        k_bind = 2 * k_bend * profile.attachment_compliance
        root.add('joint', name=f'ski_bind_{side}', type='ball', pos=tuple(t2_tip_plate),
                 **hinge_kw(k_bind, rng=(0., _BIND_RANGE_RAD)))

        seg_by_idx = {m: root}
        for direction in (+1, -1):
            parent = root
            for step in range(1, m + 1):
                idx = m + direction * step
                body = add_segment(parent, idx, (direction * seg_len, 0., 0.))
                hinge_pos = (-direction * seg_len / 2, 0., 0.)
                # 캠버(+)=팁/테일이 아래로, 로커(-)=위로. +y 회전은 +x 끝을 내린다.
                ref = direction * profile.rocker_camber * np.deg2rad(_CAMBER_DEG_PER_HINGE)
                body.add('joint', name=f'ski_bend_{side}_s{idx}', type='hinge',
                         axis=(0, 1, 0), pos=hinge_pos, springref=float(ref),
                         **hinge_kw(k_bend))
                body.add('joint', name=f'ski_twist_{side}_s{idx}', type='hinge',
                         axis=(1, 0, 0), pos=hinge_pos, **hinge_kw(k_tors))
                seg_by_idx[idx] = body
                parent = body

        def add_binding_post(leg, idx, tip_plate):
            """시각 전용(충돌 없음) 바인딩 기둥: 판 중심선 윗면 -> 발끝. 구속 자체는
            보이지 않아서 렌더에서 발이 판 위에 떠 보였다."""
            tip_seg = tip_plate - np.array([(idx - m) * seg_len, 0., 0.])
            base = np.array([tip_seg[0], 0., HALF_THICKNESS_CM])
            seg_by_idx[idx].add('geom', name=f'ski_post_{leg}_{side}', type='capsule',
                                fromto=(*base, *tip_seg), size=(0.0025,),
                                contype=0, conaffinity=0, mass=0, group=1,
                                rgba=(0.25, 0.3, 0.4, 1.0))

        add_binding_post('T2', m, t2_tip_plate)
        # T1/T3: soft connect (점 연결). 낮은 compliance = 더 물렁(timeconst 증가).
        timeconst = max(0.002 / max(profile.attachment_compliance, 1e-3), 0.0005)
        for leg, t in (('T1', t1), ('T3', t3)):
            tip_plate = R_plate_w.T @ (t['x'] + t['R'] @ t['tip_off'] - center_w)
            idx = int(np.clip(m + round(float(tip_plate[0]) / seg_len), 0, n_segments - 1))
            model.equality.add('connect', name=f'ski_bind_{leg}_{side}',
                               body1=t['claw'], body2=seg_by_idx[idx],
                               anchor=tuple(t['tip_off']), solref=(timeconst, 1.0))
            add_binding_post(leg, idx, tip_plate)

        # 같은 쪽 다리 몸체와 판의 충돌은 제외(바인딩/다리가 판을 밀어내지 않게).
        leg_bodies = [b for b in model.find_all('body')
                      if b.name and b.name.endswith(f'_{side}')
                      and b.name.startswith(_LEG_BODY_PREFIXES)]
        for seg in bodies:
            for lb in leg_bodies:
                model.contact.add('exclude', body1=seg, body2=lb)

        area = length * np.mean([_width_at(profile, (i - m) / m) for i in range(n_segments)])
        units[side] = SkiUnit(name=side, root_body=root, bodies=bodies, geoms=geoms,
                              half_length_cm=length / 2, area_cm2=float(area))
    return units
