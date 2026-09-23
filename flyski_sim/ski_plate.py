"""업무2/3 v2: 좌/우 스키 플레이트(N=2) — 한쪽 다리 3개를 탄성 연결한, 휘는 스키.

RESEARCH_NOTES.md 32번. 기존 N=6(다리당 미니스키, `ski_attachment.py`)은 스키마다
한 점(스프링 ball 조인트)으로만 자세를 잡아 롤에 약했고, 스키가 강체 box 하나라
프로파일의 flex/torsion/camber 값이 물리에 전혀 반영되지 않았다(장식용 값이었음).

구조(한쪽 기준):
- 판 = 긴 축 방향 K개 조각의 체인. 가운데(root) 조각이 T2 발끝(claw tip)에 스프링
  ball 조인트로 붙는다(피벗 = 발끝). 판 중심 = T2 발끝(RESEARCH_NOTES 34번): 다리를
  세 발끝이 일직선인 "스키 스탠스"(`ski_stance.py`)로 두고 그 자세에서 판을 만든다.
- T1/T3 발끝은 가장 가까운 조각에 **soft `connect` 등식 구속**(점 연결, 회전 자유)으로
  붙는다. 세 점이 판의 평면을 정하되 rigid weld가 아니라서 과구속이 아니다
  (RESEARCH_NOTES 7번에서 폐기된 건 "3다리 rigid weld"였다).
- 조각 사이에는 굽힘(판 로컬 y축) + 비틀림(판 로컬 x축) 스프링 힌지. 강성은
  "체중 절반이 걸렸을 때 몇 도 휘는가"로 역산한다 — 초파리 스케일이라 실제 스키
  재료 강성을 그대로 쓰면 사실상 안 휜다. 캠버/로커는 굽힘 힌지의 springref로,
  사이드컷은 조각별 폭으로 표현한다.
- 판 축은 스키 스탠스에서 T3→T1 발끝을 잇는 선 방향, 판 윗면은 발끝 높이.
"""
from __future__ import annotations

import mujoco
import numpy as np
from dm_control import mjcf

from flyski_sim.ski_attachment import SkiUnit, _find_claw_tip_offset
from flyski_sim.ski_profiles import SkiProfile
from flyski_sim.ski_stance import compute_ski_stance

SIDES = ('left', 'right')
N_SEGMENTS = None         # None이면 조각 길이 ~_SEG_LEN_CM이 되게 홀수 개(가운데 조각이 root).
_SEG_LEN_CM = 0.062
# 판 길이 = T1~T3 발끝 거리 + OVERHANG_SCALE x 프로파일 길이. T1/T2/T3 세 점이 판을 잡아
# 휨을 다리가 정하므로(스케이트에 가까움, RESEARCH_NOTES 38번) 발 밖으로 나온 팁/테일이
# 자유롭게 휘도록 길게 한다. 쐐기에서 좌우 판 팁이 닿지 않는 범위에서 고름.
OVERHANG_SCALE = 3.5         # 판 0.43 -> 0.81cm(all_mountain). 길이 스윕은 38번.
HALF_THICKNESS_CM = 0.004
BINDING_HEIGHT_CM = 0.01
_THETA_MAX_EXTRA_DEG = 8.0  # stiffness=0일 때 추가로 허용하는 휨각.
_CAMBER_DEG_PER_HINGE = 1.5
_TARGET_OMEGA = 1000.     # 힌지 고유진동수 상한(rad/s) — physics dt(2e-4s)에서 안정.
_HINGE_RANGE_RAD = 0.35   # 굽힘/비틀림 힌지 가동범위 ±20°.
_BIND_RANGE_RAD = 0.8     # T2 바인딩(부츠) 앞뒤 굽힘 최대 ~46°.
_BOOT_LATERAL_RATIO = 30.  # 부츠 옆/요 강성 = 앞뒤 강성 x 이 값.
_BOOT_ROLL_RANGE_RAD = 0.6  # 부츠 롤 가동범위 ±34°(carving_test가 springref로 판을 세움).
_LEG_BODY_PREFIXES = ('coxa', 'femur', 'tibia', 'tarsus', 'claw')
# 설면에서 잘 보이는 형광색. 좌/우를 다르게 해서 렌더에서 구분되게.
# 모서리 캡슐 필렛. 기본 끔: 반지름 = 반두께(완전히 둥근 엣지)면 판이 엣지로 서지 못하고
# 굴러 누워(엣지 20° 목표에 1.9°) 팽이처럼 돌았다. MuJoCo 마찰은 모서리 모양과 무관해서
# 각진 모서리가 "걸리는" 현상도 없다(RESEARCH_NOTES 37번). 부분 필렛은 메시가 필요.
USE_FILLET = False
# 바인딩(허리)을 더 잘록하게: 프로파일 사이드컷 깊이 x 이 값(사용자 요청, 37번).
_SIDECUT_DEPTH_SCALE = 2.0
COUPLE_BENDING = True      # 판 전체 굽힘/비틀림을 한 값으로 묶음(볼록성).
_PLATE_RGBA = {'left': (0.45, 1.0, 0.05, 1.0), 'right': (1.0, 0.15, 0.75, 1.0)}


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(float))
    return q


def _rest_pose_info(walker, joint_angles: dict[str, float] | None = None):
    """qpos0(+joint_angles로 덮어쓴 관절)에서 thorax 기준 발끝 위치, claw 월드 포즈를
    독립 컴파일로 구한다."""
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    for name, angle in (joint_angles or {}).items():
        physics.named.data.qpos[name] = angle
    physics.forward()
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


def _waist_cm(profile: SkiProfile) -> float:
    """허리(바인딩) 폭. 프로파일 사이드컷 깊이를 _SIDECUT_DEPTH_SCALE배로 더 잘록하게."""
    mean_end = (profile.width_tip_cm + profile.width_tail_cm) / 2
    return mean_end - _SIDECUT_DEPTH_SCALE * (mean_end - profile.width_waist_cm)


def _width_at(profile: SkiProfile, s: float) -> float:
    """s∈[-1(테일), +1(팁)]에서의 스키 폭 — 사이드컷을 2차 보간."""
    waist = _waist_cm(profile)
    end = profile.width_tip_cm if s >= 0 else profile.width_tail_cm
    return waist + (end - waist) * s * s


def plate_sidecut_radius_cm(profile: SkiProfile, length_cm: float) -> float:
    """판 길이와 팁/허리/테일 폭에서 구한 사이드컷 반경 R = L^2 / (8 * 사이드컷 깊이).

    깊이 = ((팁 폭 + 테일 폭)/2 - 허리 폭)/2. 프로파일의 `sidecut_radius_cm`(0.45~1.1cm)는
    옛 미니스키(길이 0.12~0.22cm) 기준 설계 상수라 지금 판(~0.4cm) 형상과 맞지 않는다.
    """
    depth = ((profile.width_tip_cm + profile.width_tail_cm) / 2 - _waist_cm(profile)) / 2
    return float(length_cm ** 2 / (8 * max(depth, 1e-6)))


def attach_side_plates(walker, profile: SkiProfile,
                       n_segments: int | None = N_SEGMENTS) -> dict[str, SkiUnit]:
    """walker의 좌/우 다리 3개씩에 휘는 스키 플레이트를 하나씩 붙인다."""
    info, R_th, x_th, weight = _rest_pose_info(walker, compute_ski_stance(walker))
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
        # 판 중심 = T2 발끝(가운데 다리 하중이 판 한가운데를 누르도록). 예전엔 T1/T3
        # 중점이라 T2가 판 옆 0.077cm(판 폭 밖)에 있었다(RESEARCH_NOTES 34번).
        mid = t2['tip_th']
        # 판 윗면을 발끝보다 BINDING_HEIGHT_CM 아래에 둔다(부츠/바인딩 높이). 발톱-판
        # 충돌은 exclude라서, 바인딩이 조금만 처져도 발톱이 판을 통과해 눈(마찰 1.0)에
        # 닿아 브레이크가 걸렸다(RESEARCH_NOTES 32번).
        center_th = np.array([mid[0], mid[1],
                              tip_z - BINDING_HEIGHT_CM - HALF_THICKNESS_CM])

        length = span + OVERHANG_SCALE * profile.length_cm
        if n_segments is None:
            n_segments = 2 * max(int(round(length / _SEG_LEN_CM / 2)), 1) + 1
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
            half_w = _width_at(profile, s) / 2
            r = HALF_THICKNESS_CM
            # 필렛: 가운데 얇은 box + 양쪽 긴 모서리 캡슐(반지름 = 판 반두께). 옆 엣지와
            # 조각 끝이 둥글어서 직진/정지 때 각진 모서리가 설면에 걸리지 않는다. 엣지
            # 그립은 형상이 아니라 엣지각 기반 모델(edge_grip.py)이라 필렛이 그립을 줄이지는
            # 않는다(실제 스키라면 둥근 엣지는 덜 박힘).
            # priority=1: 스키-지면 접촉은 지면이 아니라 이 geom의 마찰(설질별 ski_friction,
            # 매 스텝 갱신)을 쓴다.
            common = dict(contype=1, conaffinity=1, condim=3, priority=1,
                          friction=(0.05, 0.005, 0.0001), rgba=_PLATE_RGBA[side])
            if not USE_FILLET and idx == n_segments - 1:
                # 팁 조각: 위에서 봤을 때 둥근 헤드. 앞쪽 half_w 만큼을 세로축 원판으로
                # 바꿔 평면상 반원 끝이 되게 한다(옆 엣지는 각진 그대로).
                body.add('geom', name=f'ski_geom_{side}_s{idx}', type='box',
                         size=((seg_len - half_w) / 2, half_w, r), pos=(-half_w / 2, 0., 0.),
                         mass=0.7 * seg_mass, **common)
                body.add('geom', name=f'ski_geom_{side}_s{idx}_head', type='cylinder',
                         size=(half_w, r), pos=(seg_len / 2 - half_w, 0., 0.),
                         mass=0.3 * seg_mass, **common)
                bodies.append(body)
                geoms.extend(body.find_all('geom'))
                return body
            if not USE_FILLET:
                body.add('geom', name=f'ski_geom_{side}_s{idx}', type='box',
                         size=(seg_len / 2, half_w, r), mass=seg_mass, **common)
                bodies.append(body)
                geoms.extend(body.find_all('geom'))
                return body
            body.add('geom', name=f'ski_geom_{side}_s{idx}', type='box',
                     size=(seg_len / 2, half_w - r, r), mass=0.6 * seg_mass, **common)
            for k, y in (('L', half_w - r), ('R', -(half_w - r))):
                body.add('geom', name=f'ski_geom_{side}_s{idx}_{k}', type='capsule',
                         fromto=(-seg_len / 2, y, 0., seg_len / 2, y, 0.), size=(r,),
                         mass=0.2 * seg_mass, **common)
            bodies.append(body)
            geoms.extend(body.find_all('geom'))
            return body

        root = add_segment(t2['claw'], m, root_pos, root_quat)
        # T2 바인딩 = 스키 부츠: 발끝을 피벗으로 한 힌지 3개. 앞뒤(pitch)는 예전 ball
        # 조인트 강성 그대로, 옆(roll)/요(yaw)는 _BOOT_LATERAL_RATIO배 단단하다.
        # 스탠스에서 T1/T2/T3 발끝이 판 축 위 일직선이라 T1/T3 점 연결은 판 축 둘레
        # 롤을 전혀 못 잡는다 → 등방 ball이면 몸을 20° 기울여도 판은 1.4°만 섰다
        # (엣징 전달 불가, RESEARCH_NOTES 35번).
        t2_tip_plate = R_plate_w.T @ (x_c + R_c @ t2['tip_off'] - center_w)
        k_bind = 2 * k_bend * profile.attachment_compliance
        for dof, axis, ratio, rng in (('pitch', (0, 1, 0), 1., _BIND_RANGE_RAD),
                                      ('roll', (1, 0, 0), _BOOT_LATERAL_RATIO, _BOOT_ROLL_RANGE_RAD),
                                      ('yaw', (0, 0, 1), _BOOT_LATERAL_RATIO, _HINGE_RANGE_RAD)):
            root.add('joint', name=f'ski_bind_{side}_{dof}', type='hinge', axis=axis,
                     pos=tuple(t2_tip_plate), **hinge_kw(k_bind * ratio, rng=(-rng, rng)))

        seg_by_idx = {m: root}
        chain = {}                                            # (굽힘/비틀림, 방향) -> 힌지들.
        for direction in (+1, -1):
            parent = root
            for step in range(1, m + 1):
                idx = m + direction * step
                body = add_segment(parent, idx, (direction * seg_len, 0., 0.))
                hinge_pos = (-direction * seg_len / 2, 0., 0.)
                # 캠버(+)=팁/테일이 아래로, 로커(-)=위로. +y 회전은 +x 끝을 내린다.
                ref = direction * profile.rocker_camber * np.deg2rad(_CAMBER_DEG_PER_HINGE)
                chain.setdefault(('bend', direction), []).append(body.add(
                    'joint', name=f'ski_bend_{side}_s{idx}', type='hinge',
                    axis=(0, 1, 0), pos=hinge_pos, springref=float(ref), **hinge_kw(k_bend)))
                chain.setdefault(('twist', direction), []).append(body.add(
                    'joint', name=f'ski_twist_{side}_s{idx}', type='hinge',
                    axis=(1, 0, 0), pos=hinge_pos, **hinge_kw(k_tors)))
                seg_by_idx[idx] = body
                parent = body

        # 볼록성: 힌지가 제각각 꺾이면 판이 지그재그(비볼록)가 됐다(평지/플루크/엣징 모두
        # 스텝의 99~100%, RESEARCH_NOTES 37번). 구간마다 굽힘/비틀림 힌지를 한 값으로
        # 묶어 구간 안에서는 곡률(비틀림률)이 일정하게 한다.
        # - 발 구간(T3~T1 발 사이): 한 값. 절반씩 따로 두면 플루크에서 T1/T3가 팁과 테일을
        #   반대로 밀어 S자가 됐다. 테일 쪽 힌지는 부모가 +x 쪽이라 같은 모양에서 각의
        #   부호가 팁 쪽과 반대(캠버 springref도 +/- 대칭).
        # - 팁 돌출부, 테일 돌출부(발 밖): 각각 따로 한 값. 발이 잡지 않는 부분이라 설면
        #   하중으로 자유롭게 휜다(38번).
        # 판 구속이 부드러우면(solref 0.002) 판 힌지 관성이 작아 크게 어긋나서 하드하게.
        foot_step = max(int(round(abs(float((t1['tip_th'] - t2['tip_th']) @ ex)) / seg_len)), 1)
        hard = dict(solref=(0.0004, 1.), solimp=(0.99, 0.999, 0.001, 0.5, 2.))
        for kind in (('bend', 'twist') if COUPLE_BENDING else ()):
            groups = [[(j, d) for d in (+1, -1) for j in chain[(kind, d)][:foot_step]]]
            for d in (+1, -1):
                over = chain[(kind, d)][foot_step:]
                if over:
                    groups.append([(j, +1) for j in over])     # 같은 방향끼리라 부호 같음.
            for group in groups:
                leader, lead_sign = group[0]
                for follower, sign in group[1:]:
                    model.equality.add('joint', name=f'ski_{kind}_couple_{follower.name}',
                                       joint1=follower, joint2=leader,
                                       polycoef=(0., float(sign * lead_sign), 0., 0., 0.), **hard)

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
            # site 기반 connect: body1+anchor 방식은 판 쪽 anchor를 qpos0(기본 자세)에서
            # 자동 계산해서, 스키 스탠스로 만든 판을 기본 자세 발끝 쪽으로 끌어당겼다
            # (판 코가 들리고 앞발이 눈에 박힘, RESEARCH_NOTES 34번).
            site_foot = t['claw'].add('site', name=f'ski_bind_foot_{leg}_{side}',
                                      pos=tuple(t['tip_off']), size=(0.002,), group=5)
            site_plate = seg_by_idx[idx].add(
                'site', name=f'ski_bind_plate_{leg}_{side}', size=(0.002,), group=5,
                pos=tuple(tip_plate - np.array([(idx - m) * seg_len, 0., 0.])))
            model.equality.add('connect', name=f'ski_bind_{leg}_{side}',
                               site1=site_foot, site2=site_plate, solref=(timeconst, 1.0))
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
                              half_length_cm=length / 2, area_cm2=float(area),
                              sidecut_radius_cm=plate_sidecut_radius_cm(profile, length))
    return units
