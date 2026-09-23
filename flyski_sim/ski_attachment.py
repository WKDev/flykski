"""업무2/3: 스키를 flybody 다리(claw)에 compliant 하게 부착한다.

02_ski_profiles.md의 기본안(N=6, 다리당 미니 스키 1개, compliant 연결)을 구현한다.
좌/우 트라이포드를 하나의 스키에 rigid weld 하는 구버전 설계는 과구속으로 폐기됨
(RESEARCH_NOTES.md 7번, codex 교차검증). N=6는 다리마다 독립이라 애초에 과구속
문제가 없지만, "다리 관절 가동범위와 충돌할 수 있다"는 지적은 여전히 유효해서
rigid 대신 스프링형 ball 조인트로 붙인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flyski_sim.ski_profiles import SkiProfile

LEG_IDS = [f'T{i}_{side}' for i in (1, 2, 3) for side in ('left', 'right')]


@dataclass
class SkiUnit:
    """텔레메트리/항력이 공통으로 쓰는 '스키 한 짝' 기술자.

    per-leg(N=6, 이 모듈)이면 body 1개, side_plate(N=2, ski_plate.py)면 조각 K개.
    """
    name: str
    root_body: object                 # mjcf body 엘리먼트 (엣지각 기준 프레임).
    bodies: list = field(default_factory=list)
    geoms: list = field(default_factory=list)
    half_length_cm: float = 0.
    area_cm2: float = 0.
    sidecut_radius_cm: float = 0.     # side_plate만: 판 실제 형상에서 구한 사이드컷 반경.


def claw_body_name(leg_id: str) -> str:
    return f'claw_{leg_id}'


def _get_local_quat(body) -> np.ndarray:
    q = body.quat
    return np.array([1., 0., 0., 0.]) if q is None else np.array(q, dtype=float)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _claw_quat_relative_to_thorax(claw_body) -> np.ndarray:
    """thorax(몸통) 기준 claw의 rest-pose 상대 회전(quat)을 정적으로 계산한다.

    **왜 필요한가(RESEARCH_NOTES.md 14~15번)**: 다리마다 로컬 프레임 회전이
    제각각이라(예: claw_T1_left의 world z축이 [0.70,0.54,0.46]로 수직과 전혀
    다름), 스키를 claw에 그냥 붙이면 "로컬 Z축 = 발바닥 법선"이라는 가정이
    다리마다 다르게 깨진다. 부착 시점에 이 상대회전의 역방향을 스키 body의
    quat으로 걸어주면, 최소한 rest pose에서는 스키 로컬 Z축이 몸통(thorax)의
    '위' 방향과 정렬된다.
    """
    q = np.array([1., 0., 0., 0.])
    node = claw_body
    while node is not None and node.tag == 'body' and node.name != 'thorax':
        q = _quat_mul(_get_local_quat(node), q)
        node = node.parent
    return q


def _find_claw_tip_offset(claw_body) -> np.ndarray:
    """claw_body 안의 발톱 collision capsule geom의 `fromto` 끝점(발끝 위치,
    claw_body 로컬 좌표)을 반환한다.

    **중요(삽질 기록, RESEARCH_NOTES.md 14번)**: claw_body 원점은 발끝이 아니다 —
    실제 발끝은 그 안에 있는 `tarsal_claw_*_collision` capsule geom의 `fromto`
    끝점이다. 처음엔 `pos=(0,0,-drop)`(claw 로컬 -Z)로 스키를 붙였는데, 다리마다
    로컬 프레임 회전이 달라서 월드 기준 '아래'가 아니라 대각선으로 붙어버려
    스키가 땅에 전혀 안 닿는 채로 '안정적으로' 서 있었다(사실상 장식품이었음).
    """
    for geom in claw_body.get_children('geom'):
        if geom.fromto is not None:
            return np.array(geom.fromto[3:6], dtype=float)
    return np.zeros(3)


def attach_ski_to_claw(claw_body, profile: SkiProfile, leg_id: str):
    """claw_body(mjcf 엘리먼트)에 미니 스키를 compliant ball 조인트로 부착한다.

    스키는 발끝(claw tip)에서, 발이 이미 향하고 있던 방향으로 조금 더 이어지도록
    붙인다(claw tip 자체가 원래 지면과 닿던 지점이므로, 그 연장선에 스키를 두면
    방향에 상관없이 실제로 지면과 접촉한다).

    Returns:
        생성된 스키 body의 mjcf 엘리먼트.
    """
    tip_offset = _find_claw_tip_offset(claw_body)
    norm = np.linalg.norm(tip_offset)
    direction = tip_offset / norm if norm > 1e-9 else np.array([0., 0., -1.])
    drop = profile.length_cm * 0.15
    ski_pos = tip_offset + direction * drop

    # rest pose 기준 claw의 상대회전을 상쇄하는 quat을 걸어서, 스키 로컬 Z축이
    # (적어도 rest pose에서는) 몸통 '위' 방향과 정렬되게 한다.
    claw_rel_quat = _claw_quat_relative_to_thorax(claw_body)
    ski_quat = _quat_conj(claw_rel_quat)

    ski_body = claw_body.add('body', name=f'ski_{leg_id}', pos=tuple(ski_pos),
                             quat=tuple(ski_quat))

    # attachment_compliance가 낮을수록(모글 스키 등) 더 물렁하게 붙는다.
    stiffness = 50.0 * profile.attachment_compliance
    damping = 2.0 * profile.attachment_compliance
    ski_body.add('joint', name=f'ski_joint_{leg_id}', type='ball',
                 stiffness=stiffness, damping=damping, frictionloss=0.0)

    half_length = profile.length_cm / 2
    half_width = profile.width_waist_cm / 2
    thickness = 0.01
    # fruitfly.xml의 기본 geom class(class="body")는 contype/conaffinity=0인
    # 순수 시각용 메시라서, 아무 class도 안 주면 이 geom이 아무와도 충돌하지
    # 않는 장식품이 돼버린다(RESEARCH_NOTES.md 14번) — 명시적으로 켜준다.
    ski_body.add('geom', name=f'ski_geom_{leg_id}', type='box',
                 size=(half_length, half_width, thickness),
                 pos=(0, 0, 0),
                 mass=profile.mass_g,
                 contype=1, conaffinity=1, condim=3,
                 priority=1, friction=(0.05, 0.005, 0.0001),
                 rgba=(0.7, 0.85, 1.0, 0.9))
    return ski_body


def attach_skis_to_walker(walker, profile: SkiProfile) -> dict[str, SkiUnit]:
    """워커의 활성화된 claw 각각에 profile 스키를 부착한다(레거시 N=6 레이아웃).

    다리가 비활성화된 설정(use_legs=False)이면 claw 바디 자체가 없으므로
    자동으로 건너뛴다.

    Returns:
        {leg_id: SkiUnit(body 1개)}.
    """
    skis = {}
    for leg_id in LEG_IDS:
        claw = walker.mjcf_model.find('body', claw_body_name(leg_id))
        if claw is None:
            continue
        body = attach_ski_to_claw(claw, profile, leg_id)
        skis[leg_id] = SkiUnit(name=leg_id, root_body=body, bodies=[body],
                               geoms=body.get_children('geom'),
                               half_length_cm=profile.length_cm / 2,
                               area_cm2=profile.base_surface_area_cm2)
    return skis
