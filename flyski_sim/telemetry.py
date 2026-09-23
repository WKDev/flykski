"""업무3: 스키 파생 텔레메트리 (엣지각, 하중배분, 전복 판정).

03_ski_attachment_actuation.md의 설계 원칙: 액션 공간은 flybody 네이티브 관절
액추에이터이고, 엣지각/하중배분 같은 "스키 지표"는 그 결과에서 역산(derive)한
관찰/분석 전용 값이다 — 별도 상위 제어 API가 아니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SkiTelemetry:
    effective_edge_angle_deg: dict[str, float]
    plate_load_share: dict[str, float]
    fore_aft_pressure_center: dict[str, float]  # -1(테일)~+1(팁), 0=중립/무접촉.
    com_xy: tuple[float, float]
    support_polygon_xy: list[tuple[float, float]]
    com_over_support_polygon: bool
    body_tilt_deg: float
    is_fallen: bool


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew 모노톤 체인 convex hull (반시계 순서)."""
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _point_in_convex_polygon(point: np.ndarray, polygon_pts: list[np.ndarray]) -> bool:
    """polygon_pts(정렬 안 된 xy 점들)의 convex hull 안에 point가 있는지 판정."""
    if len(polygon_pts) < 3:
        return False
    # 접촉점이 많으면(side_plate) 내부 점이 섞여 있어서, 각도 정렬만으로는 볼록
    # 다각형이 안 된다 — 진짜 convex hull(모노톤 체인)을 먼저 구한다.
    hull = _convex_hull(np.array(polygon_pts, dtype=float))
    if len(hull) < 3:
        return False
    n = len(hull)
    sign = None
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        edge = b - a
        to_point = np.array(point) - a
        cross = edge[0] * to_point[1] - edge[1] * to_point[0]
        if abs(cross) < 1e-12:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


def compute_ski_telemetry(physics,
                          walker,
                          skis: dict,
                          terrain_normal_fn=None,
                          fall_tilt_threshold_deg: float = 70.) -> SkiTelemetry:
    """현재 물리 상태에서 스키 텔레메트리를 역산한다.

    Args:
        physics: 컴파일된 dm_control Physics.
        walker: flybody FruitFly 워커 엔티티.
        skis: {이름: SkiUnit} — task.skis. side_plate면 'left'/'right'(조각 K개),
            per_leg면 'T1_left' 등(body 1개). 엣지각/압력중심은 각 짝의 root 조각
            프레임 기준, 하중은 짝의 모든 조각 합.
        terrain_normal_fn: (x, y) -> (nx, ny, nz) 지형 법선 함수. None이면
            평지(0,0,1) 가정.
        fall_tilt_threshold_deg: 몸통이 이 각도 이상 기울면 전복으로 간주.
    """
    leg_ids = list(skis.keys())
    ski_body_ids = {leg_id: physics.model.name2id(skis[leg_id].root_body.full_identifier,
                                                  'body')
                    for leg_id in leg_ids}
    unit_body_ids = {leg_id: [physics.model.name2id(b.full_identifier, 'body')
                              for b in skis[leg_id].bodies] for leg_id in leg_ids}
    geom_to_leg = {physics.model.name2id(g.full_identifier, 'geom'): leg_id
                   for leg_id in leg_ids for g in skis[leg_id].geoms}

    # 스키별 접촉점 수집 (fore_aft_pressure_center, support_polygon 용).
    contact_points: dict[str, list[np.ndarray]] = {leg_id: [] for leg_id in leg_ids}
    for i in range(physics.data.ncon):
        c = physics.data.contact[i]
        for gid in (c.geom1, c.geom2):
            leg_id = geom_to_leg.get(int(gid))
            if leg_id is not None:
                contact_points[leg_id].append(np.array(c.pos))

    # 1) plate_load_share: cfrc_ext의 수직(z) 성분(위쪽 반력만) 비율, 조각 합.
    z_forces = {}
    for leg_id, bids in unit_body_ids.items():
        z_forces[leg_id] = sum(max(float(physics.data.cfrc_ext[b, 5]), 0.) for b in bids)
    total_z = sum(z_forces.values())
    plate_load_share = {
        k: (v / total_z if total_z > 1e-9 else 0.) for k, v in z_forces.items()
    }

    # 2) effective_edge_angle_deg: 스키 로컬 Z축(밑면 법선)과 지형 법선 사이 각.
    effective_edge_angle_deg = {}
    for leg_id, bid in ski_body_ids.items():
        body_pos = physics.data.xpos[bid]
        body_mat = physics.data.xmat[bid].reshape(3, 3)
        sole_normal = body_mat[:, 2]
        if terrain_normal_fn is not None:
            terrain_normal = np.array(terrain_normal_fn(body_pos[0], body_pos[1]))
        else:
            terrain_normal = np.array([0., 0., 1.])
        # 두 법선이 대략 같은 쪽(위)을 향하도록 부호 정렬 후 각도 계산.
        if np.dot(sole_normal, terrain_normal) < 0:
            sole_normal = -sole_normal
        cos_a = np.dot(sole_normal, terrain_normal) / (
            np.linalg.norm(sole_normal) * np.linalg.norm(terrain_normal) + 1e-12)
        effective_edge_angle_deg[leg_id] = float(
            np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    # 3) fore_aft_pressure_center: 접촉점들의 스키 로컬 x 평균 위치, 절반길이로 정규화.
    fore_aft_pressure_center = {}
    for leg_id, bid in ski_body_ids.items():
        pts = contact_points[leg_id]
        half_len = skis[leg_id].half_length_cm
        if not pts or half_len <= 0:
            fore_aft_pressure_center[leg_id] = 0.
            continue
        body_pos = physics.data.xpos[bid]
        body_mat = physics.data.xmat[bid].reshape(3, 3)
        local_xs = [body_mat.T @ (p - body_pos) for p in pts]
        avg_x = float(np.mean([lx[0] for lx in local_xs]))
        fore_aft_pressure_center[leg_id] = float(np.clip(avg_x / half_len, -1, 1))

    # 4) 지지기저면 vs 무게중심. 스키 짝이 2개(side_plate)면 짝 중심점만으로는
    # 다각형이 안 되므로, 모든 스키 접촉점의 xy를 지지점으로 쓴다.
    support_polygon_xy = [tuple(p[:2]) for pts in contact_points.values() for p in pts]
    com_xy = tuple(physics.bind(walker.root_body).xpos[:2])
    com_over_support = _point_in_convex_polygon(np.array(com_xy), support_polygon_xy)

    body_mat = physics.bind(walker.root_body).xmat.reshape(3, 3)
    body_up = body_mat[:, 2]
    tilt_deg = float(np.degrees(np.arccos(np.clip(body_up[2], -1, 1))))

    is_fallen = (tilt_deg > fall_tilt_threshold_deg) or (
        len(support_polygon_xy) >= 3 and not com_over_support)

    return SkiTelemetry(
        effective_edge_angle_deg=effective_edge_angle_deg,
        plate_load_share=plate_load_share,
        fore_aft_pressure_center=fore_aft_pressure_center,
        com_xy=com_xy,
        support_polygon_xy=support_polygon_xy,
        com_over_support_polygon=com_over_support,
        body_tilt_deg=tilt_deg,
        is_fallen=is_fallen,
    )
