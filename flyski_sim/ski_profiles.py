"""업무2: flybody 다리 부착형 스키 프로파일.

02_ski_profiles.md의 3~5종 프로파일 + 물리 파라미터 스키마를 구현한다.
길이/질량은 flybody 초파리 모델 스케일(다리 tarsus 세그먼트 ~0.01~0.03 "cm" 단위,
몸길이 ~0.3 "cm")에 맞춘 1차 근사치다 — 실측 게임플레이 튜닝은 후속 과제.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SkiProfile:
    name: str
    length_cm: float
    width_tip_cm: float
    width_waist_cm: float
    width_tail_cm: float
    sidecut_radius_cm: float
    flex_stiffness: float        # 상대 단위 [0,1], 낮을수록 부드러움(모글 충격 흡수).
    torsional_stiffness: float   # 상대 단위 [0,1], 높을수록 엣지 그립 유지.
    mass_g: float
    rocker_camber: float         # -1(강한 로커/파우더용) ~ +1(강한 캠버), 0=중립.
    attachment_compliance: float  # 부착 조인트 강성 배율 — 낮을수록 물렁하게 붙음.

    @property
    def base_surface_area_cm2(self) -> float:
        """대략적인 접지 면적 근사(사다리꼴 3구간 평균)."""
        avg_width = (self.width_tip_cm + 2 * self.width_waist_cm +
                    self.width_tail_cm) / 4
        return self.length_cm * avg_width


SKI_PROFILES: dict[str, SkiProfile] = {
    'slalom': SkiProfile(
        name='slalom', length_cm=0.12, width_tip_cm=0.020, width_waist_cm=0.014,
        width_tail_cm=0.018, sidecut_radius_cm=0.45,
        flex_stiffness=0.9, torsional_stiffness=0.9,
        mass_g=3e-5, rocker_camber=0.3, attachment_compliance=1.0),
    'giant_slalom': SkiProfile(
        name='giant_slalom', length_cm=0.18, width_tip_cm=0.022, width_waist_cm=0.016,
        width_tail_cm=0.020, sidecut_radius_cm=0.85,
        flex_stiffness=0.7, torsional_stiffness=0.7,
        mass_g=4.5e-5, rocker_camber=0.15, attachment_compliance=1.0),
    'all_mountain': SkiProfile(
        name='all_mountain', length_cm=0.15, width_tip_cm=0.024, width_waist_cm=0.018,
        width_tail_cm=0.022, sidecut_radius_cm=0.65,
        flex_stiffness=0.5, torsional_stiffness=0.5,
        mass_g=4e-5, rocker_camber=0.0, attachment_compliance=1.0),
    'powder': SkiProfile(
        name='powder', length_cm=0.22, width_tip_cm=0.034, width_waist_cm=0.028,
        width_tail_cm=0.030, sidecut_radius_cm=1.1,
        flex_stiffness=0.45, torsional_stiffness=0.4,
        mass_g=6e-5, rocker_camber=0.7, attachment_compliance=1.0),
    'mogul': SkiProfile(
        name='mogul', length_cm=0.13, width_tip_cm=0.020, width_waist_cm=0.015,
        width_tail_cm=0.018, sidecut_radius_cm=0.55,
        flex_stiffness=0.25, torsional_stiffness=0.35,
        mass_g=3.5e-5, rocker_camber=0.1, attachment_compliance=0.6),
}


def compute_turn_radius(profile: SkiProfile,
                        effective_edge_angle_deg: float,
                        speed_cm_s: float) -> float:
    """카빙 회전 반경(cm) 근사치.

    사이드컷 반경을 기준으로, 엣지각이 커질수록(스키를 더 세울수록) 회전 반경이
    작아지는 표준 카빙 근사식(R_turn ≈ R_sidecut * cos(edge_angle))을 사용한다.
    speed_cm_s는 현재 이 근사식에서 쓰지 않지만(순수 기하학적 카빙 근사), 추후
    원심력/슬립 모델을 추가할 때 확장 지점으로 인터페이스에 남겨둔다.
    """
    del speed_cm_s  # 현재 미사용 — 향후 슬립 모델 확장 지점.
    edge_rad = np.deg2rad(np.clip(effective_edge_angle_deg, 0.1, 89.))
    return float(profile.sidecut_radius_cm * np.cos(edge_rad))


def compute_penetration_cm(profile: SkiProfile,
                           snow_penetration_softness: float,
                           normal_load_scale: float = 1.0) -> float:
    """설질 침투 깊이(cm) 근사치 — 접지 면적이 넓을수록(파우더 스키) 덜 빠진다.

    snow_penetration_softness: 업무1 설질 프리셋에서 오는 상대적 '물렁함' 지표
        (0=거의 안 빠짐(아이스), 1=많이 빠짐(파우더)).
    """
    return float(snow_penetration_softness * normal_load_scale /
                max(profile.base_surface_area_cm2, 1e-6))


def get_profile(name: str) -> SkiProfile:
    if name not in SKI_PROFILES:
        raise ValueError(
            f"Unknown ski profile {name!r}. Options: {list(SKI_PROFILES)}")
    return SKI_PROFILES[name]
