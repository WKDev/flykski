"""업무1: 설질(snow quality) 프리셋 정의.

01_infinite_slope_terrain.md 3절의 5종 프리셋을 MuJoCo 접촉 파라미터로 옮긴다.
MuJoCo geom.friction은 (sliding, torsional, rolling) 튜플이지만, flybody 자체
관례(`tasks/base.py`의 `geom.friction = (0.5,)`)를 따라 sliding 성분만 지정한다.

수치는 실측 눈 물성 데이터가 아니라 "설질 간 상대적 차이가 방향성 있게 나야 한다"는
요구사항을 만족시키기 위한 1차 근사치다 — 실측/게임플레이 튜닝은 후속 과제.

`drag_coefficient`는 `force = -drag_coefficient * velocity / base_surface_area_cm2`
형태로 쓰인다(flyski_sim/tasks.py `_apply_snow_drag`). 스키 접지 면적이
0.002~0.007cm² 수준으로 아주 작아서, 처음에 사람이 이해하기 쉬운 크기(0.01~0.35)로
넣었더니 힘이 수백 배로 증폭돼 물리가 바로 발산했다(RESEARCH_NOTES.md 16번) —
그래서 실측 접촉력 크기(cfrc_ext ~0.1~0.3)와 비슷한 수준이 나오도록 1e-5~3e-4
스케일로 역산해서 넣었다.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnowParams:
    friction: float  # sliding friction 계수.
    solref: tuple[float, float]  # (time constant, damping ratio).
    # (dmin, dmax, width, midpoint, power) — MuJoCo solimp는 5개 성분.
    solimp: tuple[float, float, float, float, float]
    drag_coefficient: float  # "빠지는" 느낌 — 속도 비례 보조 항력 계수 (xfrc_applied).
    # 스키 바닥(base)-눈 미끄럼 마찰. `friction`은 발톱/몸통-눈 접촉용(0.15~0.75)이라
    # 그대로 스키에 쓰면 전부 tan(20°)=0.36보다 커서 스키가 경사에서 안 미끄러졌다
    # (RESEARCH_NOTES 32번). 실제 스키-눈 운동마찰은 대략 0.02~0.15. 스키 geom은
    # priority=1이라 이 값이 지면 마찰보다 우선한다(tasks.py `_apply_snow_to_skis`).
    ski_friction: float = 0.05


_MID_POWER = (0.5, 2.0)  # MuJoCo 기본 midpoint/power, 전 프리셋 공통.

SNOW_PRESETS: dict[str, SnowParams] = {
    # 아이스: 마찰 최저, 접촉이 거의 강체(짧은 time constant), 침투 저항 거의 없음.
    'ice': SnowParams(friction=0.15, solref=(0.001, 1.0),
                      solimp=(0.95, 0.99, 0.01) + _MID_POWER,
                      drag_coefficient=1e-5, ski_friction=0.02),
    # 파우더: 마찰은 낮은 편이지만 접촉이 물러서(긴 time constant) "빠지는" 느낌 — drag 최대.
    'powder': SnowParams(friction=0.40, solref=(0.02, 1.2),
                         solimp=(0.85, 0.95, 0.02) + _MID_POWER,
                         drag_coefficient=3e-4, ski_friction=0.08),
    # 패킹파우더: flybody Walking 기본값과 동일한 기준선.
    'packed_powder': SnowParams(friction=0.50, solref=(0.005, 1.0),
                                solimp=(0.95, 0.99, 0.01) + _MID_POWER,
                                drag_coefficient=4e-5, ski_friction=0.05),
    # 크러드: 불균일 설질의 대표값(공간적 노이즈는 후속 과제, 지금은 단일 대표치).
    'crud': SnowParams(friction=0.55, solref=(0.008, 1.0),
                       solimp=(0.90, 0.96, 0.02) + _MID_POWER,
                       drag_coefficient=1.3e-4, ski_friction=0.10),
    # 슬러시: 마찰 최고(끈적함), 댐핑도 크고 drag도 중상.
    'slush': SnowParams(friction=0.75, solref=(0.01, 1.0),
                        solimp=(0.90, 0.97, 0.02) + _MID_POWER,
                        drag_coefficient=2.2e-4, ski_friction=0.15),
}

SNOW_TYPES = tuple(SNOW_PRESETS.keys())
