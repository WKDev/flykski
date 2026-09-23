"""업무4: LIF 다리 운동뉴런 층 -> flybody 관절 액추에이터 디코딩 (1차 근사).

모터뉴런 타입명(고전 곤충 다리근 명명법: promotor/remotor, reductor,
flexor/extensor, levator/depressor)을 flybody의 다리 DOF 액추에이터로
키워드 매칭한다.

**중요한 한계(명시적 근사)**:
1. `superclass=='vnc_motor'`로 뽑은 360개 중 다리근으로 이름을 알아볼 수 있는
   것만 매핑했다 — 나머지(`MNad*`, `MNhl*`, `DLMn`, `DVMn` 등)는 비행/기타
   흉부 운동뉴런으로 추정되어 제외했다(RESEARCH_NOTES.md 20번 참고).
2. "근육이 어느 관절에 작용하는지"는 표준 곤충 다리 근육학 명명 관례로
   추정한 것이며, male-cns 데이터로 직접 검증하지 않았다.
3. **(2026-09-23 수정)** 이제 CSV에 `somaSide`(L/R) 컬럼이 포함돼 있어서,
   실제로 좌/우를 구분해서 해당 측 액추에이터에만 신호를 보낸다(예전엔 양쪽에
   동일 신호를 복사했었다 — RESEARCH_NOTES.md 14/20/25번에서 세 번 반복 지적된
   문제). neuprint의 `somaSide` 'L'/'R'이 flybody의 'left'/'right' 명명과
   그대로 대응한다고 가정했다(직접 검증은 안 함 — 다음 확인 대상).
"""
from __future__ import annotations

import numpy as np

# type명 -> (flybody DOF 카테고리, 부호). 카테고리는 flybody 액추에이터 접미사와
# 대응(coxa_twist, coxa_abduct, coxa, femur, tibia).
_TYPE_TO_DOF_SIGN = {
    'Sternal anterior rotator MN': ('coxa_twist', 1.0),
    'Sternal posterior rotator MN': ('coxa_twist', -1.0),
    'Tergopleural/Pleural promotor MN': ('coxa_twist', 1.0),
    'Tergotr. MN': ('coxa_twist', 0.5),
    'Pleural remotor/abductor MN': ('coxa_abduct', -1.0),
    'Sternal adductor MN': ('coxa_abduct', 1.0),
    'Sternotrochanter MN': ('coxa', 1.0),
    'Fe reductor MN': ('coxa', 1.0),
    'Tr flexor MN': ('coxa', 1.0),
    'Tr extensor MN': ('coxa', -1.0),
    'Acc. tr flexor MN': ('coxa', 0.5),
    'Ti flexor MN': ('femur', 1.0),
    'Ti extensor MN': ('femur', -1.0),
    'Acc. ti flexor MN': ('femur', 0.5),
    'ltm1-tibia MN': ('femur', 0.3),
    'ltm2-femur MN': ('femur', 0.3),
    'Ta depressor MN': ('tibia', 1.0),
    'Ta levator MN': ('tibia', -1.0),
}

_DOF_CATEGORIES = ('coxa_twist', 'coxa_abduct', 'coxa', 'femur', 'tibia')
_SEGMENTS = ('T1', 'T2', 'T3')
_SIDES = ('left', 'right')
# neuprint somaSide -> flybody 액추에이터 접미사. 직접 검증 안 된 가정(위 참고).
_SIDE_MAP = {'L': 'left', 'R': 'right'}


def build_motor_neuron_dof_map(motor_meta) -> dict[int, tuple[str, str, float, str | None]]:
    """bodyid -> (segment, dof_category, sign, side) 매핑을 만든다.

    Args:
        motor_meta: `03_leg_motor_neurons.csv`를 읽은 DataFrame
            (columns: bodyid, type, somaNeuromere, predictedNt, somaSide, ...).
            `somaSide`가 없는 구버전 CSV라도 동작한다(그 경우 side=None,
            좌우 둘 다에 신호를 보내는 예전 방식으로 자동 폴백).
    """
    has_side = 'somaSide' in motor_meta.columns
    mapping = {}
    for _, row in motor_meta.iterrows():
        dof_sign = _TYPE_TO_DOF_SIGN.get(row['type'])
        seg = row['somaNeuromere']
        if dof_sign is None or seg not in _SEGMENTS:
            continue
        dof, sign = dof_sign
        side = _SIDE_MAP.get(row['somaSide']) if has_side else None
        mapping[row['bodyid']] = (seg, dof, sign, side)
    return mapping


def decode_to_flybody_action(efferent_activity: np.ndarray,
                             motor_body_ids: list[int],
                             dof_map: dict[int, tuple[str, str, float, str | None]],
                             actuator_names: list[str],
                             gain: float = 1.0) -> np.ndarray:
    """LIF 효과기(efferent) 활성 벡터를 flybody 액션 벡터로 디코딩한다.

    Args:
        efferent_activity: LIFConnectomeController.step()의 반환값,
            motor_body_ids와 같은 순서.
        motor_body_ids: efferent_activity의 각 성분에 대응하는 bodyId 목록.
        dof_map: build_motor_neuron_dof_map()의 결과.
        actuator_names: flybody env의 실제 액추에이터 이름 목록(env.action_spec
            차원과 같은 순서) — 하드코딩 금지 원칙 준수.
        gain: 디코딩 출력에 곱하는 전체 스케일(추후 학습 대상으로 승격 가능).

    Returns:
        action: actuator_names와 같은 길이의 배열, [-1, 1] 범위로 클립됨.
    """
    # (segment, dof, side) -> 누적 신호. side가 None이면(구버전 데이터) 양쪽에
    # 다 더할 대상으로 별도 처리.
    accum: dict[tuple[str, str, str | None], float] = {}
    for bid, act in zip(motor_body_ids, efferent_activity):
        entry = dof_map.get(bid)
        if entry is None:
            continue
        seg, dof, sign, side = entry
        key = (seg, dof, side)
        accum[key] = accum.get(key, 0.0) + sign * act

    action = np.zeros(len(actuator_names))
    name_to_idx = {n: i for i, n in enumerate(actuator_names)}
    for (seg, dof, side), value in accum.items():
        target_sides = _SIDES if side is None else (side,)
        for s in target_sides:
            actuator_name = f'walker/{dof}_{seg}_{s}'
            idx = name_to_idx.get(actuator_name)
            if idx is not None:
                action[idx] += gain * value
    return np.clip(action, -1.0, 1.0)
