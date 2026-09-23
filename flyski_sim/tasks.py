"""업무1 1단계 스모크테스트용 태스크.

flybody의 TemplateTask(걷기 태스크 기본형)를 그대로 쓰되, 경사 지형 위에서
초파리가 지형 속에 파묻히지 않도록 스폰 높이만 보정한다.
"""
from __future__ import annotations

import mujoco
import numpy as np
from flybody.tasks.template_task import TemplateTask

from flyski_sim.ski_profiles import SkiProfile
from flyski_sim.ski_attachment import attach_skis_to_walker
from flyski_sim.ski_plate import attach_side_plates
from flyski_sim.snow import SNOW_PRESETS

_SPAWN_CLEARANCE_CM = 0.3  # 지형 표면 위로 띄우는 여유 높이, cm.


class SlopeSmokeTask(TemplateTask):
    """경사/모글 아레나 위에서 초파리를 스폰하는 최소 스모크테스트 태스크.

    ski_profile을 주면 업무2/3 스키까지 적용한다.
    ski_layout: 'side_plate'(기본, 좌/우 휘는 플레이트 N=2 — RESEARCH_NOTES 32번) 또는
        'per_leg'(레거시, 다리당 강체 미니스키 N=6).
    """

    def __init__(self, ski_profile: SkiProfile | None = None,
                 ski_layout: str = 'side_plate', **kwargs):
        super().__init__(**kwargs)
        self._ski_profile = ski_profile
        self._skis = {}
        if ski_profile is not None:
            if ski_layout == 'side_plate':
                self._skis = attach_side_plates(self._walker, ski_profile)
            elif ski_layout == 'per_leg':
                self._skis = attach_skis_to_walker(self._walker, ski_profile)
            else:
                raise ValueError(f'Unknown ski_layout: {ski_layout!r}')

    @property
    def skis(self):
        """{이름: SkiUnit} — side_plate면 'left'/'right', per_leg면 'T1_left' 등."""
        return self._skis

    def initialize_episode(self, physics, random_state: np.random.RandomState):
        super().initialize_episode(physics, random_state)
        # composer는 task.initialize_episode를 arena(엔티티) 훅보다 먼저 부른다 —
        # 여기서 먼저 지형을 생성해두지 않으면 height_at()이 0을 돌려줘서 초파리가
        # 지형 속(~10cm 아래)에 파묻힌 채 스폰된다(RESEARCH_NOTES.md 30번).
        # arena 자신의 훅은 이후 _regenerate=False라 재생성하지 않는다.
        if hasattr(self._arena, 'height_at'):
            self._arena.initialize_episode(physics, random_state)
        pos, quat = self._walker.get_pose(physics)
        x, y = float(pos[0]), float(pos[1])
        if hasattr(self._arena, 'height_at'):
            terrain_h = self._arena.height_at(x, y)
        else:
            terrain_h = 0.
        new_pos = (x, y, terrain_h + _SPAWN_CLEARANCE_CM)
        self._walker.set_pose(physics, position=new_pos, quaternion=quat)
        if hasattr(self._arena, 'apply_snow_friction_at'):
            self._arena.apply_snow_friction_at(physics, x)

    def before_step(self, physics, action, random_state: np.random.RandomState):
        # 초파리가 현재 있는 x 위치의 설질을 매 컨트롤 스텝마다 반영한다.
        if hasattr(self._arena, 'apply_snow_friction_at'):
            pos, _ = self._walker.get_pose(physics)
            self._arena.apply_snow_friction_at(physics, float(pos[0]))
        self._apply_snow_to_skis(physics)
        super().before_step(physics, action, random_state)

    def _apply_snow_to_skis(self, physics) -> None:
        """스키 조각마다 그 위치 설질의 ski_friction + 속도비례 항력을 적용한다."""
        if not self._skis or not hasattr(self._arena, 'get_snow_params'):
            return
        for unit in self._skis.values():
            for geom in unit.geoms:
                g = physics.bind(geom)
                snow = self._arena.get_snow_params(float(g.xpos[0]))
                g.friction = [snow.ski_friction, 0.005, 0.0001]
        self._apply_snow_drag(physics)

    def _apply_snow_drag(self, physics) -> None:
        """설질별 속도비례 보조 항력을 각 스키에 xfrc_applied로 적용한다.

        MuJoCo hfield는 강체라 실제로 '빠지는' 기하학적 침투가 없으므로(
        RESEARCH_NOTES.md 참고), 대신 스키의 현재 이동 속도에 비례하는 항력을
        걸어 파우더/슬러시 같은 무른 설질의 저항감을 근사한다. 접지 면적이
        넓은 스키(파우더 프로파일)일수록 항력이 작다(compute_penetration_cm과
        같은 방향성).
        """
        if not self._skis or not hasattr(self._arena, 'get_snow_params'):
            return
        vel6 = np.zeros(6)
        for unit in self._skis.values():
            area = max(unit.area_cm2, 1e-6)
            for body in unit.bodies:
                bid = physics.model.name2id(body.full_identifier, 'body')
                world_pos = physics.data.xpos[bid]
                snow = self._arena.get_snow_params(float(world_pos[0]))
                # cvel은 트리 루트 subtree_com 기준 속도라 스키 자체 선속도가 아니다 —
                # mj_objectVelocity로 조각 원점의 world-frame [각속도, 선속도]를 얻는다.
                mujoco.mj_objectVelocity(physics.model.ptr, physics.data.ptr,
                                         mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
                # 스키 한 짝의 항력을 조각 수로 나눠 분산.
                drag_force = (-snow.drag_coefficient * vel6[3:6] / area /
                              len(unit.bodies))
                # xfrc_applied 규약은 [힘(3), 토크(3)] — 예전엔 항력을 [3:6](토크)에
                # 넣어서 스키를 회전시키고 있었다(RESEARCH_NOTES.md 30번).
                physics.data.xfrc_applied[bid, :3] = drag_force
                physics.data.xfrc_applied[bid, 3:6] = 0.

    def get_reward_factors(self, physics):
        # 스모크테스트에는 보상이 필요 없음 — 물리 안정성만 확인한다.
        return (1.,)
