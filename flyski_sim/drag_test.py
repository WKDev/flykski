"""설질 침투 보조력(xfrc_applied) 스모크테스트.

03_ski_attachment_actuation.md: "파우더 설질에서 솔버 파라미터만으로 침투
저항이 불충분할 경우, xfrc_applied 보조 항력 경로... on/off 비교 벤치마크".
"""
from __future__ import annotations

import numpy as np
from dm_control import composer

from flybody.fruitfly import fruitfly

from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls


def check_drag_scales_with_snow_type() -> dict:
    """같은 속도에서 파우더 구간 항력이 아이스 구간보다 커야 한다."""
    profile = SKI_PROFILES['all_mountain']
    # 앞쪽 절반은 ice, 뒤쪽 절반은 powder로 고정 배치(가중치 편향으로 근사).
    arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=10., terrain_type='alpine',
                         grid_density=15, n_snow_zones=2,
                         snow_distribution={'ice': 1.0})
    task = SlopeSmokeTask(ski_profile=profile, walker=fruitfly.FruitFly, arena=arena,
                          time_limit=1.0, joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=1.0, task=task,
                               random_state=np.random.RandomState(3),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    p = env.physics

    ski_body = task.skis['left'].root_body
    bid = p.model.name2id(ski_body.full_identifier, 'body')

    def drag_force_for_snow(snow_type: str, speed_cm_s: float = 5.0):
        arena._zone_types = [snow_type] * arena._n_snow_zones
        # 몸통(root) 자유 관절의 선속도 성분에 직접 수평 속도를 주입.
        p.data.qvel[:3] = [speed_cm_s, 0., 0.]
        p.forward()
        task._apply_snow_drag(p)
        # xfrc_applied 규약: [힘(3), 토크(3)] — 항력은 힘 칸에 들어간다(RESEARCH_NOTES 30번).
        return np.array(p.data.xfrc_applied[bid, :3])

    force_ice = drag_force_for_snow('ice')
    force_powder = drag_force_for_snow('powder')

    return {
        'force_ice': [round(float(v), 5) for v in force_ice],
        'force_powder': [round(float(v), 5) for v in force_powder],
        'powder_drag_stronger': (np.linalg.norm(force_powder) >
                                 np.linalg.norm(force_ice)),
        'force_opposes_motion_x': force_powder[0] < 0,  # +x로 움직였으니 반대(-x).
    }


def check_stability_with_drag(n_steps: int = 150) -> dict:
    profile = SKI_PROFILES['powder']
    arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=20., terrain_type='mogul',
                         mogul_wavelength=3.0, mogul_height=1.0, grid_density=15,
                         snow_distribution={'powder': 1.0})
    task = SlopeSmokeTask(ski_profile=profile, walker=fruitfly.FruitFly, arena=arena,
                          time_limit=1.0, joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=1.0, task=task,
                               random_state=np.random.RandomState(4),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    action_dim = env.action_spec().shape[0]
    stable = True
    steps_done = 0
    for steps_done in range(1, n_steps + 1):
        env.step(np.zeros(action_dim, dtype=np.float32))
        if not np.all(np.isfinite(env.physics.data.qpos)):
            stable = False
            break
    return {'steps_completed': steps_done, 'stable_with_powder_drag': stable}


if __name__ == '__main__':
    print("=== 설질별 항력 크기 비교 (같은 속도) ===")
    print(f"  {check_drag_scales_with_snow_type()}")

    print("\n=== 파우더 설질 + drag 적용 상태 물리 안정성 ===")
    print(f"  {check_stability_with_drag()}")
