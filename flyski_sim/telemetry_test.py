"""업무3 파생 텔레메트리 스모크테스트.

03_ski_attachment_actuation.md 수용 기준:
- effective_edge_angle_deg가 수동 조작에서 기대한 부호/크기로 변함.
- is_fallen이 실제 넘어짐 상황에서 정확히 True가 됨.
"""
from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.telemetry import compute_ski_telemetry


def _build(seed=9):
    profile = SKI_PROFILES['all_mountain']
    arena = floors.Floor()
    task = SlopeSmokeTask(ski_profile=profile, walker=fruitfly.FruitFly, arena=arena,
                          time_limit=1.0, joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=1.0, task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    return env, task, profile


def check_standing_telemetry() -> dict:
    env, task, profile = _build()
    env.reset()
    for _ in range(20):
        env.step(np.zeros(env.action_spec().shape[0], dtype=np.float32))
    t = compute_ski_telemetry(env.physics, task.walker, task.skis)
    load_sum = sum(t.plate_load_share.values())
    return {
        'plate_load_share_sums_to_1': abs(load_sum - 1.0) < 0.05,
        'plate_load_share': {k: round(v, 3) for k, v in t.plate_load_share.items()},
        'mean_edge_angle_deg_standing': round(
            float(np.mean(list(t.effective_edge_angle_deg.values()))), 2),
        'n_support_contacts': len(t.support_polygon_xy),
        'body_tilt_deg': round(t.body_tilt_deg, 2),
        'is_fallen_while_standing': t.is_fallen,
    }


def check_fallen_detection() -> dict:
    """몸통을 강제로 크게 기울인 뒤 is_fallen=True가 되는지 확인."""
    env, task, profile = _build()
    env.reset()
    pos, quat = task.walker.get_pose(env.physics)
    # 90도 옆으로 넘어진 자세로 강제 설정.
    tipped_quat = (0.7071, 0.7071, 0., 0.)
    task.walker.set_pose(env.physics, position=(pos[0], pos[1], pos[2] + 0.05),
                         quaternion=tipped_quat)
    env.physics.forward()
    t = compute_ski_telemetry(env.physics, task.walker, task.skis)
    return {
        'body_tilt_deg_after_tip': round(t.body_tilt_deg, 2),
        'is_fallen_after_tip': t.is_fallen,
    }


def check_edge_angle_manual_tilt() -> dict:
    """스키 하나를 강제로 크게 기울인 뒤 effective_edge_angle_deg가 커지는지 확인."""
    env, task, profile = _build()
    env.reset()
    leg_id = 'left'
    # side_plate의 T2 바인딩 ball 조인트를 직접 돌려 판 전체를 기울인다(forward만
    # 하므로 T1/T3 connect 구속 위반은 텔레메트리 판독에 영향 없음).
    jid = env.physics.model.name2id(f'walker/ski_bind_{leg_id}', 'joint')
    qpos_adr = env.physics.model.jnt_qposadr[jid]

    def edge_angle_for_quat(quat):
        env.physics.data.qpos[qpos_adr:qpos_adr + 4] = quat
        env.physics.forward()
        t = compute_ski_telemetry(env.physics, task.walker, task.skis)
        return t.effective_edge_angle_deg[leg_id]

    angle_neutral = edge_angle_for_quat((1., 0., 0., 0.))
    angle_tilted = edge_angle_for_quat((0.866, 0.5, 0., 0.))  # 로컬 X축으로 60도 회전.

    return {
        'edge_angle_neutral_deg': round(angle_neutral, 2),
        'edge_angle_tilted_deg': round(angle_tilted, 2),
        'tilt_increased_edge_angle': angle_tilted > angle_neutral,
    }


if __name__ == '__main__':
    print("=== 정상 기립 텔레메트리 검사 ===")
    print(f"  {check_standing_telemetry()}")

    print("\n=== 전복 판정 검사 ===")
    print(f"  {check_fallen_detection()}")

    print("\n=== 엣지각 수동 조작 검사 ===")
    print(f"  {check_edge_angle_manual_tilt()}")
