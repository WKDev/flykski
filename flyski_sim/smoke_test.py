"""업무1 1단계 스모크테스트: 경사+모글 슬로프 위에서 flybody가 안정적으로 도는지 확인.

01_infinite_slope_terrain.md 수용 기준 중 아래 항목을 검증한다:
- 동일 seed로 두 번 생성 시 지형이 완전히 동일함.
- mean_slope_deg=20 설정 시 실측 평균 경사가 근접함(알파인: 거의 정확히 일치해야 함).
- 모글/알파인 두 지형에서 물리 스텝이 발산(NaN)하지 않고 안정적으로 시뮬레이션됨.
"""
from __future__ import annotations

import numpy as np
from dm_control import composer

from flybody.fruitfly import fruitfly

from flyski_sim.terrain import SlopedMoguls, slope_stats_deg
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.snow import SNOW_PRESETS
from flyski_sim.odor import OdorField

ARENA_DIM = (30., 5.)  # (radius_x, radius_y) cm -> 60cm 길이 x 10cm 폭 슬로프.


def build_env(terrain_type: str,
             mean_slope_deg: float,
             seed: int,
             max_slope_deg: float | None = None):
    arena = SlopedMoguls(dim=ARENA_DIM,
                         mean_slope_deg=mean_slope_deg,
                         max_slope_deg=max_slope_deg,
                         terrain_type=terrain_type,
                         mogul_wavelength=3.0,
                         mogul_height=1.5,
                         grid_density=15)
    task = SlopeSmokeTask(walker=fruitfly.FruitFly,
                          arena=arena,
                          time_limit=1.0,
                          joint_filter=0.,
                          claw_friction=1.0)
    env = composer.Environment(time_limit=1.0,
                               task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    return env, arena, task


def check_determinism(terrain_type: str, mean_slope_deg: float, seed: int) -> bool:
    """동일 seed로 두 번 생성한 지형이 완전히 동일한지 확인."""
    env1, arena1, _ = build_env(terrain_type, mean_slope_deg, seed)
    env1.reset()
    terrain1 = arena1.last_terrain.copy()

    env2, arena2, _ = build_env(terrain_type, mean_slope_deg, seed)
    env2.reset()
    terrain2 = arena2.last_terrain.copy()

    return bool(np.array_equal(terrain1, terrain2))


def run_stability_check(terrain_type: str,
                        mean_slope_deg: float,
                        seed: int,
                        n_steps: int = 200,
                        max_slope_deg: float | None = None) -> dict:
    env, arena, task = build_env(terrain_type, mean_slope_deg, seed, max_slope_deg)
    env.reset()

    mean_deg, max_deg = slope_stats_deg(arena.last_terrain, arena.dim)
    action_dim = env.action_spec().shape[0]

    stable = True
    steps_done = 0
    start_pos, _ = task.walker.get_pose(env.physics)
    for steps_done in range(1, n_steps + 1):
        action = np.zeros(action_dim, dtype=np.float32)
        env.step(action)
        qpos_finite = np.all(np.isfinite(env.physics.data.qpos))
        qvel_finite = np.all(np.isfinite(env.physics.data.qvel))
        if not (qpos_finite and qvel_finite):
            stable = False
            break
    end_pos, _ = task.walker.get_pose(env.physics)

    return {
        'terrain_type': terrain_type,
        'target_mean_slope_deg': mean_slope_deg,
        'measured_mean_slope_deg': round(mean_deg, 2),
        'measured_max_slope_deg': round(max_deg, 2),
        'action_dim': action_dim,
        'steps_completed': steps_done,
        'stable': stable,
        'start_xyz_cm': tuple(round(float(v), 2) for v in start_pos),
        'end_xyz_cm': tuple(round(float(v), 2) for v in end_pos),
    }


def check_snow_zones(seed: int = 3) -> dict:
    """설질 구간이 실제로 배정되고, geom 마찰이 x 위치에 따라 실제로 바뀌는지 확인."""
    distribution = {t: 1.0 for t in SNOW_PRESETS}  # 5종 균등 분포.
    arena = SlopedMoguls(dim=ARENA_DIM,
                         mean_slope_deg=15.,
                         terrain_type='alpine',
                         snow_distribution=distribution,
                         n_snow_zones=5)
    task = SlopeSmokeTask(walker=fruitfly.FruitFly,
                          arena=arena,
                          time_limit=1.0,
                          joint_filter=0.,
                          claw_friction=1.0)
    env = composer.Environment(time_limit=1.0,
                               task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()

    zone_types = arena.zone_types
    size_x = arena.dim[0]
    # 각 구간 중앙에서의 friction을 실제로 physics에 반영해서 읽어본다.
    measured_frictions = []
    zone_edges = [-size_x] + arena.zone_bounds_x
    for i, snow_type in enumerate(zone_types):
        x_mid = 0.5 * (zone_edges[i] + zone_edges[i + 1])
        applied_type = arena.apply_snow_friction_at(env.physics, x_mid)
        friction = float(env.physics.bind(arena._terrain_geom).friction[0])
        measured_frictions.append(friction)
        assert applied_type == snow_type
        assert abs(friction - SNOW_PRESETS[snow_type].friction) < 1e-6

    action_dim = env.action_spec().shape[0]
    stable = True
    for _ in range(100):
        env.step(np.zeros(action_dim, dtype=np.float32))
        if not np.all(np.isfinite(env.physics.data.qpos)):
            stable = False
            break

    return {
        'zone_types': zone_types,
        'n_distinct_types': len(set(zone_types)),
        'measured_frictions': [round(f, 3) for f in measured_frictions],
        'friction_varies': len(set(measured_frictions)) > 1,
        'stable_with_zones': stable,
    }


def check_odor_field() -> dict:
    """냄새 게이트 필드: 순번/교대 배치, 순차 점화, 기울기 방향, 완주까지 검증."""
    field = OdorField(gate_spacing_cm=8.0, gate_amplitude_cm=3.0,
                      arena_dim=ARENA_DIM, pass_radius_cm=1.0, seed=1)
    gates = field.get_gate_sequence()

    order_ok = all(g.order_index == i for i, g in enumerate(gates))
    sides = [g.required_side for g in gates]
    alternates = all(sides[i] != sides[i + 1] for i in range(len(sides) - 1))

    # 활성 게이트에만 반응하고, 아직 안 켜진 다음 게이트에는 반응하지 않는지 확인.
    g0, g1 = gates[0], gates[1]
    conc_at_g0, grad_at_g0 = field.sample((g0.center_xy[0], g0.center_xy[1], 0.))
    conc_at_g1_while_g0_active, _ = field.sample(
        (g1.center_xy[0], g1.center_xy[1], 0.))
    # 게이트 중심에서 기울기는 (0,0) 근처(대칭), 살짝 벗어난 지점에서는 게이트 쪽을 가리켜야 함.
    probe = (g0.center_xy[0] - 0.5, g0.center_xy[1], 0.)
    _, grad_probe = field.sample(probe)
    gradient_points_toward_gate = grad_probe[0] > 0  # +x로 밀어야 게이트(0)를 향함.

    # 순차 통과 시뮬레이션: 일직선으로 x를 따라가며 각 게이트의 y에 도달했다고 가정.
    passed_order = []
    for g in gates:
        advanced = field.try_advance(g.center_xy)
        passed_order.append((g.gate_id, advanced))
    all_advanced = all(ok for _, ok in passed_order)

    # gate_amplitude_cm을 늘리면 연속 게이트 간 좌우 꺾임 각도가 커져야 한다.
    def turn_angle_deg(amplitude):
        f = OdorField(gate_spacing_cm=8.0, gate_amplitude_cm=amplitude,
                     arena_dim=ARENA_DIM, seed=1)
        gs = f.get_gate_sequence()
        p0, p1, p2 = gs[0].center_xy, gs[1].center_xy, gs[2].center_xy
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1)
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    angle_small = turn_angle_deg(1.0)
    angle_large = turn_angle_deg(4.0)

    return {
        'n_gates': len(gates),
        'order_monotonic': order_ok,
        'sides_alternate': alternates,
        'conc_at_own_gate_high': conc_at_g0 > 0.99,
        'conc_at_next_gate_while_inactive': round(conc_at_g1_while_g0_active, 4),
        'gradient_points_toward_active_gate': gradient_points_toward_gate,
        'sequential_passthrough_all_ok': all_advanced,
        'all_passed_after_traversal': field.all_passed,
        'turn_angle_deg_small_amplitude': round(angle_small, 1),
        'turn_angle_deg_large_amplitude': round(angle_large, 1),
        'larger_amplitude_sharper_turn': angle_large > angle_small,
    }


if __name__ == '__main__':
    print("=== 결정성(determinism) 검사 ===")
    for terrain_type in ('alpine', 'mogul'):
        same = check_determinism(terrain_type, mean_slope_deg=20., seed=42)
        print(f"  {terrain_type}: 동일 seed -> 동일 지형? {same}")

    print("\n=== 물리 안정성 + 경사 통계 검사 ===")
    for terrain_type in ('alpine', 'mogul'):
        result = run_stability_check(terrain_type, mean_slope_deg=20., seed=42)
        print(f"  [{terrain_type}] {result}")

    print("\n=== max_slope_deg 자동 보정 검사 (mean=20, max=40 목표) ===")
    result = run_stability_check('mogul', mean_slope_deg=20., seed=7,
                                 max_slope_deg=40.)
    print(f"  [mogul, calibrated] {result}")

    print("\n=== 설질(snow) 구간 검사 ===")
    snow_result = check_snow_zones()
    print(f"  {snow_result}")

    print("\n=== 냄새 게이트 필드 검사 ===")
    odor_result = check_odor_field()
    print(f"  {odor_result}")
