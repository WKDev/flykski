"""업무2/3 스모크테스트: 스키 프로파일 물리량 + 실제 부착 후 안정성 검증.

02_ski_profiles.md 수용 기준:
- 최소 3개, 최대 5개의 프로파일이 정의되고 각각 고유한 파라미터 조합을 가짐.
- 동일 엣지각·속도에서 프로파일별 회전 반경이 서로 다름.
- 스키 부착 후에도 flybody가 평지에서 크래시 없이 서 있을 수 있음.
- 좌우 대칭 위치의 스키 질량·관성이 좌우 대칭으로 정확히 일치.
"""
from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.ski_profiles import SKI_PROFILES, compute_turn_radius, compute_penetration_cm
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls


def check_profiles_distinct() -> dict:
    names = list(SKI_PROFILES.keys())
    radii = {
        name: compute_turn_radius(p, effective_edge_angle_deg=30., speed_cm_s=10.)
        for name, p in SKI_PROFILES.items()
    }
    areas = {name: round(p.base_surface_area_cm2, 5) for name, p in SKI_PROFILES.items()}
    return {
        'n_profiles': len(names),
        'names': names,
        'turn_radii_cm_at_30deg': {k: round(v, 3) for k, v in radii.items()},
        'all_radii_distinct': len(set(round(v, 4) for v in radii.values())) == len(radii),
        'base_surface_area_cm2': areas,
        'powder_has_largest_area': areas['powder'] == max(areas.values()),
    }


def check_penetration_ordering() -> dict:
    """같은 '물렁함'이면, 접지 면적이 넓은 파우더 스키가 덜 빠져야 한다."""
    softness = 1.0
    pen_powder = compute_penetration_cm(SKI_PROFILES['powder'], softness)
    pen_slalom = compute_penetration_cm(SKI_PROFILES['slalom'], softness)
    return {
        'penetration_powder_cm': round(pen_powder, 4),
        'penetration_slalom_cm': round(pen_slalom, 4),
        'powder_penetrates_less': pen_powder < pen_slalom,
    }


def build_flat_ski_env(profile_name: str, seed: int, ski_layout: str = 'side_plate'):
    arena = floors.Floor()
    profile = SKI_PROFILES[profile_name]
    task = SlopeSmokeTask(ski_profile=profile,
                          ski_layout=ski_layout,
                          walker=fruitfly.FruitFly,
                          arena=arena,
                          time_limit=1.0,
                          joint_filter=0.,
                          claw_friction=1.0)
    env = composer.Environment(time_limit=1.0,
                               task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    return env, task


def _unit_mass(env, unit) -> float:
    return float(sum(env.physics.bind(b).mass for b in unit.bodies))


def check_ski_attachment_stability(profile_name: str = 'all_mountain',
                                   n_steps: int = 200,
                                   seed: int = 5,
                                   ski_layout: str = 'side_plate') -> dict:
    env, task = build_flat_ski_env(profile_name, seed, ski_layout)
    env.reset()

    n_skis = len(task.skis)
    # 좌우 대칭 질량 확인 (side_plate: left vs right, per_leg: T1_left vs T1_right 등).
    if ski_layout == 'side_plate':
        sym_pairs = [('left', 'right')]
    else:
        sym_pairs = [(f'T{i}_left', f'T{i}_right') for i in (1, 2, 3)]
    mass_symmetry_ok = all(
        abs(_unit_mass(env, task.skis[a]) - _unit_mass(env, task.skis[b])) < 1e-12
        for a, b in sym_pairs)

    action_dim = env.action_spec().shape[0]
    stable = True
    steps_done = 0
    for steps_done in range(1, n_steps + 1):
        action = np.zeros(action_dim, dtype=np.float32)
        env.step(action)
        if not np.all(np.isfinite(env.physics.data.qpos)):
            stable = False
            break

    return {
        'profile': profile_name,
        'n_skis_attached': n_skis,
        'mass_symmetry_ok': mass_symmetry_ok,
        'action_dim': action_dim,
        'steps_completed': steps_done,
        'stable_on_flat_ground': stable,
    }


def check_side_plate_physics(profile_name: str = 'all_mountain', n_steps: int = 300,
                             seed: int = 5) -> dict:
    """side_plate(N=2) 전용 검사 — 크래시 안 남 ≠ 제대로 동작(RESEARCH_NOTES 14번).

    - 좌/우 판 조각이 실제로 지면과 접촉하는가(physics.data.contact 직접 확인).
    - T1/T3 soft connect 바인딩 오차(발끝 vs 판 위 앵커 거리)가 작은가.
    - 굽힘 힌지가 하중을 받아 실제로 휘는가(최대 |각도|).
    - 판 루트 조각 롤(판 로컬 z vs 월드 z 각도)이 작은가.
    """
    env, task = build_flat_ski_env(profile_name, seed)
    env.reset()
    p = env.physics
    nu = env.action_spec().shape[0]
    for _ in range(n_steps):
        env.step(np.zeros(nu, dtype=np.float32))
    ok_finite = bool(np.all(np.isfinite(p.data.qpos)))

    geom_side = {p.model.name2id(g.full_identifier, 'geom'): side
                 for side, u in task.skis.items() for g in u.geoms}
    ground_contacts = {side: 0 for side in task.skis}
    for c in p.data.contact[:p.data.ncon]:
        for g, other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if g in geom_side and other not in geom_side and not (
                    p.model.id2name(other, 'geom') or '').startswith('walker/'):
                ground_contacts[geom_side[g]] += 1

    bind_err = {}
    for i in range(p.model.neq):
        name = p.model.id2name(i, 'equality') or ''
        if 'ski_bind_' not in name:
            continue
        b1, b2 = p.model.eq_obj1id[i], p.model.eq_obj2id[i]
        a1 = p.data.xpos[b1] + p.data.xmat[b1].reshape(3, 3) @ p.model.eq_data[i, 0:3]
        a2 = p.data.xpos[b2] + p.data.xmat[b2].reshape(3, 3) @ p.model.eq_data[i, 3:6]
        bind_err[name.split('/')[-1]] = float(np.linalg.norm(a1 - a2))

    bend = [abs(float(p.bind(j).qpos[0])) for u in task.skis.values() for b in u.bodies
            for j in b.get_children('joint') if j.name.startswith('ski_bend_')]
    roll = {}
    for side, u in task.skis.items():
        z = p.bind(u.root_body).xmat.reshape(3, 3)[:, 2]
        roll[side] = float(np.degrees(np.arccos(np.clip(z[2], -1, 1))))
    return {
        'profile': profile_name,
        'finite': ok_finite,
        'ground_contacts': ground_contacts,
        'both_plates_on_ground': all(v > 0 for v in ground_contacts.values()),
        'max_binding_error_cm': round(max(bind_err.values()), 5),
        'max_bend_deg': round(float(np.degrees(max(bend))), 2),
        'root_tilt_deg': {k: round(v, 2) for k, v in roll.items()},
    }


def check_ski_slides_downhill(profile_name: str = 'all_mountain', n_steps: int = 500,
                              seed: int = 7) -> dict:
    """스키 시뮬레이터의 최소 조건: 20° 정설 사면에서 실제로 미끄러져 내려가는가.

    RESEARCH_NOTES 32번: 예전엔 스키 마찰(0.5)이 tan(20°)보다 커서 스키가 사면에
    붙어 있었는데도 "안정적"이라 모든 테스트를 통과했다. 지면과 닿는 것이 스키뿐인지
    (발톱이 눈에 닿아 브레이크를 걸지 않는지)도 함께 본다.
    """
    arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=20., terrain_type='alpine',
                         grid_density=15)
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES[profile_name],
                          walker=fruitfly.FruitFly, arena=arena, time_limit=5.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=5.0, task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    p = env.physics
    thorax = p.model.name2id('walker/thorax', 'body')
    x0 = float(p.data.xpos[thorax][0])
    ski_geoms = {p.model.name2id(g.full_identifier, 'geom')
                 for u in task.skis.values() for g in u.geoms}
    non_ski_ground = 0
    nu = env.action_spec().shape[0]
    for t in range(n_steps):
        env.step(np.zeros(nu, dtype=np.float32))
        if t < 100:
            continue
        for c in p.data.contact[:p.data.ncon]:
            names = [p.model.id2name(g, 'geom') or '' for g in (c.geom1, c.geom2)]
            walker_side = [g for g, n in zip((c.geom1, c.geom2), names)
                           if n.startswith('walker/')]
            if len(walker_side) == 1 and walker_side[0] not in ski_geoms:
                non_ski_ground += 1
    dx = float(p.data.xpos[thorax][0]) - x0
    up_z = float(p.data.xmat[thorax].reshape(3, 3)[2, 2])
    return {'profile': profile_name, 'downhill_dx_cm': round(dx, 2),
            'slides_downhill': dx > 5.0, 'non_ski_ground_contacts': non_ski_ground,
            'body_up_z': round(up_z, 2), 'upright': up_z > 0.7,
            'finite': bool(np.all(np.isfinite(p.data.qpos)))}


def check_ski_on_slope(profile_name: str = 'mogul', seed: int = 5) -> dict:
    """스키 부착 + 경사 지형(업무1) 결합 — 스텝 안정성만 확인."""
    arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=15., terrain_type='mogul',
                         mogul_wavelength=3.0, mogul_height=1.0, grid_density=15)
    profile = SKI_PROFILES[profile_name]
    task = SlopeSmokeTask(ski_profile=profile,
                          walker=fruitfly.FruitFly,
                          arena=arena,
                          time_limit=1.0,
                          joint_filter=0.,
                          claw_friction=1.0)
    env = composer.Environment(time_limit=1.0,
                               task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    action_dim = env.action_spec().shape[0]
    stable = True
    steps_done = 0
    for steps_done in range(1, 151):
        env.step(np.zeros(action_dim, dtype=np.float32))
        if not np.all(np.isfinite(env.physics.data.qpos)):
            stable = False
            break
    return {'profile': profile_name, 'steps_completed': steps_done,
           'stable_on_slope': stable}


if __name__ == '__main__':
    print("=== 스키 프로파일 구별성 검사 ===")
    print(f"  {check_profiles_distinct()}")

    print("\n=== 설질 침투 순서 검사 ===")
    print(f"  {check_penetration_ordering()}")

    print("\n=== 스키 부착 + 평지 안정성 검사 (side_plate N=2) ===")
    for profile_name in ('slalom', 'all_mountain', 'powder'):
        result = check_ski_attachment_stability(profile_name)
        print(f"  [{profile_name}] {result}")
    print("  [per_leg 레거시] "
          f"{check_ski_attachment_stability('all_mountain', ski_layout='per_leg')}")

    print("\n=== side_plate 물리 검사 (접촉/바인딩 오차/휨/롤) ===")
    results = {name: check_side_plate_physics(name) for name in ('slalom', 'mogul')}
    for name, r in results.items():
        print(f"  [{name}] {r}")
    print(f"  부드러운 스키(mogul, flex=0.25)가 딱딱한 스키(slalom, flex=0.9)보다 더 휨? "
          f"{results['mogul']['max_bend_deg'] > results['slalom']['max_bend_deg']}")

    print("\n=== 20° 정설 사면 활강 검사 (zero action) ===")
    for name in ('slalom', 'powder', 'mogul'):
        print(f"  {check_ski_slides_downhill(name)}")

    print("\n=== 스키 부착 + 경사 지형 결합 검사 ===")
    result = check_ski_on_slope('mogul')
    print(f"  {result}")
