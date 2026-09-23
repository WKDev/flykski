"""실시간 화면 뷰어 — 오프스크린 렌더 대신 MuJoCo 네이티브 창으로 시뮬레이션을 띄운다.

사용법(flykski 루트에서):
    python -m flyski_sim.live_view                       # 모글 슬로프, 힘 뺀 수동 활강
    python -m flyski_sim.live_view --terrain alpine
    python -m flyski_sim.live_view --controller connectome   # LIF 커넥톰 리듬으로 다리 구동
    python -m flyski_sim.live_view --flat                # 평지

창 조작: 마우스 드래그로 회전/이동, 스크롤로 줌, Space 일시정지, Tab 패널.
카메라는 기본적으로 초파리 몸통(thorax)을 따라간다. 에피소드가 끝나면 자동 리셋.
"""
from __future__ import annotations

import argparse
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls

_CONN_DIR = os.path.join(os.path.dirname(__file__), 'connectome')


def build_env(terrain: str, flat: bool, ski: str, seed: int, time_limit: float):
    if flat:
        arena = floors.Floor()
    else:
        arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=20., terrain_type=terrain,
                             mogul_wavelength=3.0, mogul_height=1.5, grid_density=15)
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES[ski], walker=fruitfly.FruitFly,
                          arena=arena, time_limit=time_limit,
                          joint_filter=0., claw_friction=1.0)
    return composer.Environment(time_limit=time_limit, task=task,
                                random_state=np.random.RandomState(seed),
                                strip_singleton_obs_buffer_dim=True)


def make_policy(kind: str, env):
    nu = env.action_spec().shape[0]
    if kind == 'zero':
        return lambda: np.zeros(nu, np.float32), lambda: None
    from flyski_sim.connectome_controller_lif import LIFConnectomeController
    from flyski_sim.motor_decoder import build_motor_neuron_dof_map, decode_to_flybody_action
    meta = pd.read_csv(os.path.join(_CONN_DIR, '03_leg_motor_neurons.csv'))
    dof_map = build_motor_neuron_dof_map(meta)
    body_ids = meta['bodyid'].tolist()
    names = [env.physics.model.id2name(i, 'actuator') for i in range(env.physics.model.nu)]
    # train_connectome_params 23번 최적값(전역 스칼라) 근처.
    ctrl = LIFConnectomeController(topology_mode='real', weight_scale=61.8,
                                   v_thresh=0.59, tau_mem_init=13.7)

    def act():
        eff = ctrl.step(afferent_drive=0.6, dt_steps=3)
        return decode_to_flybody_action(eff, body_ids, dof_map, names,
                                        gain=1.45).astype(np.float32)
    return act, ctrl.reset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--terrain', default='mogul', choices=('mogul', 'alpine'))
    ap.add_argument('--flat', action='store_true')
    ap.add_argument('--ski', default='all_mountain', choices=tuple(SKI_PROFILES))
    ap.add_argument('--controller', default='zero', choices=('zero', 'connectome'))
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--time-limit', type=float, default=10.0)
    ap.add_argument('--slowmo', type=float, default=1.0,
                    help='1=실시간, 5=5배 느리게')
    args = ap.parse_args()

    env = build_env(args.terrain, args.flat, args.ski, args.seed, args.time_limit)
    env.reset()
    act, reset_policy = make_policy(args.controller, env)
    physics = env.physics
    m, d = physics.model.ptr, physics.data.ptr
    thorax = physics.model.name2id('walker/thorax', 'body')
    dt = env.control_timestep()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 120
        while viewer.is_running():
            t0 = time.time()
            ts = env.step(act())
            if ts.last():
                env.reset()
                reset_policy()
            viewer.sync()
            # 물리가 실시간보다 느려도 렌더 스레드가 돌 틈은 준다.
            time.sleep(max(dt * args.slowmo - (time.time() - t0), 0.001))


if __name__ == '__main__':
    main()
