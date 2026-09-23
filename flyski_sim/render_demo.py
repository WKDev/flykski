"""시각 검증용 렌더링 스크립트.

MuJoCo 오프스크린 렌더링으로 현재 flyski_sim 구현 상태를 이미지/영상으로
저장한다. GUI 창이 필요 없어서 헤드리스 환경에서도 동작한다.

사용법: python -m flyski_sim.render_demo
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from dm_control import composer
from dm_control.mujoco import Camera
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls
from flyski_sim.connectome_controller_lif import LIFConnectomeController
from flyski_sim.motor_decoder import build_motor_neuron_dof_map, decode_to_flybody_action

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'renders')
_CONN_DIR = os.path.join(os.path.dirname(__file__), 'connectome')


def save_png(pixels: np.ndarray, name: str):
    path = os.path.join(OUT_DIR, name)
    Image.fromarray(pixels).save(path)
    print(f'saved: {path}')


def render_flat_ground_multi_camera():
    arena = floors.Floor()
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES['all_mountain'],
                          walker=fruitfly.FruitFly, arena=arena, time_limit=1.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=1.0, task=task,
                               random_state=np.random.RandomState(1),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    for _ in range(30):  # 살짝 안정화.
        env.step(np.zeros(env.action_spec().shape[0], dtype=np.float32))

    for cam_name in ('walker/hero', 'walker/side', 'walker/back', 'top_camera'):
        pixels = env.physics.render(height=720, width=960, camera_id=cam_name)
        save_png(pixels, f'flat_ground_{cam_name.replace("/", "_")}.png')


def render_slope_gif(terrain_type: str = 'mogul', n_frames: int = 60):
    arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=20., terrain_type=terrain_type,
                         mogul_wavelength=3.0, mogul_height=1.5, grid_density=15)
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES['mogul'],
                          walker=fruitfly.FruitFly, arena=arena, time_limit=2.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=2.0, task=task,
                               random_state=np.random.RandomState(7),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()

    frames = []
    action_dim = env.action_spec().shape[0]
    for i in range(n_frames):
        env.step(np.zeros(action_dim, dtype=np.float32))
        pixels = env.physics.render(height=480, width=640, camera_id='walker/side')
        frames.append(Image.fromarray(pixels))

    gif_path = os.path.join(OUT_DIR, f'slope_{terrain_type}.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=1000 // 30, loop=0)
    print(f'saved: {gif_path} ({len(frames)} frames)')


def render_slope_establishing_shot(terrain_type: str = 'mogul'):
    """슬로프 전체(경사+모글)가 보이는 넓은 각도의 '설정샷'을 렌더링한다."""
    arena = SlopedMoguls(dim=(30., 5.), mean_slope_deg=20., terrain_type=terrain_type,
                         mogul_wavelength=3.0, mogul_height=1.5, grid_density=15)
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES['mogul'],
                          walker=fruitfly.FruitFly, arena=arena, time_limit=1.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=1.0, task=task,
                               random_state=np.random.RandomState(7),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()

    cam = Camera(env.physics, height=720, width=1280, camera_id=-1)
    cam._render_camera.lookat[:] = [0., 0., 5.]  # 슬로프 중앙 높이 근처.
    cam._render_camera.distance = 45.
    cam._render_camera.azimuth = 200.
    cam._render_camera.elevation = -25.
    pixels = cam.render()
    save_png(pixels, f'slope_establishing_{terrain_type}.png')


def render_connectome_driven_gif(n_frames: int = 90, neural_substeps: int = 3):
    """업무4 커넥톰 리듬으로 구동되는 다리를 GIF로 저장한다."""
    arena = floors.Floor()
    task = SlopeSmokeTask(walker=fruitfly.FruitFly, arena=arena, time_limit=6.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=6.0, task=task,
                               random_state=np.random.RandomState(2),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    actuator_names = [env.physics.model.id2name(i, 'actuator')
                      for i in range(env.physics.model.nu)]

    ctrl = LIFConnectomeController(topology_mode='real', seed=0,
                                   weight_scale=30., v_thresh=0.5, tau_mem_init=8.0)
    ctrl.reset()
    motor_meta = pd.read_csv(os.path.join(_CONN_DIR, '03_leg_motor_neurons.csv'))
    dof_map = build_motor_neuron_dof_map(motor_meta)
    motor_body_ids = motor_meta['bodyid'].tolist()

    frames = []
    for _ in range(n_frames):
        efferent = ctrl.step(afferent_drive=0.6, dt_steps=neural_substeps)
        action = decode_to_flybody_action(efferent, motor_body_ids, dof_map,
                                          actuator_names, gain=1.0)
        env.step(action.astype(np.float32))
        pixels = env.physics.render(height=480, width=640, camera_id='walker/side')
        frames.append(Image.fromarray(pixels))

    gif_path = os.path.join(OUT_DIR, 'connectome_driven_legs.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=1000 // 20, loop=0)
    print(f'saved: {gif_path} ({len(frames)} frames)')


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 평지 + 스키, 여러 카메라 앵글 ===")
    render_flat_ground_multi_camera()
    print("\n=== 경사/모글 슬로프 위, side 카메라 GIF ===")
    render_slope_gif('mogul')
    render_slope_gif('alpine')
    print("\n=== 슬로프 전체가 보이는 설정샷 ===")
    render_slope_establishing_shot('mogul')
    render_slope_establishing_shot('alpine')
    print("\n=== 업무4: 커넥톰 리듬으로 구동되는 다리 GIF ===")
    render_connectome_driven_gif()
