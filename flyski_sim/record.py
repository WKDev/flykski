# 환경 + 컨트롤러(전문가/학습 정책) 한 에피소드를 목표/실제 궤적 선과 함께 MP4로 녹화하는 체크포인트 녹화기
"""체크포인트 녹화(영상 자료용).

    python -m flyski_sim.record --stage turn --controller steer --name 03_stem_steer_expert
    python -m flyski_sim.record --stage speed --controller expert --name 02_snowplow_speed
    python -m flyski_sim.record --stage turn --model runs/x/model.zip --name 04_policy

MuJoCo 기본 렌더러(mujoco.Renderer)를 쓴다. dm_control 카메라 장면에는 선(user geom)을
덧붙일 수 없어서(geoms가 현재 개수만큼만 보임) 녹화는 이쪽으로 한다. 초록 = 목표, 빨강 =
실제(play.TrajectoryOverlay와 같음). 프레임은 ffmpeg(PATH)로 MP4, 없으면 GIF.
결과: renders/checkpoints/<name>.mp4 (+ 같은 이름 .png 대표 프레임).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import warnings

import mujoco
import numpy as np
from PIL import Image

from flyski_sim.play import TrajectoryOverlay
from flyski_sim.rl_task import CONTROL_DT, ENVS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'renders', 'checkpoints')


def _compat(model, env):
    """관측에 목표 오차 2칸(방향/옆)이 생기기 전(148차원)에 학습된 모델이면 그 2칸을 빼고 넣는다.
    관측 순서: ... 속도 명령(1), 목표 오차(2), 직전 액션(42)."""
    n_model = model.observation_space.shape[0]
    n_env = env.observation_space.shape[0]
    n_act = env.action_space.shape[0]
    if n_model == n_env - 2:
        cut = n_env - n_act - 2
        return lambda o: model.predict(np.delete(o, [cut, cut + 1]), deterministic=True)[0]
    return lambda o: model.predict(o, deterministic=True)[0]


def make_policy(env, controller, model_path=None):
    if model_path:
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device='cpu')
        return _compat(model, env)
    if controller in ('steer', 'parallel'):
        from flyski_sim.experts import ParallelExpert, SteerExpert
        return (SteerExpert if controller == 'steer' else ParallelExpert)(env)
    if controller == 'expert':
        from flyski_sim.experts import SnowplowExpert
        return SnowplowExpert(env)
    return lambda o: np.zeros(env.action_space.shape)


def record(env, policy, name, seconds=None, fps=30, width=960, height=540,
           distance=3.0, azimuth=0., elevation=-35., seed=0):
    os.makedirs(OUT, exist_ok=True)
    obs, _ = env.reset(seed=seed)
    p = env.env.physics
    m, d = p.model.ptr, p.data.ptr
    overlay = TrajectoryOverlay(env)
    renderer = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    every = max(int(round(1. / (fps * CONTROL_DT))), 1)
    frames, total, info, k = [], 0., {}, 0
    max_steps = int((seconds or env.episode_seconds) / CONTROL_DT)
    while k < max_steps:
        obs, r, term, trunc, info = env.step(policy(obs))
        overlay.step()
        total += r
        k += 1
        if k % every == 0:
            cam.lookat[:] = p.data.xpos[env._th]
            renderer.update_scene(d, camera=cam)
            overlay.draw(renderer.scene, append=True)
            frames.append(renderer.render().copy())
        if term or trunc:
            break
    renderer.close()
    base = os.path.join(OUT, name)
    Image.fromarray(frames[len(frames) // 2]).save(base + '.png')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        tmp = base + '_frames'
        os.makedirs(tmp, exist_ok=True)
        for i, f in enumerate(frames):
            Image.fromarray(f).save(os.path.join(tmp, f'{i:05d}.png'))
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-framerate', str(fps), '-i',
                        os.path.join(tmp, '%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '23', base + '.mp4'], check=True)
        shutil.rmtree(tmp)
        path = base + '.mp4'
    else:
        imgs = [Image.fromarray(f) for f in frames[::2]]
        imgs[0].save(base + '.gif', save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0)
        path = base + '.gif'
    print(f'{name}: {k * CONTROL_DT:.1f}s, return {total:.1f}, end {info.get("reason", "?")} -> {path}',
          flush=True)
    return path


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='turn', choices=tuple(ENVS))
    ap.add_argument('--controller', default='steer', choices=('steer', 'parallel', 'expert', 'zero'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--name', required=True)
    ap.add_argument('--seconds', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--distance', type=float, default=3.0, help='카메라 거리(cm), S자 전체를 보려면 8~10')
    ap.add_argument('--elevation', type=float, default=-35.)
    args = ap.parse_args()
    env = ENVS[args.stage](seed=args.seed)
    record(env, make_policy(env, args.controller, args.model), args.name, args.seconds, seed=args.seed,
           distance=args.distance, elevation=args.elevation)


if __name__ == '__main__':
    main()
