# RL 환경(속도 제어/코스)을 스크립트 플루크 컨트롤러나 학습된 PPO 정책으로 실시간 3D 창에 재생하는 뷰어
"""정책 실시간 재생.

    python -m flyski_sim.play --stage speed --controller snowplow     # 스크립트 플루크(명령 따라 쐐기)
    python -m flyski_sim.play --stage speed --model runs/speed1/model.zip
    python -m flyski_sim.play --stage course --model runs/ppo_try1/model.zip

창 제목 옆 콘솔에 현재 속도 명령/실제 속도/보상 누적이 찍힌다. 에피소드가 끝나면 자동
리셋. macOS는 mjpython으로 실행.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from flyski_sim.rl_task import ENVS
from flyski_sim.ski_stance import snowplow_pose

# 속도 명령(cm/s) -> (쐐기°, 안쪽 엣지°). snowplow_test 측정값에서 고른 표. 판이 0.81cm로
# 길어져(38번) 쐐기 20°부터 좌우 판 팁이 닿으므로 최대 20°.
SNOWPLOW_TABLE = ((0., 20., 15.), (5., 12., 12.), (10., 7., 8.), (15., 0., 0.))


def snowplow_controller(env):
    """명령이 느릴수록 쐐기를 크게 여는 스크립트 컨트롤러(IK 자세, 되먹임 없음)."""
    table = {}
    for cmd, wedge, edge in SNOWPLOW_TABLE:
        pose, _ = snowplow_pose(env.task.walker, wedge, edge)
        table[cmd] = np.clip(np.array([pose[n] - env._stance_q[i]
                                       for i, n in enumerate(env._leg_names)]) / env._action_scale, -1, 1)
    return lambda obs: table[min(table, key=lambda c: abs(c - env._cmd))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='speed', choices=tuple(ENVS))
    ap.add_argument('--controller', default='model', choices=('model', 'snowplow', 'zero'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--slowmo', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    env = ENVS[args.stage](seed=args.seed)
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device='cpu')
        policy = lambda obs: model.predict(obs, deterministic=True)[0]
    elif args.controller == 'snowplow':
        policy = snowplow_controller(env)
    else:
        policy = lambda obs: np.zeros(env.action_space.shape)

    obs, _ = env.reset(seed=args.seed)
    p = env.env.physics
    th = p.model.name2id('walker/thorax', 'body')
    total, last_print = 0., 0.
    with mujoco.viewer.launch_passive(p.model.ptr, p.data.ptr) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = th
        viewer.cam.distance = 1.5
        viewer.cam.elevation = -35
        viewer.cam.azimuth = 150
        while viewer.is_running():
            t0 = time.time()
            obs, r, term, trunc, info = env.step(policy(obs))
            total += r
            if time.time() - last_print > 0.5:
                v = float(np.linalg.norm(p.data.qvel[:2]))
                print(f'command={env._cmd:4.1f}cm/s speed={v:5.1f}cm/s return={total:7.1f} '
                      f'gates={info["gates"]}', flush=True)
                last_print = time.time()
            if term or trunc:
                print(f'--- episode end: {info.get("reason")} return={total:.1f}', flush=True)
                obs, _ = env.reset()
                total = 0.
            viewer.sync()
            time.sleep(max(env.env.control_timestep() * args.slowmo - (time.time() - t0), 0.001))


if __name__ == '__main__':
    main()
