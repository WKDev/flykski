# SkiCourseEnv를 PPO(병렬 env)로 정해진 시간만큼 학습하고 학습곡선/롤아웃 GIF를 남기는 스크립트
"""S자 카빙 RL 가능성 확인용 1회 학습(RESEARCH_NOTES 36번).

    python -m flyski_sim.train_rl --minutes 35 --envs 10                  # 코스(게이트/카빙)
    python -m flyski_sim.train_rl --stage speed --log-std -2 --name speed1 # 커리큘럼 1단계(정지/출발)

제대로 된 학습이 아니라 "파이프라인이 돌고 보상이 오르는가"만 본다. 결과는 runs/<이름>/:
progress.csv(롤아웃마다 평균 에피소드 보상/게이트/카빙), model.zip, curve.png,
rollout_zero.gif / rollout_policy.gif(0 액션 vs 학습된 정책, 결정적).
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs')
INFO_KEYS = ('gates', 'misses', 'carve', 'steps', 'track', 'reason')


def make_env(rank: int, stage: str):
    def _f():
        from stable_baselines3.common.monitor import Monitor
        from flyski_sim.rl_task import ENVS
        return Monitor(ENVS[stage](seed=rank), info_keywords=INFO_KEYS)
    return _f


class LogCallback:
    """롤아웃이 끝날 때마다 최근 에피소드 통계를 CSV에 쓰고 시간 예산을 넘기면 멈춘다."""

    def __init__(self, path, deadline):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self

        class _CB(BaseCallback):
            def _on_step(self):
                for info in self.locals['infos']:
                    ep = info.get('episode')
                    if ep:
                        outer.buf.append((ep['r'], ep['l'], info.get('gates', 0),
                                          info.get('misses', 0), info.get('carve', 0.),
                                          info.get('track', 0.) / max(ep['l'], 1),
                                          info.get('reason', '')))
                return time.time() < deadline

            def _on_rollout_end(self):
                outer.flush(self.num_timesteps)

        self.cb = _CB()
        self.buf = []
        self.path = path
        self.t0 = time.time()
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(['timesteps', 'minutes', 'episodes', 'mean_return', 'mean_len',
                                    'mean_gates', 'mean_misses', 'mean_carve', 'mean_speed_err',
                                    'fall_frac'])

    def flush(self, timesteps):
        if not self.buf:
            return
        r, l, g, m, c, e, reason = zip(*self.buf)
        row = [timesteps, round((time.time() - self.t0) / 60, 2), len(r), np.mean(r), np.mean(l),
               np.mean(g), np.mean(m), np.mean(c), np.mean(e), np.mean([x != 'time' for x in reason])]
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        print('ts={} min={} eps={} return={:.2f} len={:.0f} gates={:.2f} misses={:.2f} carve={:.2f} '
              'speed_err={:.2f} fall={:.2f}'.format(*row), flush=True)
        self.buf = []


def rollout_gif(env, policy, path, every=3):
    from PIL import Image
    obs, _ = env.reset(seed=123)
    frames, total, info = [], 0., {}
    for k in range(10 ** 6):
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        if k % every == 0:
            frames.append(Image.fromarray(env.render()))
        if term or trunc:
            break
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    return total, info


def plot_curve(csv_path, out, stage='course'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(df.timesteps, df.mean_return)
    axes[0].plot(df.timesteps, df.mean_len / 10, alpha=0.5, label='episode length / 10')
    axes[0].legend()
    axes[0].set_title('mean episode return')
    if stage == 'speed':
        axes[1].plot(df.timesteps, df.mean_speed_err)
        axes[1].set_title('mean |speed - command| (cm/s)')
    else:
        axes[1].plot(df.timesteps, df.mean_gates, label='gates passed')
        axes[1].plot(df.timesteps, df.mean_misses, label='gates missed')
        axes[1].legend()
        axes[1].set_title('gates / episode')
    axes[2].plot(df.timesteps, df.mean_carve, label='carve quality sum')
    axes[2].plot(df.timesteps, 100 * df.fall_frac, label='fall %')
    axes[2].legend()
    axes[2].set_title('carving / falls')
    for a in axes:
        a.set_xlabel('policy steps')
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=35.)
    ap.add_argument('--envs', type=int, default=10)
    ap.add_argument('--name', default='ppo_try1')
    ap.add_argument('--stage', default='course', choices=('course', 'speed'))
    ap.add_argument('--log-std', type=float, default=-1.0,
                    help='초기 탐색 노이즈 log 표준편차(액션 범위 1.0rad인 speed 단계는 -2 권장)')
    args = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from flyski_sim.rl_task import ENVS
    torch.set_num_threads(2)

    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    venv = SubprocVecEnv([make_env(i, args.stage) for i in range(args.envs)])
    model = PPO('MlpPolicy', venv, n_steps=256, batch_size=640, n_epochs=5, learning_rate=3e-4,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=args.log_std), verbose=0, seed=0)
    log = LogCallback(os.path.join(out, 'progress.csv'), time.time() + 60 * args.minutes)
    model.learn(total_timesteps=10 ** 8, callback=log.cb)
    log.flush(model.num_timesteps)
    model.save(os.path.join(out, 'model.zip'))
    venv.close()
    plot_curve(os.path.join(out, 'progress.csv'), os.path.join(out, 'curve.png'), args.stage)

    env = ENVS[args.stage](seed=123)
    zero = rollout_gif(env, lambda o: np.zeros(env.action_space.shape), os.path.join(out, 'rollout_zero.gif'))
    pol = rollout_gif(env, lambda o: model.predict(o, deterministic=True)[0],
                      os.path.join(out, 'rollout_policy.gif'))
    for label, (ret, info) in (('zero', zero), ('policy', pol)):
        print(f'[{label}] return={ret:.2f} gates={info.get("gates")} misses={info.get("misses")} '
              f'carve={info.get("carve", 0):.2f} speed_err_sum={info.get("track", 0):.1f} '
              f'steps={info.get("steps")} reason={info.get("reason")}')
    print('total policy steps', model.num_timesteps)


if __name__ == '__main__':
    main()
