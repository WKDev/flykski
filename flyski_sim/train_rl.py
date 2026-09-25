# SkiCourseEnv를 PPO(병렬 env)로 정해진 시간만큼 학습하고 학습곡선/롤아웃 GIF를 남기는 스크립트
"""S자 카빙 RL 가능성 확인용 1회 학습(RESEARCH_NOTES 36번).

    python -m flyski_sim.train_rl --minutes 35 --envs 10                  # 코스(게이트/카빙)
    python -m flyski_sim.train_rl --stage speed --log-std -2 --name speed1 # 커리큘럼 1단계(정지/출발)
    python -m flyski_sim.train_rl --stage speed --bc-episodes 40 --name speed3  # 플루크 전문가 모방 후 PPO

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


BEST_WINDOW = 50            # model_best.zip 판정에 쓰는 최근 에피소드 수.


class LogCallback:
    """롤아웃이 끝날 때마다 최근 에피소드 통계를 CSV에 쓰고 시간 예산을 넘기면 멈춘다."""

    def __init__(self, path, deadline, save_every_min=10.):
        from stable_baselines3.common.callbacks import BaseCallback

        outer = self
        self.save_every = 60 * save_every_min
        self.last_save = time.time()

        class _CB(BaseCallback):
            def _on_step(self):
                for info in self.locals['infos']:
                    ep = info.get('episode')
                    if ep:
                        outer.buf.append((ep['r'], ep['l'], info.get('gates', 0),
                                          info.get('misses', 0), info.get('carve', 0.),
                                          info.get('track', 0.) / max(ep['l'], 1),
                                          info.get('reason', ''), info.get('arc', 0.)))
                return time.time() < deadline

            def _on_rollout_end(self):
                outer.flush(self.num_timesteps)
                # 학습 도중에도 play로 볼 수 있게 주기적으로 최신 모델 저장.
                if time.time() - outer.last_save > outer.save_every:
                    self.model.save(os.path.join(os.path.dirname(outer.path), 'model_latest.zip'))
                    outer.last_save = time.time()
                # 최근 BEST_WINDOW 에피소드 평균 수익이 최고면 따로 저장(gs2/gs3가 도중에 무너져
                # 마지막 모델이 최고가 아니었다, 46번).
                recent = outer.returns[-BEST_WINDOW:]
                if len(recent) >= BEST_WINDOW and np.mean(recent) > outer.best:
                    outer.best = float(np.mean(recent))
                    self.model.save(os.path.join(os.path.dirname(outer.path), 'model_best.zip'))
                    print(f'best {outer.best:.1f} at ts={self.num_timesteps}', flush=True)

        self.cb = _CB()
        self.buf = []
        self.returns = []
        self.best = -np.inf
        self.path = path
        self.t0 = time.time()
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(['timesteps', 'minutes', 'episodes', 'mean_return', 'mean_len',
                                    'mean_gates', 'mean_misses', 'mean_carve', 'mean_speed_err',
                                    'fall_frac', 'mean_arc'])

    def flush(self, timesteps):
        if not self.buf:
            return
        r, l, g, m, c, e, reason, arc = zip(*self.buf)
        self.returns = (self.returns + list(r))[-BEST_WINDOW:]
        row = [timesteps, round((time.time() - self.t0) / 60, 2), len(r), np.mean(r), np.mean(l),
               np.mean(g), np.mean(m), np.mean(c), np.mean(e), np.mean([x not in ('time', 'finish') for x in reason]),
               np.mean(arc)]
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
        print('ts={} min={} eps={} return={:.2f} len={:.0f} gates={:.2f} misses={:.2f} carve={:.2f} '
              'speed_err={:.2f} fall={:.2f} arc={:.1f}'.format(*row), flush=True)
        self.buf = []


def _collect_expert(args):
    """워커: 플루크 전문가 시범 수집. 실행은 노이즈 섞은 액션, 기록은 전문가 액션(DART)."""
    stage, seed, n_eps, table, noise = args
    from flyski_sim.experts import expert_for
    from flyski_sim.rl_task import ENVS
    env = ENVS[stage](seed=seed)
    expert = expert_for(stage)[1](env, table)
    rng = np.random.RandomState(seed)
    obs_l, act_l, rets = [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=seed * 1000 + ep)
        total = 0.
        while True:
            a = expert(obs)
            obs_l.append(obs)
            act_l.append(a)
            obs, r, term, trunc, _ = env.step(np.clip(a + noise * rng.randn(*a.shape), -1, 1))
            total += r
            if term or trunc:
                break
        rets.append(total)
    return np.array(obs_l, np.float32), np.array(act_l, np.float32), rets


def collect_expert(stage, n_eps, n_workers, out, noise=0.1):
    """단계별 전문가 시범을 병렬 수집(학습 워커를 띄우기 전에 끝낸다, 메모리)."""
    import multiprocessing as mp
    from flyski_sim.experts import expert_for
    from flyski_sim.rl_task import ENVS
    table = expert_for(stage)[0](ENVS[stage](seed=0))
    per = max(n_eps // n_workers, 1)
    with mp.get_context('spawn').Pool(n_workers) as pool:
        parts = pool.map(_collect_expert, [(stage, 100 + i, per, table, noise) for i in range(n_workers)])
    obs = np.concatenate([o for o, _, _ in parts])
    act = np.concatenate([a for _, a, _ in parts])
    rets = [r for _, _, rr in parts for r in rr]
    print(f'BC data: {len(obs)} samples from {len(rets)} expert episodes, expert return {np.mean(rets):.1f}',
          flush=True)
    np.savez_compressed(os.path.join(out, 'bc_data.npz'), obs=obs, act=act)
    return obs, act


def behavior_clone(model, obs, act, epochs=30):
    """전문가 시범으로 PPO 정책 평균을 지도학습(MSE)해 워밍업한다."""
    import torch
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    o_t, a_t = torch.as_tensor(obs), torch.as_tensor(act)
    for ep in range(epochs):
        perm = torch.randperm(len(o_t))
        losses = []
        for i in range(0, len(o_t), 256):
            idx = perm[i:i + 256]
            mean = pol.get_distribution(o_t[idx]).distribution.mean
            loss = ((mean - a_t[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'BC epoch {ep} mse {np.mean(losses):.4f}', flush=True)


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
    ap.add_argument('--envs', type=int, default=0,
                    help='병렬 환경 수. 0이면 코어 수 - 2(이 PC 22). 워커는 수치 라이브러리 스레드 1개')
    ap.add_argument('--name', default='ppo_try1')
    ap.add_argument('--stage', default='course', choices=('course', 'speed', 'turn', 'parallel', 'residual', 'carve', 'race'))
    ap.add_argument('--init', default=None, help='앞 단계 model.zip 가중치로 시작(커리큘럼)')
    ap.add_argument('--bc-episodes', type=int, default=0,
                    help='>0이면 플루크 전문가 시범으로 행동 복제 워밍업 후 PPO(speed 단계)')
    ap.add_argument('--log-std', type=float, default=-1.0,
                    help='초기 탐색 노이즈 log 표준편차(액션 범위 1.0rad인 speed 단계는 -2 권장)')
    ap.add_argument('--lr', type=float, default=3e-4, help='학습률(미세조정은 1e-4 이하 권장, 46번)')
    ap.add_argument('--target-kl', type=float, default=None,
                    help='업데이트당 KL 상한(넘으면 그 롤아웃의 남은 epoch 중단). 0.02 정도면 붕괴 방지')
    ap.add_argument('--gamma', type=float, default=0.99,
                    help='할인율. 0.99 = 약 1초 앞(100스텝). race는 0.995(턴 하나)를 권장(48번)')
    args = ap.parse_args()
    if args.envs <= 0:
        args.envs = max((os.cpu_count() or 4) - 2, 1)
    # 워커마다 numpy/torch가 코어 수만큼 스레드를 띄우면 환경 20여 개가 서로 경합한다(45번).
    # spawn 워커는 부모 환경 변수를 물려받으므로 워커 생성 전에 1로 고정.
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ.setdefault(var, '1')

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from flyski_sim.rl_task import ENVS
    torch.set_num_threads(2)

    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    if args.bc_episodes:
        bc_obs, bc_act = collect_expert(args.stage, args.bc_episodes, args.envs, out)
    venv = SubprocVecEnv([make_env(i, args.stage) for i in range(args.envs)])
    model = PPO('MlpPolicy', venv, n_steps=256, batch_size=640, n_epochs=5, learning_rate=args.lr,
                target_kl=args.target_kl,
                gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=args.log_std), verbose=0, seed=0)
    if args.init:
        prev = PPO.load(args.init, device='cpu')
        model.policy.load_state_dict(prev.policy.state_dict())
        print(f'init from {args.init}', flush=True)
    if args.bc_episodes:
        behavior_clone(model, bc_obs, bc_act)
        env = ENVS[args.stage](seed=7)
        for ep in range(3):
            obs, _ = env.reset(seed=ep)
            tot, n = 0., 0
            while True:
                obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
                tot += r
                n += 1
                if term or trunc:
                    break
            print(f'[BC policy] ep{ep} return={tot:.1f} steps={n} reason={info["reason"]} '
                  f'mean|speed-cmd|={info["track"] / n:.1f}', flush=True)
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
