# 냄새 게이트 S자 코스를 카빙으로 내려가는 강화학습 환경(gymnasium): 감각 관측 + 보상 설계
"""S자 카빙 활강 RL 환경(RESEARCH_NOTES 36번, 설계: prompts/06_rl_carving_design.md).

- 코스: 평평한 경사면(알파인, 코듀로이 0), 폴라인 = +x. 순차 점화 냄새 게이트
  (`odor.OdorField`, 좌우 교대)가 S자 경로를 만든다. 사람이 궤적을 정하지 않는다.
- 액션: 다리 주 관절 42개의 "스키 스탠스 대비 오프셋", [-1, 1] * ACTION_SCALE rad.
  날개/머리/복부/흡착은 0. 스탠스 오프셋 해석은 SlopeSmokeTask가 한다.
- 관측(감각 되먹임, 모두 몸 기준 좌표 또는 스칼라):
  고유감각(다리 관절각-스탠스, 관절속도), 전정(몸 기준 중력 방향, 각속도, 선속도),
  스키 감각(좌/우 부호 있는 엣지각, 옆미끄럼 비율, 접지 여부, 그립 사용률),
  후각(좌/우 더듬이 농도, 좌우 차, 직전 대비 변화), 직전 액션.
  게이트 위치 같은 특권 정보는 넣지 않는다(후각으로만 찾아야 함).
- 보상: 아래 REWARD_WEIGHTS와 `_reward` 참고.

    python -m flyski_sim.rl_task      # 무작위/0 액션으로 한 에피소드 돌려 보상 항목 출력
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from dm_control import composer
from dm_control.rl.control import PhysicsError

from flybody.fruitfly import fruitfly
from flyski_sim.odor import OdorField
from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls

CONTROL_DT = 0.01            # 정책 주기(s). 물리 2e-4 x 50 서브스텝.
ACTION_SCALE = 0.3           # 액션 1 = 스탠스에서 0.3rad.
EPISODE_SECONDS = 3.0
COURSE = dict(dim=(30., 8.), slope_deg=15., gate_spacing=8., gate_amplitude=3.,
              pass_radius=1.5, spawn_margin=2.)

REWARD_WEIGHTS = dict(
    progress=0.005,      # x 방향(폴라인) 속도 cm/s 당. 0.02면 직진만으로 게이트 보상을 압도했다.
    odor=10.0,           # 두 더듬이 평균 농도 증가량(퍼텐셜 차) 당.
    gate=10.0,           # 활성 게이트 통과 1회.
    gate_miss=-3.0,      # 게이트를 통과 못 하고 지나침.
    carve=0.3,           # 카빙 품질(0~1): 접지+속도 있을 때 (1-skid) x 엣지 세기.
    edge_turn=0.5,       # 카빙 방향(엣지 쪽)이 다음 게이트 쪽과 같을 때 카빙 품질 x 이 값.
    alive=0.05,          # 서 있는 매 스텝.
    ctrl=-0.01,          # 액션 제곱 평균.
    fall=-5.0,           # 종료(넘어짐/몸 접지/코스 이탈) 1회.
)


class SkiCourseTask(SlopeSmokeTask):
    """게이트 추적/보상 계산을 붙인 SlopeSmokeTask."""

    def __init__(self, profile: str = 'all_mountain', seed: int = 0):
        c = COURSE
        arena = SlopedMoguls(dim=c['dim'], mean_slope_deg=c['slope_deg'], terrain_type='alpine',
                             mogul_height=0., grid_density=15)
        super().__init__(ski_profile=SKI_PROFILES[profile], walker=fruitfly.FruitFly,
                         arena=arena, time_limit=EPISODE_SECONDS, joint_filter=0.,
                         claw_friction=1.0, spawn_xy=(-c['dim'][0] + c['spawn_margin'], 0.))
        self.set_timesteps(control_timestep=CONTROL_DT, physics_timestep=2e-4)
        self.odor = OdorField(gate_spacing_cm=c['gate_spacing'], gate_amplitude_cm=c['gate_amplitude'],
                              arena_dim=c['dim'], pass_radius_cm=c['pass_radius'], seed=seed)
        self._slope_n = np.array([np.sin(np.deg2rad(c['slope_deg'])), 0.,
                                  np.cos(np.deg2rad(c['slope_deg']))])

    # 게이트: 스폰 지점보다 앞(아래)에 있는 것만 쓴다.
    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        self.odor.reset()
        x0 = -COURSE['dim'][0] + COURSE['spawn_margin']
        while self.odor.active_gate is not None and self.odor.active_gate.center_xy[0] < x0 + 2.:
            self.odor.advance_gate(self.odor.active_gate.gate_id)

    def get_reward_factors(self, physics):
        return (1.,)                      # 보상은 gym 래퍼(SkiCourseEnv)가 계산한다.

    def check_termination(self, physics):
        return False                      # 종료도 래퍼가 판단(flybody 기본 qacc 종료 끔).


class SkiCourseEnv(gym.Env):
    """gymnasium 래퍼. 관측/보상/종료를 계산한다."""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, profile: str = 'all_mountain', seed: int = 0):
        self.task = SkiCourseTask(profile, seed)
        self.env = composer.Environment(task=self.task, time_limit=EPISODE_SECONDS,
                                        random_state=np.random.RandomState(seed),
                                        strip_singleton_obs_buffer_dim=True,
                                        recompile_mjcf_every_episode=False)
        self.env.reset()
        p = self.env.physics
        m = p.model
        names = self.env.action_spec().name.split('\t')
        stance = self.task._stance
        self._leg_names = [n for n in names if n in stance]
        self._act_idx = np.array([names.index(n) for n in self._leg_names])
        self._nu = len(names)
        jids = [m.name2id(f'walker/{n}', 'joint') for n in self._leg_names]
        self._qadr = m.jnt_qposadr[jids]
        self._vadr = m.jnt_dofadr[jids]
        self._stance_q = np.array([stance[n] for n in self._leg_names])
        self._th = m.name2id('walker/thorax', 'body')
        self._ant = [m.name2id(f'walker/antenna_{s}', 'body') for s in ('left', 'right')]
        self._roots = [m.name2id(self.task.skis[s].root_body.full_identifier, 'body')
                       for s in ('left', 'right')]
        ski_geoms = {m.name2id(g.full_identifier, 'geom') for u in self.task.skis.values()
                     for g in u.geoms}
        walker_geoms = {i for i in range(m.ngeom)
                        if (m.id2name(i, 'geom') or '').startswith('walker/')}
        self._body_geoms = walker_geoms - ski_geoms
        n_act = len(self._leg_names)
        self.action_space = gym.spaces.Box(-1., 1., (n_act,), np.float32)
        obs_dim = 2 * n_act + 9 + 8 + 4 + n_act
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self._prev_a = np.zeros(n_act)

    # ---- 감각 ----
    def _odor(self, p):
        return np.array([self.task.odor.sample(tuple(p.data.xpos[b]))[0] for b in self._ant])

    def _ski_sense(self, p, metrics):
        out = []
        n = self.task._slope_n
        for s, b in zip(('left', 'right'), self._roots):
            R = p.data.xmat[b].reshape(3, 3)
            edge = np.arctan2(R[:, 1] @ n, R[:, 2] @ n)        # +: 오른쪽 엣지.
            per = metrics['per_ski'].get(s)
            if per:
                out += [edge, per['skid_ratio'], 1., per['grip_utilization']]
            else:
                out += [edge, 0., 0., 0.]
        return np.array(out)

    def _obs(self, metrics, odor, d_odor):
        p = self.env.physics
        d = p.data
        R = d.xmat[self._th].reshape(3, 3)
        grav_body = R.T @ np.array([0., 0., -1.])
        angvel = R.T @ d.cvel[self._th][:3]
        linvel = R.T @ d.qvel[:3]
        obs = np.concatenate([
            d.qpos[self._qadr] - self._stance_q, 0.05 * d.qvel[self._vadr],
            grav_body, 0.05 * angvel, 0.05 * linvel,
            self._ski_sense(p, metrics),
            [odor[0], odor[1], 10. * (odor[0] - odor[1]), 10. * d_odor],
            self._prev_a])
        return np.nan_to_num(obs).astype(np.float32)

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.task.ski_metrics(self.env.physics)
        self._prev_a[:] = 0.
        self._odor_prev = self._odor(self.env.physics).mean()
        self._stats = dict(gates=0, misses=0, carve=0., steps=0)
        metrics = self.task.ski_metrics(self.env.physics)
        return self._obs(metrics, self._odor(self.env.physics), 0.), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1., 1.)
        full = np.zeros(self._nu)
        full[self._act_idx] = ACTION_SCALE * a
        p = self.env.physics
        try:
            ts = self.env.step(full)
        except PhysicsError:
            obs = np.zeros(self.observation_space.shape, np.float32)
            return obs, REWARD_WEIGHTS['fall'], True, False, dict(self._stats, reason='physics_error')
        metrics = self.task.ski_metrics(p)
        odor = self._odor(p)
        r, parts, terminated, reason = self._reward(p, a, metrics, odor)
        d_odor = odor.mean() - self._odor_prev
        self._odor_prev = odor.mean()
        self._prev_a = a
        self._stats['steps'] += 1
        truncated = bool(ts.last()) and not terminated
        info = dict(self._stats, parts=parts)
        if terminated or truncated:
            info['reason'] = reason or 'time'
        return self._obs(metrics, odor, d_odor), float(r), terminated, truncated, info

    # ---- 보상/종료 ----
    def _reward(self, p, a, metrics, odor):
        w = REWARD_WEIGHTS
        d = p.data
        pos = d.xpos[self._th]
        v = d.qvel[:3]
        speed = float(np.linalg.norm(v))
        parts = dict(progress=w['progress'] * float(v[0]),
                     odor=w['odor'] * (odor.mean() - self._odor_prev),
                     alive=w['alive'], ctrl=w['ctrl'] * float(np.mean(a ** 2)),
                     gate=0., carve=0., edge_turn=0.)
        # 게이트 통과/놓침.
        gate = self.task.odor.active_gate
        if gate is not None:
            if self.task.odor.try_advance((float(pos[0]), float(pos[1]))):
                parts['gate'] = w['gate']
                self._stats['gates'] += 1
            elif pos[0] > gate.center_xy[0] + 1.:
                self.task.odor.advance_gate(gate.gate_id)
                parts['gate'] = w['gate_miss']
                self._stats['misses'] += 1
        # 카빙 품질과 엣지 방향 회전.
        if metrics['skid_ratio'] is not None and speed > 5.:
            edge_deg = metrics['edge_deg']
            q = (1. - min(metrics['skid_ratio'], 1.)) * float(np.clip((edge_deg - 5.) / 15., 0., 1.))
            parts['carve'] = w['carve'] * q
            self._stats['carve'] += q
            gate = self.task.odor.active_gate
            if gate is not None and q > 0.:
                n = self.task._slope_n
                edges = [np.arctan2(p.data.xmat[b].reshape(3, 3)[:, 1] @ n,
                                    p.data.xmat[b].reshape(3, 3)[:, 2] @ n) for b in self._roots]
                carve_dir = -np.sign(np.mean(edges))            # +: 왼쪽으로 도는 카빙.
                to_gate = np.array(gate.center_xy) - pos[:2]
                want = np.sign(v[0] * to_gate[1] - v[1] * to_gate[0])   # +: 게이트가 왼쪽.
                if carve_dir == want:
                    parts['edge_turn'] = w['edge_turn'] * q
        # 종료: 넘어짐, 몸(스키 외) 접지, 코스 이탈.
        up = float(d.xmat[self._th].reshape(3, 3)[:, 2] @ self.task._slope_n)
        body_touch = any((c.geom1 in self._body_geoms and p.model.geom_bodyid[c.geom2] == 0) or
                         (c.geom2 in self._body_geoms and p.model.geom_bodyid[c.geom1] == 0)
                         for c in d.contact[:d.ncon])
        dim = COURSE['dim']
        off = abs(pos[1]) > dim[1] - 0.5 or pos[0] > dim[0] - 1.
        reason = None
        if up < 0.5:
            reason = 'fallen'
        elif body_touch:
            reason = 'body_touch'
        elif off:
            reason = 'off_course'
        if reason:
            parts['fall'] = w['fall']
        return sum(parts.values()), parts, reason is not None, reason

    def render(self):
        from dm_control.mujoco.engine import MovableCamera
        p = self.env.physics
        cam = MovableCamera(p, height=360, width=480)
        cam.set_pose(p.data.xpos[self._th], 3.0, 90, -60)
        return cam.render()


def main():
    import time
    env = SkiCourseEnv()
    for label, policy in (('zero', lambda: np.zeros(env.action_space.shape)),
                          ('random', lambda: env.action_space.sample())):
        obs, _ = env.reset(seed=0)
        total, sums, t0, n = 0., {}, time.time(), 0
        while True:
            obs, r, term, trunc, info = env.step(policy())
            total += r
            n += 1
            for k, v in info['parts'].items():
                sums[k] = sums.get(k, 0.) + v
            if term or trunc:
                break
        print(f'[{label}] steps={n} return={total:.2f} reason={info["reason"]} gates={info["gates"]} '
              f'misses={info["misses"]} carve_sum={info["carve"]:.2f} '
              f'{n / (time.time() - t0):.1f} policy steps/s')
        print('   ', {k: round(v, 2) for k, v in sums.items()})
    print('obs dim', env.observation_space.shape, 'act dim', env.action_space.shape)


if __name__ == '__main__':
    main()
