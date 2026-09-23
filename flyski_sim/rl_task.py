# 냄새 게이트 S자 코스를 카빙으로 내려가는 강화학습 환경(gymnasium): 감각 관측 + 보상 설계
"""S자 카빙 활강 RL 환경(RESEARCH_NOTES 36번, 설계: prompts/06_rl_carving_design.md).

- 코스: 평평한 경사면(알파인, 코듀로이 0), 폴라인 = +x. 순차 점화 냄새 게이트
  (`odor.OdorField`, 좌우 교대)가 S자 경로를 만든다. 사람이 궤적을 정하지 않는다.
- 액션: 다리 주 관절 42개의 "스키 스탠스 대비 오프셋", [-1, 1] * ACTION_SCALE rad.
  날개/머리/복부/흡착은 0. 스탠스 오프셋 해석은 SlopeSmokeTask가 한다.
- 관측(감각 되먹임, 모두 몸 기준 좌표 또는 스칼라):
  고유감각(다리 관절각-스탠스, 관절속도), 전정(몸 기준 중력 방향, 각속도, 선속도),
  스키 감각(좌/우 부호 있는 엣지각, 옆미끄럼 비율, 접지 여부, 그립 사용률),
  후각(좌/우 더듬이 농도, 좌우 차, 직전 대비 변화), 속도 명령(목표 속력/20), 직전 액션.
  게이트 위치 같은 특권 정보는 넣지 않는다(후각으로만 찾아야 함).
- 보상: 아래 REWARD_WEIGHTS와 `_task_reward` 참고.
- 커리큘럼(RESEARCH_NOTES 37번): 단계마다 관측/액션 공간이 같아서 앞 단계 정책을 이어서
  학습할 수 있다. 1단계 `SpeedControlEnv`(정지/출발, 플루크 제동), 이후 `SkiCourseEnv`.

    python -m flyski_sim.rl_task [course|speed]   # 0/무작위 액션으로 한 에피소드씩 보상 항목 출력
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
ACTION_SCALE = 0.3           # 액션 1 = 스탠스에서 0.3rad(PPO 1회차 값). 쐐기 20°에 관절이
                             # 최대 0.78rad 움직여야 해서 커리큘럼 단계는 1.0을 쓴다.
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

    episode_seconds = EPISODE_SECONDS

    def __init__(self, profile: str = 'all_mountain', seed: int = 0,
                 action_scale: float = ACTION_SCALE):
        self.task = SkiCourseTask(profile, seed)
        self._action_scale = action_scale
        self._cmd = 15.                  # 속도 명령(cm/s). 코스 단계는 "진행" 고정.
        self.env = composer.Environment(task=self.task, time_limit=self.episode_seconds,
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
        obs_dim = 2 * n_act + 9 + 8 + 4 + 1 + 2 + n_act
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
            [self._cmd / 20.],
            self._track_obs(),
            self._prev_a])
        return np.nan_to_num(obs).astype(np.float32)

    # ---- gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.task.ski_metrics(self.env.physics)
        self._prev_a[:] = 0.
        self._odor_prev = self._odor(self.env.physics).mean()
        self._stats = dict(gates=0, misses=0, carve=0., steps=0, track=0.)
        self._on_reset()
        metrics = self.task.ski_metrics(self.env.physics)
        return self._obs(metrics, self._odor(self.env.physics), 0.), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1., 1.)
        full = np.zeros(self._nu)
        full[self._act_idx] = self._action_scale * a
        self._on_step()
        p = self.env.physics
        try:
            ts = self.env.step(full)
        except PhysicsError:
            obs = np.zeros(self.observation_space.shape, np.float32)
            return obs, self.fall_penalty, True, False, dict(self._stats, reason='physics_error')
        metrics = self.task.ski_metrics(p)
        odor = self._odor(p)
        r, parts, terminated, reason = self._reward(p, a, metrics, odor)
        finished = reason == 'finish'
        if finished:
            terminated = False                  # 완주는 실패가 아니라 시간 제한과 같은 절단.
        d_odor = odor.mean() - self._odor_prev
        self._odor_prev = odor.mean()
        self._prev_a = a
        self._stats['steps'] += 1
        truncated = (bool(ts.last()) or finished) and not terminated
        info = dict(self._stats, parts=parts)
        if terminated or truncated:
            info['reason'] = reason or 'time'
        return self._obs(metrics, odor, d_odor), float(r), terminated, truncated, info

    # ---- 단계별 훅 ----
    def _track_obs(self):
        """목표 궤적 대비 (방향 오차 rad, 옆 오차 cm/3). 목표 궤적이 없는 단계는 0."""
        return [0., 0.]

    def _on_reset(self):
        pass

    def _on_step(self):
        pass

    # ---- 보상/종료 ----
    fall_penalty = REWARD_WEIGHTS['fall']

    def _reward(self, p, a, metrics, odor):
        parts = self._task_reward(p, a, metrics, odor)
        reason = self._termination(p)
        if reason and reason != 'finish':
            parts['fall'] = self.fall_penalty
        return sum(parts.values()), parts, reason is not None, reason

    def _termination(self, p):
        """넘어짐, 몸(스키 외) 접지, 코스 이탈이면 이유 문자열, 아니면 None."""
        d = p.data
        pos = d.xpos[self._th]
        up = float(d.xmat[self._th].reshape(3, 3)[:, 2] @ self.task._slope_n)
        body_touch = any((c.geom1 in self._body_geoms and p.model.geom_bodyid[c.geom2] == 0) or
                         (c.geom2 in self._body_geoms and p.model.geom_bodyid[c.geom1] == 0)
                         for c in d.contact[:d.ncon])
        dim = COURSE['dim']
        if up < 0.5:
            return 'fallen'
        if body_touch:
            return 'body_touch'
        if pos[0] > dim[0] - 1.:
            return 'finish'                    # 코스 끝 도달 = 완주(벌점 없음, 40번 이전엔
                                               # off_course로 넘어짐과 같은 벌점을 받았다).
        if abs(pos[1]) > dim[1] - 0.5:
            return 'off_course'
        return None

    def _task_reward(self, p, a, metrics, odor):
        """코스 단계: 게이트/카빙/엣지 회전/주행/후각 셰이핑."""
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
        return parts

    def render(self):
        from dm_control.mujoco.engine import MovableCamera
        p = self.env.physics
        cam = MovableCamera(p, height=360, width=480)
        cam.set_pose(p.data.xpos[self._th], 3.0, 90, -60)
        return cam.render()


SPEED_WEIGHTS = dict(
    track=1.0,           # exp(-((속력 - 명령)/SPEED_SIGMA)^2).
    lateral=-0.02,       # 옆(y) 속도 cm/s 당. 폴라인에서 크게 벗어나지 않게.
    heading=-0.5,        # (1 - cos 몸 요각) 당. speed1은 몸을 사면에 가로로 돌려 속도를
                         # 조절했다(플루크가 아님, 37번). 폴라인을 보고 쐐기로 서게 한다.
    alive=0.05,
    ctrl=-0.01,
    fall=-20.,           # PPO 1회차는 -5가 카빙 보상에 묻혀 넘어짐이 ~70%였다.
)
SPEED_SIGMA = 4.               # cm/s
SPEED_COMMANDS = (0., 5., 10., 15.)
SPEED_HOLD = (0.6, 1.2)        # 명령 유지 시간(s) 범위.
MAX_YAW_DEG = 70.              # 몸이 폴라인에서 이만큼 넘게 돌면 종료(가로 서기 금지).


class SpeedControlEnv(SkiCourseEnv):
    """커리큘럼 1단계: 속도 명령(정지/천천히/출발) 따라가기. 플루크 제동을 배우는 단계.

    15° 사면에서 스탠스 그대로면 ~18cm/s로 내려가므로, 정지/저속 명령을 지키려면
    쐐기(플루크)나 엣지로 제동해야 한다(snowplow_test: 쐐기 20°면 ~2cm/s). 명령은
    SPEED_HOLD 간격으로 무작위로 바뀌고, 첫 명령은 항상 0(출발 전 정지)이다.
    """

    episode_seconds = 4.0
    fall_penalty = SPEED_WEIGHTS['fall']

    def __init__(self, profile: str = 'all_mountain', seed: int = 0, action_scale: float = 1.0):
        super().__init__(profile, seed, action_scale)
        self._rng = np.random.RandomState(seed + 1000)

    def _on_reset(self):
        self._cmd = 0.
        self._t = 0.
        self._next_switch = self._rng.uniform(*SPEED_HOLD)

    def _on_step(self):
        self._t += CONTROL_DT
        if self._t >= self._next_switch:
            self._cmd = float(self._rng.choice([c for c in SPEED_COMMANDS if c != self._cmd]))
            self._next_switch = self._t + self._rng.uniform(*SPEED_HOLD)

    def _task_reward(self, p, a, metrics, odor):
        w = SPEED_WEIGHTS
        v = p.data.qvel[:3]
        err = float(np.linalg.norm(v[:2])) - self._cmd
        self._stats['track'] += abs(err)
        return dict(track=w['track'] * float(np.exp(-(err / SPEED_SIGMA) ** 2)),
                    lateral=w['lateral'] * abs(float(v[1])),
                    heading=w['heading'] * (1. - np.cos(self._yaw(p))),
                    alive=w['alive'], ctrl=w['ctrl'] * float(np.mean(a ** 2)))

    def _yaw(self, p):
        R = p.data.xmat[self._th].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))       # 폴라인(+x) 대비 몸 요.

    def _termination(self, p):
        reason = super()._termination(p)
        if reason is None and abs(self._yaw(p)) > np.deg2rad(MAX_YAW_DEG):
            return 'traverse'
        return reason


class SCurve:
    """폴라인(+x)을 따라 반복되는 완만한 S자 목표 궤적 y(x) = A sin(2 pi (x - x0) / L)."""

    def __init__(self, x0: float, y0: float = 0., amplitude: float = 3., wavelength: float = 40.):
        self.x0, self.y0, self.a, self.k = x0, y0, amplitude, 2 * np.pi / wavelength

    def y(self, x):
        return self.y0 + self.a * np.sin(self.k * (x - self.x0))

    def heading(self, x):
        """목표 진행 방향(rad, + = 왼쪽)."""
        return float(np.arctan(self.a * self.k * np.cos(self.k * (x - self.x0))))


TURN_WEIGHTS = dict(
    lateral=1.0,         # exp(-(옆 오차/TURN_SIGMA_Y)^2).
    heading=0.5,         # cos(방향 오차).
    speed=0.3,           # exp(-((속력 - 명령)/SPEED_SIGMA)^2).
    alive=0.05,
    ctrl=-0.01,
    fall=-20.,
)
TURN_SIGMA_Y = 1.0             # cm
INITIAL_SPEED = 8.             # cm/s, 폴라인 방향 초기 속도.
TURN_MAX_HEADING_ERR_DEG = 80.


class TurnTrackEnv(SpeedControlEnv):
    """커리큘럼 2단계(플루크 보겐/슈템): S자 목표 궤적 따라가기. 속도 명령은 10cm/s 고정.

    관측의 목표 오차 2칸(방향 오차, 옆 오차)은 상위 내비게이터가 주는 명령에 해당한다.
    나중엔 후각(게이트)이 이 목표를 대신한다.
    """

    episode_seconds = 5.0
    fall_penalty = TURN_WEIGHTS['fall']

    def _on_reset(self):
        self._cmd = 10.
        p = self.env.physics
        pos = p.data.xpos[self._th]
        # 에피소드마다 S자 방향/진폭/파장을 무작위로(궤적 하나만 외우지 않게).
        amp = self._rng.uniform(2., 4.) * self._rng.choice([-1., 1.])
        self.reference = SCurve(float(pos[0]), float(pos[1]), amp, self._rng.uniform(35., 50.))
        # 브레이크로 시작하지 않고 11자로 이미 달리는 상태에서 시작(사용자 요청, 39번).
        slope = np.deg2rad(COURSE['slope_deg'])
        p.data.qvel[:3] = INITIAL_SPEED * np.array([np.cos(slope), 0., -np.sin(slope)])

    def _on_step(self):
        pass

    def _errors(self, p):
        pos = p.data.xpos[self._th]
        v = p.data.qvel[:2]
        head = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 3. else self._yaw(p)
        e_psi = float(np.angle(np.exp(1j * (self.reference.heading(pos[0]) - head))))
        e_y = float(self.reference.y(pos[0]) - pos[1])
        return e_psi, e_y

    def _track_obs(self):
        if not hasattr(self, 'reference'):
            return [0., 0.]
        e_psi, e_y = self._errors(self.env.physics)
        return [e_psi, e_y / 3.]

    def _task_reward(self, p, a, metrics, odor):
        w = TURN_WEIGHTS
        e_psi, e_y = self._errors(p)
        speed = float(np.linalg.norm(p.data.qvel[:2]))
        self._stats['track'] += abs(e_y)
        return dict(lateral=w['lateral'] * float(np.exp(-(e_y / TURN_SIGMA_Y) ** 2)),
                    heading=w['heading'] * float(np.cos(e_psi)),
                    speed=w['speed'] * float(np.exp(-((speed - self._cmd) / SPEED_SIGMA) ** 2)),
                    alive=w['alive'], ctrl=w['ctrl'] * float(np.mean(a ** 2)))

    def _termination(self, p):
        reason = SkiCourseEnv._termination(self, p)
        # 착지(처음 0.2초) 동안은 방향 오차로 끝내지 않는다(착지 충격에 요가 튐).
        if (reason is None and self._stats['steps'] > 20
                and abs(self._errors(p)[0]) > np.deg2rad(TURN_MAX_HEADING_ERR_DEG)):
            return 'heading'
        return reason


PARALLEL_BONUS = 0.4
PARALLEL_MAX_WEDGE_DEG = 5.
PARALLEL_MIN_EDGE_DEG = 1.5
PARALLEL_TURN_HEADING_DEG = 10.   # 목표 진행각이 이보다 클 때(S자의 꺾이는 구간)만 판정.


class ParallelTrackEnv(TurnTrackEnv):
    """커리큘럼 3단계(패럴렐 턴): TurnTrackEnv + 패럴렐 자세 보상.

    패럴렐 = S자의 꺾이는 구간(목표 진행각 > 10°)에서 두 판 방향 차 < 5°이고 두 판 엣지가
    같은 쪽(부호 같음, 각 1.5° 이상). 직진 구간에선 두 판이 안쪽 엣지로 살짝 선(A자) 게
    정상이라 판정하지 않는다. stats['carve']에 패럴렐 스텝 수를, stats['turn_steps']에
    판정 대상 스텝 수를 누적한다(로그 carve 칸 = 패럴렐 스텝 수).
    """

    def _plate_state(self, p):
        n = self.task._slope_n
        yaws, edges = [], []
        for b in self._roots:
            R = p.data.xmat[b].reshape(3, 3)
            yaws.append(np.arctan2(R[1, 0], R[0, 0]))
            edges.append(np.arctan2(R[:, 1] @ n, R[:, 2] @ n))
        wedge = abs(float(np.angle(np.exp(1j * (yaws[0] - yaws[1])))))
        return wedge, edges

    def _task_reward(self, p, a, metrics, odor):
        parts = super()._task_reward(p, a, metrics, odor)
        x = float(p.data.xpos[self._th][0])
        if abs(self.reference.heading(x)) < np.deg2rad(PARALLEL_TURN_HEADING_DEG):
            parts['parallel'] = 0.
            return parts
        wedge, (e_l, e_r) = self._plate_state(p)
        parallel = (wedge < np.deg2rad(PARALLEL_MAX_WEDGE_DEG) and np.sign(e_l) == np.sign(e_r)
                    and min(abs(e_l), abs(e_r)) > np.deg2rad(PARALLEL_MIN_EDGE_DEG))
        parts['parallel'] = PARALLEL_BONUS * float(parallel)
        self._stats['carve'] += float(parallel)
        self._stats['turn_steps'] = self._stats.get('turn_steps', 0) + 1
        return parts


ENVS = {'course': SkiCourseEnv, 'speed': SpeedControlEnv, 'turn': TurnTrackEnv,
        'parallel': ParallelTrackEnv}


def main():
    import sys
    import time
    env = ENVS[sys.argv[1] if len(sys.argv) > 1 else 'course']()
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
              f'mean|speed-cmd|={info["track"] / max(n, 1):.1f} '
              f'{n / (time.time() - t0):.1f} policy steps/s')
        print('   ', {k: round(v, 2) for k, v in sums.items()})
    print('obs dim', env.observation_space.shape, 'act dim', env.action_space.shape)


if __name__ == '__main__':
    main()
