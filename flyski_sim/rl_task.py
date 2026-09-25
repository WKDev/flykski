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
from flyski_sim import ski_stance
from flyski_sim.odor import OdorField
from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls

CONTROL_DT = 0.01            # 정책 주기(s). 물리 PHYSICS_DT x 서브스텝.
PHYSICS_DT = 2e-4            # 물리 타임스텝(s). 4e-4는 쐐기/볼록성 검사는 통과했지만 대회전 전문가 생존 7/8 -> 2/8로 망가져 되돌림(46번).
ACTION_SCALE = 0.3           # 액션 1 = 스탠스에서 0.3rad(PPO 1회차 값). 쐐기 20°에 관절이
                             # 최대 0.78rad 움직여야 해서 커리큘럼 단계는 1.0을 쓴다.
EPISODE_SECONDS = 3.0
# 코스(반길이, 반폭) cm. 41번에 60x8, 사용자 요청으로 200x50(400cm x 100cm)으로 크게.
# 격자는 cm당 15점 유지: 평면이어도 2점(0.5cm 칸)이면 판 조각(0.06cm)보다 칸이 커서 접촉이
# 들쭉날쭉해져 전문가가 코스를 벗어났다. 15점(격자 900만 개)도 생성 7초라 괜찮음(41번).
COURSE = dict(dim=(200., 50.), grid_density=15, slope_deg=20., gate_spacing=8., gate_amplitude=3.,
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
                             mogul_height=0., grid_density=c['grid_density'])
        super().__init__(ski_profile=SKI_PROFILES[profile], walker=fruitfly.FruitFly,
                         arena=arena, time_limit=EPISODE_SECONDS, joint_filter=0.,
                         claw_friction=1.0, spawn_xy=(-c['dim'][0] + c['spawn_margin'], 0.))
        self.set_timesteps(control_timestep=CONTROL_DT, physics_timestep=PHYSICS_DT)
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
    # 단계별 스탠스 폭 배율(None이면 ski_stance 기본값). 판은 태스크 생성 때 스탠스로 만들어지고
    # 전문가 IK도 같은 값을 써야 해서 프로세스 전역값(ski_stance.STANCE_WIDTH_SCALE)을 바꾼다.
    # 한 프로세스에서 폭이 다른 환경을 섞어 만들면 안 된다(학습 워커는 환경 하나씩이라 괜찮음).
    stance_width = None

    def __init__(self, profile: str = 'all_mountain', seed: int = 0,
                 action_scale: float = ACTION_SCALE):
        if self.stance_width is not None:
            ski_stance.STANCE_WIDTH_SCALE = self.stance_width
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


class TurnPath:
    """대회전(GS)형 목표 경로. 호 길이 s를 따라 진행 방향이 psi(s) = PSI sin(2 pi s / P)로
    폴라인(+x) 기준 +-PSI까지 번갈아 꺾인다. PSI 70~85°면 턴 하나에 방향이 140~170° 바뀌고
    사면을 가로질렀다(트래버스) 돌아온다. 41번 이전의 S자(y = A sin(kx))는 최대 ~30°라
    직활강에 가까웠다(사용자 지적).
    """

    def __init__(self, x0: float, y0: float, psi_max: float, period: float,
                 length: float = 800., ds: float = 0.05):
        s = np.arange(0., length, ds)
        self.psi = psi_max * np.sin(2 * np.pi * s / period)
        self.x = x0 + np.concatenate([[0.], np.cumsum(np.cos(self.psi[:-1]) * ds)])
        self.yy = y0 + np.concatenate([[0.], np.cumsum(np.sin(self.psi[:-1]) * ds)])
        self.psi_max, self.period = psi_max, period
        self._i = 0

    def y(self, x):
        """같은 x에서 경로 y(그림/로그용, |psi| < 90°라 x는 단조 증가)."""
        return float(np.interp(x, self.x, self.yy))

    def heading(self, x):
        """같은 x에서 경로 진행 방향(rad, + = 왼쪽)."""
        return float(np.interp(x, self.x, self.psi))

    def nearest(self, x, y, window=4000):
        """가장 가까운 경로점의 (진행 방향, 크로스트랙 오차 cm, 점 좌표). 오차 + = 경로가 왼쪽."""
        lo = max(self._i - window // 4, 0)
        hi = min(self._i + window, len(self.x))
        d2 = (self.x[lo:hi] - x) ** 2 + (self.yy[lo:hi] - y) ** 2
        i = lo + int(np.argmin(d2))
        self._i = i
        psi = self.psi[i]
        cross = float(-(self.x[i] - x) * np.sin(psi) + (self.yy[i] - y) * np.cos(psi))
        return float(psi), cross, np.array([self.x[i], self.yy[i]])


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
TURN_PSI_MAX_DEG = (55., 85.)  # 폴라인 대비 최대 진행각(턴당 110~170°). 복구 모드 전 전문가는 70~85에서 첫 턴 뒤
                               # 90° 넘게 돌아 멈췄다. 복구 모드 후엔 70~85도 12초 버팀(43번).
TURN_PERIOD_CM = (50., 80.)    # 경로 호 길이 기준 한 주기(좌+우 턴).
TURN_MAX_HEADING_ERR_DEG = 150.   # 43번 이전 80(멈추면 바로 끝나 복구를 못 배움).
STALL_SECONDS = 2.


class TurnTrackEnv(SpeedControlEnv):
    """커리큘럼 2단계(플루크 보겐/슈템): S자 목표 궤적 따라가기. 속도 명령은 10cm/s 고정.

    관측의 목표 오차 2칸(방향 오차, 옆 오차)은 상위 내비게이터가 주는 명령에 해당한다.
    나중엔 후각(게이트)이 이 목표를 대신한다.
    """

    episode_seconds = 12.0            # 대회전 경로 1.5~2주기(턴 3~4번).
    fall_penalty = TURN_WEIGHTS['fall']

    def _on_reset(self):
        self._cmd = 10.
        self._stall = 0
        p = self.env.physics
        pos = p.data.xpos[self._th]
        # 에피소드마다 첫 턴 방향/최대 진행각/주기를 무작위로(궤적 하나만 외우지 않게).
        psi_max = np.deg2rad(self._rng.uniform(*TURN_PSI_MAX_DEG)) * self._rng.choice([-1., 1.])
        self.reference = TurnPath(float(pos[0]), float(pos[1]), psi_max,
                                  self._rng.uniform(*TURN_PERIOD_CM))
        # 브레이크로 시작하지 않고 11자로 이미 달리는 상태에서 시작(사용자 요청, 39번).
        slope = np.deg2rad(COURSE['slope_deg'])
        p.data.qvel[:3] = INITIAL_SPEED * np.array([np.cos(slope), 0., -np.sin(slope)])

    def _on_step(self):
        pass

    def _errors(self, p):
        pos = p.data.xpos[self._th]
        v = p.data.qvel[:2]
        head = float(np.arctan2(v[1], v[0])) if np.linalg.norm(v) > 3. else self._yaw(p)
        psi_path, cross, _ = self.reference.nearest(float(pos[0]), float(pos[1]))
        e_psi = float(np.angle(np.exp(1j * (psi_path - head))))
        return e_psi, cross

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
        if reason is not None:
            return reason
        # 방향 오차가 커도 바로 끝내지 않고 STALL_SECONDS 넘게 멈춰 있을 때만 끝낸다. 멈춰도
        # 엣지를 풀고 판을 틀어 다시 출발할 수 있어서(복구, 43번) 그걸 배울 기회를 준다.
        # 완전히 거꾸로(방향 오차 > 150°) 가면 끝낸다.
        speed = float(np.linalg.norm(p.data.qvel[:2]))
        self._stall = self._stall + 1 if speed < 1. else 0
        if self._stall > int(STALL_SECONDS / CONTROL_DT):
            return 'stalled'
        if self._stats['steps'] > 20 and abs(self._errors(p)[0]) > np.deg2rad(TURN_MAX_HEADING_ERR_DEG):
            return 'heading'
        return None


PARALLEL_BONUS = 1.0            # 41번 이전 0.4(패럴렐 비율 33%에 그침).
PARALLEL_FULL_EDGE_DEG = 8.     # 두 판 중 얕은 쪽 엣지가 이 각이면 보상 최대.
PARALLEL_MAX_WEDGE_DEG = 5.
PARALLEL_MIN_EDGE_DEG = 1.5
PARALLEL_TURN_HEADING_DEG = 10.   # 목표 진행각이 이보다 클 때(S자의 꺾이는 구간)만 판정.


class ParallelTrackEnv(TurnTrackEnv):
    """커리큘럼 3단계(패럴렐 턴): TurnTrackEnv + 패럴렐 자세 보상. 스탠스 폭 0.85배.

    패럴렐 = S자의 꺾이는 구간(목표 진행각 > 10°)에서 두 판 방향 차 < 5°이고 두 판 엣지가
    같은 쪽(부호 같음, 각 1.5° 이상). 직진 구간에선 두 판이 안쪽 엣지로 살짝 선(A자) 게
    정상이라 판정하지 않는다. stats['carve']에 패럴렐 스텝 수를, stats['turn_steps']에
    판정 대상 스텝 수를 누적한다(로그 carve 칸 = 패럴렐 스텝 수).
    """

    stance_width = 1.0                # 0.85면 패럴렐 비율은 높지만 엣지 0 주행 드리프트가 더 커서(78도/0.7초) 되돌림(42번).

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
        pos = p.data.xpos[self._th]
        if abs(self.reference.nearest(float(pos[0]), float(pos[1]))[0]) < np.deg2rad(PARALLEL_TURN_HEADING_DEG):
            parts['parallel'] = 0.
            return parts
        wedge, (e_l, e_r) = self._plate_state(p)
        parallel = (wedge < np.deg2rad(PARALLEL_MAX_WEDGE_DEG) and np.sign(e_l) == np.sign(e_r)
                    and min(abs(e_l), abs(e_r)) > np.deg2rad(PARALLEL_MIN_EDGE_DEG))
        depth = min(abs(e_l), abs(e_r)) / np.deg2rad(PARALLEL_FULL_EDGE_DEG)
        parts['parallel'] = PARALLEL_BONUS * float(parallel) * float(np.clip(depth, 0., 1.))
        self._stats['carve'] += float(parallel)
        self._stats['turn_steps'] = self._stats.get('turn_steps', 0) + 1
        return parts


CARVE_BONUS = 1.0               # 턴 구간 스텝당 최대 카빙 보상(패럴렐 보상과 같은 크기).
CARVE_SIGMA = 0.5               # ln(턴 반경 / 판 호 반경)의 허용 폭. 0.5면 비율 0.6~1.6에서 보상 0.37 이상.
CARVE_MIN_EDGE_DEG = 5.


class CarveTrackEnv(ParallelTrackEnv):
    """커리큘럼 4단계(카빙): ParallelTrackEnv + "판 호를 따라 돌았나" 보상(47번).

    판이 엣지각 phi로 서면 사이드컷 R_sc 판은 반경 R_sc x cos(phi) 호를 그린다(edge_grip과 같은
    식). 카빙 = 실제 턴 반경(속력 / 요레이트)이 그 호 반경과 같고, 도는 방향이 엣지 쪽. 다리로 판을
    비틀어 도는 턴(gs4: 반경 4cm, 호 14cm)은 비율이 1에서 멀어 보상이 작다. 스키딩 비율은 그립
    모델이 옆미끄럼을 늘 작게 눌러서(0.07) 카빙을 구분하지 못해 쓰지 않았다.
    stats['arc']에 카빙 점수 합을 누적한다.
    """

    def _task_reward(self, p, a, metrics, odor):
        parts = super()._task_reward(p, a, metrics, odor)
        parts['carve'] = 0.
        pos = p.data.xpos[self._th]
        if abs(self.reference.nearest(float(pos[0]), float(pos[1]))[0]) < np.deg2rad(PARALLEL_TURN_HEADING_DEG):
            return parts
        speed = float(np.linalg.norm(p.data.qvel[:2]))
        wz = float(p.data.cvel[self._th][2])
        edge = metrics['edge_deg']
        if edge is None or speed < 5. or abs(wz) < 0.3 or edge < CARVE_MIN_EDGE_DEG:
            return parts
        _, (e_l, e_r) = self._plate_state(p)
        if np.sign(e_l) != np.sign(e_r) or np.sign(wz) != -np.sign(e_l):
            return parts                    # 두 판 엣지 방향이 다르거나 엣지 반대쪽으로 도는 중.
        r_arc = float(np.mean(self.task._grip._rsc)) * np.cos(np.deg2rad(edge))
        q = float(np.exp(-(np.log((speed / abs(wz)) / r_arc) / CARVE_SIGMA) ** 2))
        parts['carve'] = CARVE_BONUS * q
        self._stats['arc'] = self._stats.get('arc', 0.) + q
        return parts


RACE_PROGRESS_PER_CM = 5.       # 경로를 따라 전진한 호 길이 1cm당 보상.
RACE_CORRIDOR_CM = 3.0          # 경로에서 이만큼 넘게 벗어나면 게이트 놓침 = 실격(종료 + 넘어짐 벌점). 1.5면 새 물리(48번)에서 carve1이 8/8 초반 실격.
RACE_CTRL = -0.01
RACE_FAIL_PENALTY = -100.       # 실격/넘어짐. -20(TURN)이면 턴 하나 진행 보상보다 작아 race1/race2가 빠르게 달리다
                                # 통로를 벗어나는 쪽으로 무너졌다(10판 중 7판 실격, 48번).


class RaceTrackEnv(CarveTrackEnv):
    """커리큘럼 5단계(경주, 48번): 보상은 결과만. 경로(게이트 통로)를 벗어나지 않고 빨리 내려가기.

    자세(패럴렐, 엣지각), 카빙(호 반경)에 대한 보상은 없다. 옆미끄럼은 그립 마찰로 에너지를 잃어
    느려지므로, 물리가 맞다면 카빙은 보상 없이도 빠른 방법으로 나와야 한다. 패럴렐/arc 통계는
    비교용으로만 계속 쌓는다(부모 _task_reward를 부르고 보상은 버림). 관측/스탠스는 ParallelTrackEnv와
    같아서 carve1 등의 가중치로 시작할 수 있다. 할인율은 --gamma 0.995(턴 하나 ~2초)로 학습한다.
    """

    fall_penalty = RACE_FAIL_PENALTY

    def _on_reset(self):
        super()._on_reset()
        self._s_prev = 0.

    def _task_reward(self, p, a, metrics, odor):
        super()._task_reward(p, a, metrics, odor)            # 통계(track/carve/arc)만.
        s = self.reference._i * 0.05                       # TurnPath 경로점 간격 ds = 0.05cm.
        ds = float(np.clip(s - self._s_prev, -0.5, 0.5))   # 경로 인덱스가 튀는 경우 방지.
        self._s_prev = s
        return dict(progress=RACE_PROGRESS_PER_CM * ds, ctrl=RACE_CTRL * float(np.mean(a ** 2)))

    def _termination(self, p):
        reason = super()._termination(p)
        if reason is None and abs(self._errors(p)[1]) > RACE_CORRIDOR_CM:
            return 'gate'
        return reason


RESIDUAL_SCALE = 0.3           # 잔차 정책 액션(-1~1)에 곱하는 배율. 최종 = 전문가 + 이 값 x 정책.


class ResidualParallelEnv(ParallelTrackEnv):
    """패럴렐 전문가(복구 모드 포함) 액션 위에 정책이 보정값만 더한다(47번). 정책 출력이 0이면
    전문가 그대로라 학습 시작부터 전문가 수준(대회전 7/8)에서 출발한다. 관측/보상은 ParallelTrackEnv와 같다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from flyski_sim.experts import ParallelExpert
        self.expert = ParallelExpert(self)

    def _on_reset(self):
        super()._on_reset()
        self.expert.last_u = 0.
        self.expert.recovering = 0.

    def step(self, action):
        base = self.expert(None)
        return super().step(np.clip(base + RESIDUAL_SCALE * np.asarray(action, dtype=float), -1., 1.))


ENVS = {'course': SkiCourseEnv, 'speed': SpeedControlEnv, 'turn': TurnTrackEnv,
        'parallel': ParallelTrackEnv, 'residual': ResidualParallelEnv, 'carve': CarveTrackEnv,
        'race': RaceTrackEnv}


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
