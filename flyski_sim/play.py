# RL 환경(속도 제어/코스)을 스크립트 플루크 컨트롤러나 학습된 PPO 정책으로 실시간 3D 창에 재생하는 뷰어
"""정책 실시간 재생.

    python -m flyski_sim.play --stage speed --controller expert       # 플루크 전문가(쐐기 + 요 제어)
    python -m flyski_sim.play --stage speed --controller snowplow     # 스크립트 플루크(명령 따라 쐐기)
    python -m flyski_sim.play --stage speed --model runs/speed1/model.zip
    python -m flyski_sim.play --stage course --model runs/ppo_try1/model.zip

창에 목표(DESIRED, 초록)와 실제(ACTUAL, 빨강) 궤적을 그린다.
- speed 단계: 목표 = 출발점에서 폴라인(+x)으로 곧게 내려가는 선, 초록 공 = 속도 명령을
  정확히 따랐다면 지금 있어야 할 위치(명령 속력 적분). 빨간 선/공 = 실제 몸 궤적/위치.
- turn 단계: 목표 = S자 궤적(초록 공 = 지금 x에서 목표 y).
- course 단계: 목표 = 스폰 -> 냄새 게이트 중심들을 잇는 선(노란 공 = 게이트, 활성은 크게).
화면 왼쪽 위에 명령/실제 속력, 목표 대비 앞뒤/옆 오차가 뜬다. 에피소드가 끝나면 자동
리셋. macOS는 mjpython으로 실행.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from flyski_sim.rl_task import CONTROL_DT, ENVS
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


GREEN = np.array([0.1, 0.9, 0.2, 0.9], np.float32)
RED = np.array([1.0, 0.1, 0.1, 0.9], np.float32)
YELLOW = np.array([1.0, 0.85, 0.0, 0.9], np.float32)
_LIFT = 0.03          # 선을 설면 위로 띄우는 높이(cm).
_R_LINE = 0.006       # 선 굵기(반지름, cm).
_TRAIL_EVERY = 3      # 몇 스텝마다 실제 궤적 점을 찍나.
_TRAIL_MAX = 600


class TrajectoryOverlay:
    """viewer.user_scn에 목표/실제 궤적을 그린다."""

    def __init__(self, env):
        self.env = env
        self.arena = env.task._arena
        self.reset()

    def reset(self):
        pos = self.env.env.physics.data.xpos[self.env._th]
        self.start = pos[:2].copy()
        self.desired_x = float(pos[0])
        self.trail = [pos[:2].copy()]
        self.k = 0

    def _pt(self, xy):
        x, y = float(xy[0]), float(xy[1])
        return np.array([x, y, self.arena.height_at(x, y) + _LIFT])

    def step(self):
        self.desired_x += self.env._cmd * CONTROL_DT
        self.k += 1
        if self.k % _TRAIL_EVERY == 0:
            self.trail.append(self.env.env.physics.data.xpos[self.env._th][:2].copy())
            self.trail = self.trail[-_TRAIL_MAX:]

    def _seg(self, scn, a, b, rgba, r=_R_LINE):
        if scn.ngeom >= min(scn.maxgeom, len(scn.geoms)):
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, a, b)
        scn.ngeom += 1

    def _ball(self, scn, c, r, rgba):
        if scn.ngeom >= min(scn.maxgeom, len(scn.geoms)):
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([r, 0., 0.]), c, np.eye(3).ravel(), rgba)
        scn.ngeom += 1

    def _polyline(self, scn, pts, rgba):
        for a, b in zip(pts[:-1], pts[1:]):
            self._seg(scn, a, b, rgba)

    def draw(self, scn, append: bool = False):
        """그리고 나서 현재 목표 위치(xy)를 돌려준다. append=True면 기존 geom 뒤에 덧붙인다
        (오프스크린 카메라 장면에는 모델 geom이 이미 들어 있음)."""
        if not append:
            scn.ngeom = 0
        env = self.env
        cur = env.env.physics.data.xpos[env._th][:2]
        if hasattr(env, 'reference'):                    # turn 단계: S자 목표 궤적.
            ref = env.reference
            end_x = float(self.arena._dim[0]) - 1.
            xs = np.linspace(self.start[0], end_x, 120)
            self._polyline(scn, [self._pt((x, ref.y(x))) for x in xs], GREEN)
            desired = np.array([cur[0], ref.y(cur[0])])
            self._ball(scn, self._pt(desired), 0.05, GREEN)
        elif hasattr(env, '_next_switch'):               # speed 단계: 폴라인 + 명령 적분 위치.
            end_x = float(self.arena._dim[0]) - 1.
            xs = np.linspace(self.start[0], end_x, 40)
            self._polyline(scn, [self._pt((x, self.start[1])) for x in xs], GREEN)
            self._ball(scn, self._pt((self.desired_x, self.start[1])), 0.05, GREEN)
            desired = np.array([self.desired_x, self.start[1]])
        else:                                            # course 단계: 게이트를 잇는 선.
            odor = env.task.odor
            gates = [g for g in odor.get_gate_sequence() if g.center_xy[0] > self.start[0]]
            self._polyline(scn, [self._pt(self.start)] + [self._pt(g.center_xy) for g in gates], GREEN)
            active = odor.active_gate
            for g in gates:
                big = active is not None and g.gate_id == active.gate_id
                self._ball(scn, self._pt(g.center_xy), 0.12 if big else 0.06, YELLOW)
            desired = np.array(active.center_xy) if active is not None else cur.copy()
        self._polyline(scn, [self._pt(q) for q in self.trail] + [self._pt(cur)], RED)
        self._ball(scn, self._pt(cur), 0.05, RED)
        return desired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='speed', choices=tuple(ENVS))
    ap.add_argument('--controller', default='model', choices=('model', 'expert', 'snowplow', 'zero'))
    ap.add_argument('--model', default=None)
    ap.add_argument('--slowmo', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    env = ENVS[args.stage](seed=args.seed)
    if args.model:
        from stable_baselines3 import PPO
        from flyski_sim.record import _compat
        model = PPO.load(args.model, device='cpu')
        policy = _compat(model, env)
    elif args.controller == 'expert':
        from flyski_sim.experts import SnowplowExpert
        policy = SnowplowExpert(env)
    elif args.controller == 'snowplow':
        policy = snowplow_controller(env)
    else:
        policy = lambda obs: np.zeros(env.action_space.shape)

    obs, _ = env.reset(seed=args.seed)
    p = env.env.physics
    overlay = TrajectoryOverlay(env)
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
            overlay.step()
            total += r
            v = float(np.linalg.norm(p.data.qvel[:2]))
            with viewer.lock():
                desired = overlay.draw(viewer.user_scn)
            err = p.data.xpos[th][:2] - desired
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              'command\nspeed\nalong err\nlateral err\nreturn',
                              f'{env._cmd:.1f} cm/s\n{v:.1f} cm/s\n{err[0]:+.2f} cm\n'
                              f'{err[1]:+.2f} cm\n{total:.1f}'))
            if time.time() - last_print > 0.5:
                print(f'command={env._cmd:4.1f}cm/s speed={v:5.1f}cm/s along_err={err[0]:+.2f} '
                      f'lateral_err={err[1]:+.2f} return={total:7.1f} gates={info["gates"]}', flush=True)
                last_print = time.time()
            if term or trunc:
                print(f'--- episode end: {info.get("reason")} return={total:.1f}', flush=True)
                obs, _ = env.reset()
                overlay.reset()
                total = 0.
            viewer.sync()
            time.sleep(max(env.env.control_timestep() * args.slowmo - (time.time() - t0), 0.001))


if __name__ == '__main__':
    main()
