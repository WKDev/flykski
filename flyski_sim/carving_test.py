# 몸통 롤을 외부 토크로 고정한 채 경사면을 내려가며 엣지각에 따라 궤적이 휘는지(카빙) 측정하는 테스트
"""카빙 물리 검사(RESEARCH_NOTES 35번).

컨트롤러 없이 스키-눈 물리만 보려고, 좌/우 부츠(T2 바인딩) roll 힌지의 springref를
목표 엣지각으로 돌려 판을 세운다(부츠 캔팅). 요(yaw)는 자유다. 처음엔 몸통을 외부
토크로 기울였는데 다리가 휘면서 몸만 옆으로 가고 판은 눈에 누운 채였고(몸 롤 11°에
판 엣지 1.3°), 판 자체에 PD 토크를 걸면 판 관성이 너무 작아 발산했다. 실제 주행에서
엣징은 다리가 판을 기울여야 생긴다(정책의 몫).
평평한 경사면(알파인, 코듀로이 높이 0)에서 초파리는 폴라인(+x)을 향해 출발한다.

    python -m flyski_sim.carving_test            # 엣지각 스윕 표 + 판정
    python -m flyski_sim.carving_test --gif      # renders/carving/ 에 GIF 저장

측정값
- turn_deg: 스키가 눈에 닿아 있던 스텝들의 스키 요 변화 합(+면 왼쪽, -면 오른쪽으로 돎).
  착지 중 요 드리프트와 공중에 뜬 구간은 빼고 잰다.
- contact_frac: 측정 구간 중 스키가 눈에 닿아 있던 스텝 비율.
- skid_ratio: 스키 조각 속도 중 판 횡방향 성분 비율의 평균(0=완전 카빙, 1=옆으로만 밀림).
  접촉 중이고 5cm/s 이상 움직인 스텝만.
- edge_deg: 판 z축과 지면 법선 사이 평균 각.
- yaw_rate_ratio: 접촉 중이고 5cm/s 이상 움직인 스텝의 (스키 요레이트 x 예측반경 / 속도)
  중앙값. 카빙 호를 정확히 따라가면 오른쪽 엣지 -1, 왼쪽 엣지 +1.
- radius_cm: 예측반경 / |yaw_rate_ratio| (실측 회전 반경), predicted_cm:
  R_sidecut * cos(엣지각) (edge_grip.py 카빙 호 모델이 따라가야 하는 값).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from dm_control import composer

from flybody.fruitfly import fruitfly
from flyski_sim.ski_profiles import SKI_PROFILES
from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.terrain import SlopedMoguls

SLOPE_DEG = 15.
_KP, _KD = 1.0, 0.006        # 몸통 롤 PD(dyn*cm/rad, dyn*cm*s/rad). 롤 관성 ~8.4e-6 g*cm^2 기준.
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'renders', 'carving')


def build(profile='all_mountain', edge_grip=True, seed=3):
    arena = SlopedMoguls(dim=(30., 15.), mean_slope_deg=SLOPE_DEG, terrain_type='alpine',
                         mogul_height=0., grid_density=15)
    task = SlopeSmokeTask(ski_profile=SKI_PROFILES[profile], walker=fruitfly.FruitFly,
                          arena=arena, time_limit=1e9, joint_filter=0., claw_friction=1.0,
                          edge_grip=edge_grip)
    env = composer.Environment(time_limit=1e9, task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True,
                               recompile_mjcf_every_episode=False)
    return env, task


def _slope_normal():
    t = np.deg2rad(SLOPE_DEG)
    return np.array([np.sin(t), 0., np.cos(t)])    # 높이가 +x로 갈수록 낮아지는 면.


def roll_about_normal(R: np.ndarray, n: np.ndarray) -> float:
    """R 좌표계(몸통 또는 판)의 롤(rad): +면 오른쪽(-y)이 내려감 = 오른쪽 엣지."""
    return float(np.arctan2(R[:, 1] @ n, R[:, 2] @ n))


def run(target_edge_deg: float, profile='all_mountain', edge_grip=True, seconds=1.0,
        settle=0.05, frames: list | None = None):
    env, task = build(profile, edge_grip)
    env.reset()
    p = env.physics
    th = p.model.name2id('walker/thorax', 'body')
    roots = [p.model.name2id(u.root_body.full_identifier, 'body') for u in task.skis.values()]
    rolls = [p.model.name2id(f'walker/ski_bind_{s}_roll', 'joint') for s in task.skis]
    # 부츠 roll +φ(x축 둘레) = 판 왼쪽(+y)이 들림 = 오른쪽 엣지. 목표 엣지각 +는 오른쪽
    # 엣지(오른쪽 = -y로 돌아야 함, 스키 요 감소). 좌/우 판을 같은 쪽 엣지로 세우고
    # 몸통은 세워 둔다(PD 목표 0). 카빙에 필요한 몸 기울기는 tan(lean) = v^2/(gR)인데
    # 초파리 스케일(v~15cm/s, R~9cm)에선 ~1.5°뿐이다(낮은 프루드 수). 몸을 엣지각만큼
    # 기울였더니 과도한 안쪽 중력 성분 때문에 예측보다 3~4배 급하게 돌았다
    # (RESEARCH_NOTES 35번). 즉 이 스케일의 카빙 자세 = 몸은 거의 세우고 다리로 판을 세움.
    n = _slope_normal()
    nu = env.action_spec().shape[0]
    dt = env.control_timestep()
    skid, edge = [], []
    ds_sum = dyaw_sum = 0.
    n_contact = n_meas = 0
    ratios = []                                      # 요레이트 * 예측반경 / 속도.
    prev_yaw = None
    xy = []
    for k in range(int(seconds / dt)):
        # 착지(settle) 동안은 0°, 이후 목표 엣지각으로.
        target = np.deg2rad(target_edge_deg) if k * dt > settle else 0.
        p.model.qpos_spring[p.model.jnt_qposadr[rolls]] = target
        R = p.data.xmat[th].reshape(3, 3)
        w = p.data.cvel[th][:3] @ R[:, 0]
        p.data.xfrc_applied[th, 3:6] = (_KP * (0. - roll_about_normal(R, n)) - _KD * w) * R[:, 0]
        env.step(np.zeros(nu))
        m = task.ski_metrics(p)
        yaw = float(np.mean([np.arctan2(p.data.xmat[b][3], p.data.xmat[b][0]) for b in roots]))
        if k * dt > settle + 0.1:
            n_meas += 1
            if m['skid_ratio'] is not None:          # 이 스텝에 스키가 눈에 닿아 있었다.
                n_contact += 1
                edge.append(m['edge_deg'])
                dyaw = float(np.angle(np.exp(1j * (yaw - prev_yaw))))
                v = float(np.linalg.norm(p.data.qvel[:3]))
                ds_sum += v * dt
                dyaw_sum += dyaw
                if v > 5.:                           # 거의 멈춘 채 도는 건 반경/skid가 무의미.
                    skid.append(m['skid_ratio'])
                    pred = task.skis['left'].sidecut_radius_cm * np.cos(np.deg2rad(m['edge_deg']))
                    ratios.append(dyaw / dt * pred / v)
        prev_yaw = yaw
        xy.append(p.data.xpos[th][:2].copy())
        if frames is not None and k % 10 == 0:
            frames.append(_frame(p, th))
    r_sc = task.skis['left'].sidecut_radius_cm
    edge_mean = float(np.mean(edge)) if edge else 0.
    R = p.data.xmat[th].reshape(3, 3)
    return dict(target_edge_deg=target_edge_deg,
                body_roll_deg=round(float(np.degrees(roll_about_normal(R, n))), 1),
                plate_roll_deg=[round(float(np.degrees(roll_about_normal(
                    p.data.xmat[b].reshape(3, 3), n))), 1) for b in roots],
                edge_deg=round(edge_mean, 1),
                contact_frac=round(n_contact / max(n_meas, 1), 2),
                turn_deg=round(float(np.degrees(dyaw_sum)), 1),
                path_cm=round(ds_sum, 1),
                yaw_rate_ratio=round(float(np.median(ratios)), 2) if ratios else None,
                radius_cm=(round(r_sc * np.cos(np.deg2rad(edge_mean)) / abs(float(np.median(ratios))), 1)
                           if ratios and abs(np.median(ratios)) > 1e-3 else None),
                predicted_cm=round(r_sc * np.cos(np.deg2rad(edge_mean)), 1),
                skid_ratio=round(float(np.mean(skid)), 2) if skid else None,
                speed_cm_s=round(float(np.linalg.norm(p.data.qvel[:3])), 1),
                upright=bool(R[:, 2] @ n > 0.5),
                _xy=np.array(xy))


def _frame(p, th):
    from dm_control.mujoco.engine import MovableCamera
    cam = MovableCamera(p, height=360, width=480)
    cam.set_pose(p.data.xpos[th], 1.3, 90, -75)       # 거의 위에서, +x(폴라인)가 화면 오른쪽.
    return cam.render()


def _plot_paths(rows, grip, profile):
    """엣지각별 몸통 궤적(위에서 본 모습)을 한 장에 그린다."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        xy = r['_xy'] - r['_xy'][0]
        ax.plot(xy[:, 0], xy[:, 1], label=f"edge {r['target_edge_deg']:+.0f} deg")
    ax.set_xlabel('downhill x (cm)')
    ax.set_ylabel('y (cm, + = left)')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_title(f"{profile}, edge grip {'ON' if grip else 'OFF'}, slope {SLOPE_DEG:.0f} deg (1 s)")
    name = f"paths_{'grip' if grip else 'nogrip'}_{profile}.png"
    fig.savefig(os.path.join(OUT_DIR, name), dpi=120, bbox_inches='tight')
    print('saved', name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='all_mountain')
    ap.add_argument('--no-grip', action='store_true')
    ap.add_argument('--gif', action='store_true')
    ap.add_argument('--seconds', type=float, default=1.0)
    args = ap.parse_args()
    grip = not args.no_grip
    print(f'profile={args.profile} edge_grip={grip} slope={SLOPE_DEG}deg')
    rows = []
    for e in (0., 10., 20., 30., -20.):
        frames = [] if args.gif and e in (0., 20., -20.) else None
        r = run(e, args.profile, grip, args.seconds, frames=frames)
        rows.append(r)
        print({k: v for k, v in r.items() if not k.startswith('_')})
        if frames:
            from PIL import Image
            os.makedirs(OUT_DIR, exist_ok=True)
            name = f"{'grip' if grip else 'nogrip'}_{args.profile}_edge{int(e)}.gif"
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(os.path.join(OUT_DIR, name), save_all=True, append_images=imgs[1:],
                         duration=60, loop=0)
            print('saved', name)
    by = {r['target_edge_deg']: r for r in rows}
    if args.gif:
        _plot_paths(rows, grip, args.profile)
    q20, qm20 = by[20.]['yaw_rate_ratio'], by[-20.]['yaw_rate_ratio']
    sign_ok = q20 is not None and qm20 is not None and q20 < 0 < qm20
    grip_ok = all(by[e]['skid_ratio'] < 0.3 for e in (20., -20.))
    radius_ok = sign_ok and all(0.5 < abs(q) < 2.5 for q in (q20, qm20))
    print('오른쪽 엣지(+)는 오른쪽(-), 왼쪽 엣지(-)는 왼쪽(+)으로?', bool(sign_ok),
          '| 엣지 ±20°에서 skid<0.3?', grip_ok, '| 요레이트가 카빙 호의 0.5~2.5배?', radius_ok)
    return sign_ok and grip_ok and radius_ok

if __name__ == '__main__':
    raise SystemExit(0 if main() else 1)
