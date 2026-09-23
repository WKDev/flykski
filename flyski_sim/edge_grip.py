# 엣지가 눈에 박히는 효과(카빙 그립)를 스키 조각별 횡방향 저항력으로 근사하는 현상론 모델
"""엣지 그립 모델(RESEARCH_NOTES 35번).

MuJoCo 접촉은 강체 지면 + 방향 무관(등방) 마찰이라, 엣지가 눈을 파고들어 선반을
만들고 옆으로는 버티면서 앞으로는 미끄러지는 현상이 나오지 않는다. 그래서 스키 조각이
지면과 닿는 접촉마다 판 횡방향으로만 저항력을 추가한다.

    F_lat = -mu_edge(phi, 설질) * N * tanh(v_lat / V0) * t_lat
    mu_edge = snow.edge_grip * clip((|phi| - PHI0) / (PHI1 - PHI0), 0, 1)

- phi: 조각의 엣지각(조각 y축이 지면에서 들린 각), N: 접촉 수직력,
  v_lat: 접촉점 속도의 횡방향 성분, t_lat: 아래 "카빙 호"의 접촉점 접선에 수직인 방향.
- 필요한 횡력이 mu_edge*N을 넘으면 tanh가 포화해서 옆으로 밀린다(스키딩).
- 카빙 호: 실제 스키는 엣징되면 사이드컷 때문에 허리가 떠 있다가 하중에 휘어, 엣지가
  반경 R_turn = R_sidecut * cos(phi)인 호를 그리고 그 호를 따라 간다. 처음엔 판 조각이
  하중에 휘는 것만으로 호가 생기길 기대했지만, 초파리 스케일 판의 휨이 수 um라서 호가
  사실상 없었고 스키는 "가리키는 방향으로만" 갔다(좌/우 엣지 모두 같은 쪽으로 돎,
  RESEARCH_NOTES 35번). 그래서 접촉점의 판 길이 방향 위치 s에서 접선을 판 축에서
  s / R_turn만큼(엣지 쪽으로) 돌린 방향으로 그립한다. R_sidecut은 판 실제 형상에서
  구한다(`ski_plate.plate_sidecut_radius_cm`).
- 앞뒤(판 길이) 방향은 기존 MuJoCo 마찰(설질별 ski_friction)만 받는다.
- 힘은 접촉점에 걸되 스키 조각이 아니라 몸통(thorax, 몸 전체의 자유 관절 루트)에
  건다(mj_applyFT가 접촉점 기준 모멘트까지 넣어 준다). 조각 질량(~1e-5g)에 속도비례
  힘을 명시적으로 걸면 μN/V0 감쇠가 너무 세서 바로 발산했다. 대가로 그립 횡력이 판을
  직접 휘게 하지는 않는다(판 휨은 수직하중으로만 생김).
- 접촉당 N은 몸무게의 N_CAP_BODYWEIGHTS배로 자른다. 착지 충격 때 mj_contactForce의 N이
  치솟으면 그립력(∝N)이 접촉력과 양의 되먹임을 일으켜 0.017초 만에 발산했다.
가정이지 물리 법칙이 아니므로 carving_test.py로 엣지각-회전 관계를 확인하고 쓴다.
"""
from __future__ import annotations

import mujoco
import numpy as np

PHI0 = np.deg2rad(3.)    # 이보다 평평하면 그립 없음(플랫 스키).
PHI1 = np.deg2rad(20.)   # 이 이상이면 그립 최대.
V0 = 2.0                 # cm/s, 정지마찰을 부드럽게 근사하는 속도 스케일. 0.5면 과도응답이 커서
                         # 예측보다 2~4배 급하게 돌았다(RESEARCH_NOTES 35번).
N_CAP_BODYWEIGHTS = 2.   # 접촉 하나의 수직력 상한(몸무게 배수).


class EdgeGrip:
    """스키 geom 접촉을 매 물리 서브스텝마다 훑어 횡방향 그립을 qfrc_applied에 넣는다."""

    def __init__(self, physics, units: dict, snow_at, enabled: bool = True,
                 target_body: str = 'walker/thorax'):
        m = physics.model
        self._enabled = enabled
        self._target = m.name2id(target_body, 'body')
        weight = float(m.body_subtreemass[self._target] * np.linalg.norm(m.opt.gravity))
        self._n_cap = N_CAP_BODYWEIGHTS * weight
        self._snow_at = snow_at            # x(cm) -> SnowParams
        self._geom_side = {}
        self._root = {}
        self._r_sidecut = {}
        for side, u in units.items():
            for g in u.geoms:
                self._geom_side[m.name2id(g.full_identifier, 'geom')] = side
            self._root[side] = m.name2id(u.root_body.full_identifier, 'body')
            self._r_sidecut[side] = u.sidecut_radius_cm
        self._sides = tuple(units)
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)
        self.reset_metrics()

    def reset_metrics(self):
        self._acc = {s: dict(n=0, lat=0., tan=0., edge=0., util=0.) for s in self._sides}

    def step(self, physics):
        m, d = physics.model.ptr, physics.data.ptr
        d.qfrc_applied[:] = 0.
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self._geom_side and m.geom_bodyid[c.geom2] == 0:
                g, sign = c.geom1, -1.
            elif c.geom2 in self._geom_side and m.geom_bodyid[c.geom1] == 0:
                g, sign = c.geom2, 1.
            else:
                continue
            n = sign * c.frame[:3]                   # 지면 -> 스키 방향 법선.
            mujoco.mj_contactForce(m, d, i, self._f6)
            N = min(max(self._f6[0], 0.), self._n_cap)
            side = self._geom_side[g]
            ey = d.geom_xmat[g].reshape(3, 3)[:, 1]
            phi = float(np.arcsin(min(abs(ey @ n), 1.)))
            # 카빙 호의 접선: 판 축(지면 투영)을 접촉점 위치 s만큼 엣지 쪽으로 돌린다.
            # ey*n > 0 = 왼쪽(+y)이 들림 = 오른쪽 엣지 = 오른쪽(시계방향, 곡률 -)으로 돎.
            root = self._root[side]
            ex = d.xmat[root].reshape(3, 3)[:, 0]
            ex = ex - (ex @ n) * n
            ex /= max(np.linalg.norm(ex), 1e-9)
            r_sc = self._r_sidecut[side]
            kappa = -np.sign(ey @ n) / (r_sc * np.cos(phi)) if r_sc > 0 else 0.
            theta = kappa * float((c.pos - d.xpos[root]) @ ex)
            tangent = np.cos(theta) * ex + np.sin(theta) * np.cross(n, ex)
            t = np.cross(n, tangent)                 # 호의 왼쪽 법선.
            body = m.geom_bodyid[g]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, body, self._v6, 0)
            v = self._v6[3:] + np.cross(self._v6[:3], c.pos - d.xpos[body])
            v_t = v - (v @ n) * n
            v_lat = float(v_t @ t)
            snow = self._snow_at(float(c.pos[0]))
            mu = snow.edge_grip * float(np.clip((phi - PHI0) / (PHI1 - PHI0), 0., 1.))
            if self._enabled and mu > 0. and N > 0.:
                f = -mu * N * np.tanh(v_lat / V0) * t
                mujoco.mj_applyFT(m, d, f, np.zeros(3), c.pos, self._target, d.qfrc_applied)
            a = self._acc[side]
            a['n'] += 1
            a['lat'] += abs(v_lat)
            a['tan'] += float(np.linalg.norm(v_t))
            a['edge'] += phi
            a['util'] += abs(np.tanh(v_lat / V0)) if mu > 0. else 1.

    def metrics(self) -> dict:
        """마지막 reset_metrics 이후 누적 평균. 접촉이 없던 스키는 None."""
        out = {}
        for s, a in self._acc.items():
            if a['n'] == 0:
                out[s] = None
                continue
            out[s] = dict(skid_ratio=a['lat'] / max(a['tan'], 1e-9),
                          edge_deg=float(np.degrees(a['edge'] / a['n'])),
                          grip_utilization=a['util'] / a['n'], n_contacts=a['n'])
        return out
