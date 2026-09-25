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
# 판이 평평해도(엣지 < PHI0) 주는 기본 횡 그립 비율. 0이면 평평한 스키가 진행 방향으로
# 정렬되려는 힘이 전혀 없어서, 0.81cm 판에선 폴라인으로 미끄러지면서 몸이 0.4초에 70°
# 넘게 돌았다(RESEARCH_NOTES 38번). 실제 스키도 옆미끄럼 저항이 활주 마찰보다 크다(가정값).
FLAT_GRIP_FRACTION = 0.25
# 그립 힘을 걸 대상: 'segment'(접촉한 판 조각, 실제 하중 경로) 또는 'thorax'(몸통 루트).
# thorax는 V0=0.5 시절 조각에 걸면 발산해서 택했지만, 옆 힘이 판->다리->몸 하중 경로를
# 우회해서 사면에서 무게중심을 옮겨도 판 하중이 거의 안 바뀌었다(38번).
APPLY_TO = 'thorax'
# 그립이 버티는 방향(RESEARCH_NOTES 48번).
# - 'formula': 판 축을 접촉 위치 s만큼 s / (R_sidecut x cos phi) 돌린 "카빙 호" 접선(35번의 가정).
# - 'shape': 접촉한 판 조각 자신의 x축(설면 투영). 호는 판이 실제로 휜 모양에서만 나온다.
ARC_SOURCE = 'shape'
# 옆으로 버티는 한계(마찰 계수 꼴, 수직력 N 배수).
# - 'ramp': edge_grip x (FLAT + (1-FLAT) x clip((phi-PHI0)/(PHI1-PHI0))), 20°에서 포화(35번).
# - 'platform': 엣지가 눈을 깎아 만든 선반 역학. 선반이 판 바닥과 나란히 phi만큼 기울어 있으면
#   수직력의 옆 성분 + 마찰로 N x tan(phi + atan(mu_flat))까지 버틴다. 필요한 엣지각은 힘의
#   균형(사면 기울기, 원심력)으로 정해진다. 상한은 눈이 버티는 전단 한계 snow.edge_grip(선반 붕괴).
GRIP_MODEL = 'platform'


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
        # 벡터화용 조회 배열: geom id -> 판 번호(-1 = 스키 아님), 판 번호 -> 루트 body/사이드컷.
        self._side_of = np.full(m.ngeom, -1, dtype=int)
        for g, side in self._geom_side.items():
            self._side_of[g] = self._sides.index(side)
        self._roots = np.array([self._root[s] for s in self._sides])
        self._rsc = np.array([self._r_sidecut[s] for s in self._sides], dtype=float)
        self._geom_body = np.array(m.geom_bodyid)
        self._body_root = np.array(m.body_rootid)
        self.dissipated = 0.   # 그립이 옆미끄럼으로 소산한 에너지 누적(erg). reset_metrics와 무관하게 계속 쌓인다.
        self.reset_metrics()

    def reset_metrics(self):
        self._acc = {s: dict(n=0, lat=0., tan=0., edge=0., util=0.) for s in self._sides}

    def step(self, physics):
        """numpy 벡터화 버전(RESEARCH_NOTES 45번). 파이썬 접촉 루프(step_loop)가 정책 스텝 시간의
        ~59%를 먹어서 바꿨다. 수직력은 efc_force[efc_address](elliptic cone의 첫 행 = 법선),
        접촉점 속도는 cvel(트리 루트 질량중심 기준)에서 계산, 힘은 몸통에 합력/합모멘트로 한 번에
        건다(APPLY_TO='thorax'와 같은 결과). APPLY_TO='segment'면 루프 버전을 쓴다."""
        if APPLY_TO != 'thorax':
            return self.step_loop(physics)
        m, d = physics.model.ptr, physics.data.ptr
        d.qfrc_applied[:] = 0.
        ncon = d.ncon
        if ncon == 0:
            return
        con = d.contact
        g = con.geom[:ncon]
        s1, s2 = self._side_of[g[:, 0]], self._side_of[g[:, 1]]
        ground1 = self._geom_body[g[:, 0]] == 0
        ground2 = self._geom_body[g[:, 1]] == 0
        use1 = (s1 >= 0) & ground2
        use2 = (s2 >= 0) & ground1 & ~use1
        idx = np.nonzero(use1 | use2)[0]
        if idx.size == 0:
            return
        geom = np.where(use1[idx], g[idx, 0], g[idx, 1])
        side = self._side_of[geom]
        sign = np.where(use1[idx], -1., 1.)[:, None]
        n = sign * con.frame[idx, :3]                      # 지면 -> 스키 방향 법선.
        pos = con.pos[idx]
        # 비활성 접촉(여유 거리 밖)은 efc_address = -1. 그대로 인덱싱하면 efc_force의 마지막 값을
        # 읽어 엉뚱한 힘이 걸리고 4스텝 만에 발산했다(mj_contactForce는 이때 0을 돌려줌).
        addr = con.efc_address[idx]
        N = np.where(addr >= 0, d.efc_force[np.maximum(addr, 0)], 0.)
        # efc_force에 NaN이 든 행이 가끔 있다(루프 버전의 mj_contactForce는 유한값을 냄). 0으로.
        N = np.nan_to_num(N, nan=0., posinf=0., neginf=0.)
        # efc_force의 일부 행이 NaN인 순간이 있었다(판 접촉 행, 시뮬레이션은 정상 진행). 그대로 쓰면
        # 몸통 합력 전체가 NaN이 돼 4스텝 만에 발산해서 NaN/inf는 0으로 둔다(45번).
        N = np.nan_to_num(N, nan=0., posinf=0., neginf=0.)
        N = np.minimum(np.maximum(N, 0.), self._n_cap)
        body = self._geom_body[geom]
        ey = d.xmat[body].reshape(-1, 3, 3)[:, :, 1]
        eyn = np.sum(ey * n, axis=1)
        phi = np.arcsin(np.minimum(np.abs(eyn), 1.))
        ex = d.xmat[self._roots[side]].reshape(-1, 3, 3)[:, :, 0]
        ex = ex - np.sum(ex * n, axis=1)[:, None] * n
        ex /= np.maximum(np.linalg.norm(ex, axis=1), 1e-9)[:, None]
        if ARC_SOURCE == 'shape':
            tangent = d.xmat[body].reshape(-1, 3, 3)[:, :, 0]
            tangent = tangent - np.sum(tangent * n, axis=1)[:, None] * n
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1), 1e-9)[:, None]
        else:
            rsc = self._rsc[side]
            kappa = np.where(rsc > 0, -np.sign(eyn) / (np.maximum(rsc, 1e-9) * np.cos(phi)), 0.)
            theta = kappa * np.sum((pos - d.xpos[self._roots[side]]) * ex, axis=1)
            tangent = np.cos(theta)[:, None] * ex + np.sin(theta)[:, None] * np.cross(n, ex)
        t = np.cross(n, tangent)                           # 호의 왼쪽 법선.
        cv = d.cvel[body]
        v = cv[:, 3:] + np.cross(cv[:, :3], pos - d.subtree_com[self._body_root[body]])
        v_t = v - np.sum(v * n, axis=1)[:, None] * n
        v_lat = np.sum(v_t * t, axis=1)
        snow = self._snow_at(float(d.xpos[self._target][0]))   # 설질 구간은 수십 cm라 몸 위치로.
        if GRIP_MODEL == 'platform':
            # 선반 각 phi + 마찰각이 90°를 넘으면(판이 거의 옆으로 누움) tan이 음수가 돼 그립이 거꾸로
            # 걸렸다(몇 스텝 만에 넘어짐). 88°에서 자른다.
            ang = np.minimum(phi + np.arctan(FLAT_GRIP_FRACTION * snow.edge_grip), np.deg2rad(88.))
            mu = np.minimum(np.tan(ang), snow.edge_grip)
        else:
            ramp = np.clip((phi - PHI0) / (PHI1 - PHI0), 0., 1.)
            mu = snow.edge_grip * (FLAT_GRIP_FRACTION + (1. - FLAT_GRIP_FRACTION) * ramp)
        th = np.tanh(v_lat / V0)
        self.dissipated += float(np.sum(mu * N * th * v_lat)) * m.opt.timestep
        if self._enabled:
            f = -(mu * N * th)[:, None] * t
            center = d.xipos[self._target]
            force = np.nan_to_num(f.sum(axis=0))
            torque = np.nan_to_num(np.cross(pos - center, f).sum(axis=0))
            mujoco.mj_applyFT(m, d, force, torque, center, self._target, d.qfrc_applied)
        for k, sname in enumerate(self._sides):
            sel = side == k
            if not sel.any():
                continue
            a = self._acc[sname]
            a['n'] += int(sel.sum())
            a['lat'] += float(np.abs(v_lat[sel]).sum())
            a['tan'] += float(np.linalg.norm(v_t[sel], axis=1).sum())
            a['edge'] += float(phi[sel].sum())
            a['util'] += float(np.where(mu[sel] > 0., np.abs(th[sel]), 1.).sum())

    def step_loop(self, physics):
        """원래 접촉별 파이썬 루프 버전(벡터화 전, 비교/APPLY_TO='segment'용)."""
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
            # 조각 body 좌표계(geom 좌표계는 캡슐 필렛이면 축이 돌아가 있음).
            ey = d.xmat[m.geom_bodyid[g]].reshape(3, 3)[:, 1]
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
            ramp = float(np.clip((phi - PHI0) / (PHI1 - PHI0), 0., 1.))
            mu = snow.edge_grip * (FLAT_GRIP_FRACTION + (1. - FLAT_GRIP_FRACTION) * ramp)
            if self._enabled and mu > 0. and N > 0.:
                f = -mu * N * np.tanh(v_lat / V0) * t
                target = body if APPLY_TO == 'segment' else self._target
                mujoco.mj_applyFT(m, d, f, np.zeros(3), c.pos, target, d.qfrc_applied)
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
