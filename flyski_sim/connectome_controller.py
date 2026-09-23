"""업무4: MaleCNS 다리 CPG 회로 기반 ConnectomeController.

flyski_sim/connectome/extract_leg_cpg_circuit.R로 추출한 실제 위상
(DNg100/DNb08 -> 인터뉴런 -> 다리 운동뉴런)을 **고정**하고, 뉴런별 소수 파라미터
(시상수 decay, 게인 gain, 바이어스 bias)만 학습 대상으로 삼는다 — FlyGM 논문
레시피(04_connectome_interface.md 참고), 단 이 프로젝트에서는 아직 학습 전
(랜덤/기본 초기값) 순전파 동역학만 구현·검증한다.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

_CONN_DIR = os.path.join(os.path.dirname(__file__), 'connectome')
# acetylcholine=흥분성, gaba/glutamate=억제성(곤충 CNS에서 흔한 관례).
# 관측된 NT가 없거나(unclear/NA) 그 외(octopamine 등)인 경우 흥분성으로 취급.
_NT_SIGN = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0}


def _nt_sign(nt) -> float:
    return _NT_SIGN.get(nt, 1.0)


class ConnectomeController:
    """DNg100/DNb08 -> 인터뉴런 -> 다리 운동뉴런 회로의 rate-model 컨트롤러.

    Args:
        topology_mode: 'real'(실제 위상) | 'shuffled'(타깃 치환, degree 근사 보존)
            | 'random'(동일 노드/엣지 수의 Erdos-Renyi) — 업무5 대조군 비교용.
        seed: shuffled/random 위상 생성 및 파라미터 초기화 재현성.
        dt_tau_init: 뉴런별 decay(시상수)의 초기값, (0,1) 범위.
        weight_scale: 시냅스 카운트를 rate-model 입력 스케일로 정규화하는 상수.
    """

    def __init__(self, topology_mode: str = 'real', seed: int = 0,
                dt_tau_init: float = 0.3, weight_scale: float = 100.0):
        self._rng = np.random.RandomState(seed)
        self._weight_scale = weight_scale
        self._load_data()
        self._build_topology(topology_mode)
        n = self.n_neurons
        self._decay = np.full(n, dt_tau_init)
        self._gain = np.ones(n)
        self._bias = np.zeros(n)
        self._h = np.zeros(n)

    def _load_data(self) -> None:
        cand = pd.read_csv(os.path.join(_CONN_DIR, '01_candidate_neurons.csv'))
        conn1 = pd.read_csv(os.path.join(_CONN_DIR, '02_internal_connectivity.csv'))
        motor = pd.read_csv(os.path.join(_CONN_DIR, '03_leg_motor_neurons.csv'))
        conn2 = pd.read_csv(
            os.path.join(_CONN_DIR, '04_interneuron_to_motor_connectivity.csv'))

        cand_ids = cand['bodyid'].tolist()
        motor_ids = motor['bodyid'].tolist()
        self.node_ids = cand_ids + motor_ids
        self.node_index = {bid: i for i, bid in enumerate(self.node_ids)}
        self.node_role = cand['role'].tolist() + ['efferent'] * len(motor_ids)
        self.node_type = cand['type'].tolist() + motor['type'].tolist()

        nt_map = dict(zip(cand['bodyid'], cand['predictedNt']))

        edges = []
        for _, row in pd.concat([conn1, conn2], ignore_index=True).iterrows():
            f, t = row['bodyid'], row['partner']
            if f in self.node_index and t in self.node_index:
                sign = _nt_sign(nt_map.get(f, 'acetylcholine'))
                edges.append((f, t, sign * row['weight']))
        self._raw_edges = edges

        self.n_neurons = len(self.node_ids)
        self.afferent_idx = [i for i, r in enumerate(self.node_role)
                             if r == 'afferent_command']
        self.efferent_idx = [i for i, r in enumerate(self.node_role)
                             if r == 'efferent']

    def _build_topology(self, topology_mode: str) -> None:
        n = self.n_neurons
        edge_list = [(self.node_index[f], self.node_index[t], w)
                    for f, t, w in self._raw_edges]

        if topology_mode == 'real':
            pass
        elif topology_mode == 'shuffled':
            # 각 엣지의 타깃(post-synaptic) 인덱스를 무작위 치환 — in-degree
            # 분포는 대략 보존되고(같은 타깃 목록을 재배치), 실제 위상 정보는 깨짐.
            targets = [t for _, t, _ in edge_list]
            self._rng.shuffle(targets)
            edge_list = [(f, t, w) for (f, _, w), t in zip(edge_list, targets)]
        elif topology_mode == 'random':
            n_edges = len(edge_list)
            weights = [w for _, _, w in edge_list]
            self._rng.shuffle(weights)
            froms = self._rng.randint(0, n, n_edges)
            tos = self._rng.randint(0, n, n_edges)
            edge_list = list(zip(froms, tos, weights))
        else:
            raise ValueError(f"Unknown topology_mode: {topology_mode!r}")

        W = np.zeros((n, n))
        for f, t, w in edge_list:
            W[t, f] += w / self._weight_scale  # 행=post(to), 열=pre(from).
        self._W = W
        self.topology_mode = topology_mode
        self.n_edges = len(edge_list)

    def reset(self) -> None:
        self._h = np.zeros(self.n_neurons)

    def step(self, afferent_drive, dt_steps: int = 1) -> np.ndarray:
        """afferent_command 노드에 구동을 주고 dt_steps만큼 내부 상태를 갱신한다.

        Args:
            afferent_drive: 스칼라 또는 afferent 노드 수 길이의 벡터.
            dt_steps: 내부 적분 스텝 수(물리 스텝과 신경 스텝 주파수 차이를
                여기서 흡수 — 04_connectome_interface.md 5절 참고).

        Returns:
            efferent(다리 운동뉴런) 노드들의 활성 벡터, shape (n_efferent,).
        """
        drive = np.zeros(self.n_neurons)
        drive[self.afferent_idx] = afferent_drive
        decay = np.clip(self._decay, 0.01, 0.99)
        for _ in range(dt_steps):
            pre_act = np.tanh(self._h)
            total_input = (self._W @ pre_act) * self._gain + self._bias + drive
            self._h = (1 - decay) * self._h + decay * total_input
        return np.tanh(self._h[self.efferent_idx])

    def get_learnable_params(self) -> np.ndarray:
        return np.concatenate([self._decay, self._gain, self._bias])

    def set_learnable_params(self, params: np.ndarray) -> None:
        n = self.n_neurons
        params = np.asarray(params)
        self._decay = params[:n].copy()
        self._gain = params[n:2 * n].copy()
        self._bias = params[2 * n:3 * n].copy()
