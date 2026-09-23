"""업무4: LIF(leaky integrate-and-fire) 뉴런 모델 버전의 ConnectomeController.

RESEARCH_NOTES.md 18번: 매끄러운 tanh rate-model로는 지속구동 하에서 리듬이 전혀
안 생겼다(gain 1~15, decay 0.05~0.95 스윕에도 전부 고정점 수렴). CPG 논문이
실제로 쓴 것과 같은 계열인 LIF(불응기 있는 스파이킹) 모델로 같은 위상을
재구동해서, 뉴런 모델 선택 자체가 원인이었는지 확인한다.
"""
from __future__ import annotations

import numpy as np

from flyski_sim.connectome_controller import ConnectomeController


class LIFConnectomeController(ConnectomeController):
    """동일 위상(DNg100/DNb08 -> 인터뉴런 -> 다리운동뉴런)을 LIF 뉴런으로 구동."""

    def __init__(self,
                topology_mode: str = 'real',
                seed: int = 0,
                weight_scale: float = 20.0,
                v_thresh: float = 1.0,
                v_reset: float = 0.0,
                refractory_steps: int = 3,
                tau_mem_init: float = 5.0,
                tau_syn_init: float = 3.0):
        super().__init__(topology_mode=topology_mode, seed=seed,
                         weight_scale=weight_scale)
        n = self.n_neurons
        self._tau_mem = np.full(n, tau_mem_init)
        self._tau_syn = np.full(n, tau_syn_init)
        self._v_thresh = np.full(n, v_thresh)
        self._bias = np.zeros(n)
        self._v_reset = v_reset
        self._refractory_steps = refractory_steps
        self.reset()

    def set_role_based_params(self, v_thresh_by_role: dict[str, float],
                              tau_mem_by_role: dict[str, float]) -> None:
        """뉴런 역할군(afferent_command/intrinsic_inhibitory/intrinsic_excitatory/
        efferent)별로 다른 v_thresh/tau_mem을 준다.

        05_training_evaluation.md: 전역 스칼라 4개짜리 1차 실험에서는 real/
        shuffled/random 사이 유의미한 차이가 없었다(RESEARCH_NOTES 26번) — 완전
        뉴런별(392차원)은 탐색이 너무 비싸서, 역할군 단위(4группы)를 중간
        지점으로 시도한다. FlyGM의 '뉴런별 소수 파라미터'에 좀 더 가깝다.
        """
        for i, role in enumerate(self.node_role):
            if role in v_thresh_by_role:
                self._v_thresh[i] = v_thresh_by_role[role]
            if role in tau_mem_by_role:
                self._tau_mem[i] = tau_mem_by_role[role]

    def reset(self) -> None:
        n = self.n_neurons
        self._V = np.zeros(n)
        self._refrac = np.zeros(n, dtype=int)
        self._syn_trace = np.zeros(n)
        # 침묵 판정/발화율 보정용(RESEARCH_NOTES.md 31번): reset 이후 누적 스파이크 수와
        # 막전위의 문턱 대비 최대 비율.
        self.spike_count = np.zeros(n)
        self.max_v_ratio = np.zeros(n)

    def step(self, afferent_drive, dt_steps: int = 1) -> np.ndarray:
        drive = np.zeros(self.n_neurons)
        drive[self.afferent_idx] = afferent_drive
        for _ in range(dt_steps):
            synaptic_input = self._W @ self._syn_trace + self._bias + drive
            active = self._refrac <= 0
            dV = (-(self._V) + synaptic_input) / self._tau_mem
            self._V = np.where(active, self._V + dV, self._v_reset)
            self._refrac = np.maximum(self._refrac - 1, 0)
            spiked = active & (self._V >= self._v_thresh)
            self.max_v_ratio = np.maximum(self.max_v_ratio, self._V / self._v_thresh)
            self.spike_count += spiked
            self._V = np.where(spiked, self._v_reset, self._V)
            self._refrac = np.where(spiked, self._refractory_steps, self._refrac)
            self._syn_trace = (self._syn_trace * np.exp(-1.0 / self._tau_syn) +
                               spiked.astype(float))
        return self._syn_trace[self.efferent_idx]

    def get_learnable_params(self) -> np.ndarray:
        return np.concatenate([self._tau_mem, self._tau_syn, self._v_thresh,
                               self._bias])

    def set_learnable_params(self, params: np.ndarray) -> None:
        n = self.n_neurons
        params = np.asarray(params)
        self._tau_mem = params[:n].copy()
        self._tau_syn = params[n:2 * n].copy()
        self._v_thresh = params[2 * n:3 * n].copy()
        self._bias = params[3 * n:4 * n].copy()
