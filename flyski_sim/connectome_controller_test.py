"""업무4 ConnectomeController 스모크테스트.

04_connectome_interface.md 수용 기준:
- topology_mode="real"과 "shuffled"/"random"이 동일 뉴런 수/연결 수를 가지면서
  위상만 다름.
- get/set_learnable_params 왕복 검증.
- (핵심 과학적 질문) DNg100/DNb08에 지속적인 '걷기 명령'을 주면 다리 운동뉴런
  층에서 리듬(진동)이 실제로 생기는가? — CPG 논문의 개방루프 리듬생성 주장을
  이 추출된 위상이 재현하는지 확인하는 첫 시도.
"""
from __future__ import annotations

import numpy as np

from flyski_sim.connectome_controller import ConnectomeController


def check_topology_consistency() -> dict:
    real = ConnectomeController(topology_mode='real', seed=0)
    shuffled = ConnectomeController(topology_mode='shuffled', seed=0)
    random_ = ConnectomeController(topology_mode='random', seed=0)
    return {
        'n_neurons': real.n_neurons,
        'n_edges_real': real.n_edges,
        'n_edges_shuffled': shuffled.n_edges,
        'n_edges_random': random_.n_edges,
        'same_node_edge_counts': (real.n_neurons == shuffled.n_neurons ==
                                  random_.n_neurons and
                                  real.n_edges == shuffled.n_edges ==
                                  random_.n_edges),
        'topology_actually_differs': not np.array_equal(real._W, shuffled._W),
        'n_afferent': len(real.afferent_idx),
        'n_efferent': len(real.efferent_idx),
    }


def check_param_roundtrip() -> dict:
    ctrl = ConnectomeController(topology_mode='real', seed=0)
    original = ctrl.get_learnable_params().copy()
    new_params = original + np.random.RandomState(1).normal(
        0, 0.01, size=original.shape)
    ctrl.set_learnable_params(new_params)
    recovered = ctrl.get_learnable_params()
    return {
        'param_dim': original.shape[0],
        'roundtrip_matches': bool(np.allclose(recovered, new_params)),
    }


def _count_zero_crossings(signal: np.ndarray) -> int:
    centered = signal - signal.mean()
    return int(np.sum(np.diff(np.sign(centered)) != 0))


def check_rhythm_under_sustained_drive(topology_mode: str = 'real',
                                       n_steps: int = 500,
                                       drive: float = 1.5,
                                       seed: int = 0) -> dict:
    """DNg100/DNb08에 지속적 구동을 주고 다리 운동뉴런 층의 평균 활성을 기록."""
    ctrl = ConnectomeController(topology_mode=topology_mode, seed=seed)
    ctrl.reset()
    trace = []
    for _ in range(n_steps):
        out = ctrl.step(afferent_drive=drive, dt_steps=1)
        trace.append(out.mean())
    trace = np.array(trace)
    zero_crossings = _count_zero_crossings(trace)
    return {
        'topology_mode': topology_mode,
        'output_std': round(float(trace.std()), 5),
        'output_range': (round(float(trace.min()), 4), round(float(trace.max()), 4)),
        'zero_crossings': zero_crossings,
        'looks_rhythmic_not_flat': trace.std() > 1e-4,
        'saturated': bool(np.all(np.abs(trace[-50:]) > 0.99)),
    }


if __name__ == '__main__':
    print("=== 위상 일관성 검사 (real/shuffled/random 동일 노드·엣지 수) ===")
    print(f"  {check_topology_consistency()}")

    print("\n=== 학습 파라미터 왕복 검사 ===")
    print(f"  {check_param_roundtrip()}")

    print("\n=== 지속 구동 하 리듬(진동) 검사 ===")
    for mode in ('real', 'shuffled', 'random'):
        print(f"  [{mode}] {check_rhythm_under_sustained_drive(mode)}")
