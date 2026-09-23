"""LIF ConnectomeController 리듬 생성 검증.

RESEARCH_NOTES.md 18번(rate-model은 리듬 없음) -> 19번(LIF로 교체하니 리듬 생김)의
재현 스크립트. real/shuffled/random 위상을 여러 시드로 비교한다.
"""
from __future__ import annotations

import numpy as np

from flyski_sim.connectome_controller_lif import LIFConnectomeController

# 수동 스윕으로 찾은, real 위상에서 뚜렷한 리듬이 나오는 파라미터.
_DEFAULT_KWARGS = dict(weight_scale=30., v_thresh=0.5, tau_mem_init=8.0)


def _zero_crossings(signal: np.ndarray) -> int:
    centered = signal - signal.mean()
    return int(np.sum(np.diff(np.sign(centered)) != 0))


def run_trace(topology_mode: str, seed: int, n_steps: int = 600,
             drive: float = 0.6, warmup: int = 100) -> np.ndarray:
    ctrl = LIFConnectomeController(topology_mode=topology_mode, seed=seed,
                                   **_DEFAULT_KWARGS)
    ctrl.reset()
    trace = np.array([ctrl.step(afferent_drive=drive, dt_steps=1).mean()
                      for _ in range(n_steps)])
    return trace[warmup:]


def compare_topologies(n_seeds: int = 5) -> dict:
    results = {}
    for mode in ('real', 'shuffled', 'random'):
        stds, zcs, means = [], [], []
        for seed in range(n_seeds):
            trace = run_trace(mode, seed)
            stds.append(trace.std())
            zcs.append(_zero_crossings(trace))
            means.append(trace.mean())
        results[mode] = {
            'std_mean': round(float(np.mean(stds)), 5),
            'std_std_across_seeds': round(float(np.std(stds)), 5),
            'zero_crossings_mean': round(float(np.mean(zcs)), 1),
            'mean_activity': round(float(np.mean(means)), 4),
        }
    results['real_and_shuffled_both_beat_random'] = (
        results['real']['std_mean'] > 5 * results['random']['std_mean'] and
        results['shuffled']['std_mean'] > 5 * results['random']['std_mean'])
    return results


if __name__ == '__main__':
    print("=== real/shuffled/random 리듬 강도 비교 (LIF, 5 시드) ===")
    for k, v in compare_topologies().items():
        print(f"  {k}: {v}")
