"""업무5: 역할군(role-group) 파라미터 버전의 real/shuffled/random 다중 시드 비교.

RESEARCH_NOTES.md 26번: 전역 스칼라 4개짜리 비교에서는 유의미한 차이가 없었다.
"파라미터 자유도가 부족했을 수 있다"는 가설을 검증하기 위해, 역할군 단위
10차원 파라미터로 같은 비교를 반복한다.
"""
from __future__ import annotations

import csv
import os
import time

from flyski_sim.train_connectome_params_role import run_es_role

OUT_CSV = os.path.join(os.path.dirname(__file__), '..',
                       'topology_comparison_role_results.csv')

N_SEEDS = 5
TOPOLOGIES = ('real', 'shuffled', 'random')
ES_KWARGS = dict(n_generations=6, population=6, sigma=0.3, sigma_decay=0.85,
                 verbose=False)


def main():
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['topology_mode', 'seed', 'best_score', 'elapsed_sec'])
            f.flush()
        for topology in TOPOLOGIES:
            for seed in range(N_SEEDS):
                t0 = time.time()
                result = run_es_role(seed=seed, topology_mode=topology,
                                     topology_seed=seed, **ES_KWARGS)
                elapsed = time.time() - t0
                writer.writerow([topology, seed, result['best_score'],
                                 round(elapsed, 1)])
                f.flush()
                print(f"[{topology} seed={seed}] best_score="
                     f"{result['best_score']:.4f} ({elapsed:.0f}s)", flush=True)
    print("DONE - all topology/seed combinations complete.")


if __name__ == '__main__':
    main()
