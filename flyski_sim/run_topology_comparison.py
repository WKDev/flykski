"""업무5: real/shuffled/random 위상 다중 시드 비교 (백그라운드 실행용).

RESEARCH_NOTES.md 24번: n=1 예비 비교(real=2.14>shuffled=1.29>random=0.64)가
가설과 일치하는 방향으로 나왔지만, 05번 문서가 요구하는 "5회 이상 독립 실행 +
통계 검정"에는 못 미쳤다. 이 스크립트가 그 격차를 메운다.

시간이 오래 걸릴 수 있어(조건당 여러 시드 x ES 여러 세대) 백그라운드로 돌리고,
매 (topology, seed) 완료 시마다 결과를 CSV에 즉시 append한다 — 중간에 확인하거나
중단해도 그때까지의 결과는 남는다.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np

from flyski_sim.train_connectome_params import run_es

OUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'topology_comparison_results.csv')

N_SEEDS = 5
TOPOLOGIES = ('real', 'shuffled', 'random')
# 시간 예산을 위해 1차 비교(n=1)보다 살짝 줄인 예산 — 그래도 추세 비교엔 충분.
ES_KWARGS = dict(n_generations=6, population=4, sigma=0.3, sigma_decay=0.85,
                 verbose=False)


def main():
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['topology_mode', 'seed', 'best_score',
                             'weight_scale', 'v_thresh', 'tau_mem', 'gain',
                             'elapsed_sec'])
            f.flush()

        for topology in TOPOLOGIES:
            for seed in range(N_SEEDS):
                t0 = time.time()
                result = run_es(seed=seed, topology_mode=topology,
                                topology_seed=seed, **ES_KWARGS)
                elapsed = time.time() - t0
                ws, vt, tm, g = result['best_params_exp']
                writer.writerow([topology, seed, result['best_score'],
                                 ws, vt, tm, g, round(elapsed, 1)])
                f.flush()
                print(f"[{topology} seed={seed}] best_score="
                     f"{result['best_score']:.4f} ({elapsed:.0f}s)", flush=True)

    print("DONE - all topology/seed combinations complete.")


if __name__ == '__main__':
    main()
