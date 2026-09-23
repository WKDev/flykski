"""업무5: 발화율 보정(weight_scale) + 침묵 페널티를 넣은 real/shuffled/random 비교.

RESEARCH_NOTES.md 29/31번: 27번 비교는 모든 위상이 같은 초기 weight_scale(40)에서
출발해서, 결과가 "초기값에서 침묵 고원을 탈출했나"에 지배됐다(같은 ws에서 real
발화율 0.023, random ~0; random은 ws가 [5,80]으로 클립돼 발화 영역에 갈 수도
없었음). 이번엔 위상별로 대응 운동뉴런 발화율을 같게 맞춰서 출발시키고, 침묵은
발화한 어떤 해보다 낮게 채점한다. ES 예산/적합도 함수는 27번과 동일.

주의: 27번 이후 motor_decoder가 좌우(side)를 구분하게 바뀌었으므로(28번), 27번
결과와의 차이는 보정+페널티만의 효과가 아니다.
"""
from __future__ import annotations

import csv
import os
import time

from flyski_sim.train_connectome_params_role import run_es_role

OUT_CSV = os.path.join(os.path.dirname(__file__), '..',
                       'topology_comparison_calibrated_results.csv')

N_SEEDS = 5
TOPOLOGIES = ('real', 'shuffled', 'random')
ES_KWARGS = dict(n_generations=6, population=6, sigma=0.3, sigma_decay=0.85,
                 calibrate=True, target_rate=0.02, silence_penalty=True,
                 verbose=False)


def main():
    write_header = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['topology_mode', 'seed', 'best_score', 'best_silent',
                             'silent_eval_frac', 'base_weight_scale', 'calib_rate',
                             'elapsed_sec'])
            f.flush()
        for topology in TOPOLOGIES:
            for seed in range(N_SEEDS):
                t0 = time.time()
                r = run_es_role(seed=seed, topology_mode=topology,
                                topology_seed=seed, **ES_KWARGS)
                elapsed = time.time() - t0
                writer.writerow([topology, seed, r['best_score'], r['best_silent'],
                                 round(r['silent_eval_frac'], 3),
                                 round(r['base_weight_scale'], 4),
                                 round(r['calib_rate'], 4), round(elapsed, 1)])
                f.flush()
                print(f"[{topology} seed={seed}] best={r['best_score']:.4f} "
                      f"ws0={r['base_weight_scale']:.3f} rate0={r['calib_rate']:.4f} "
                      f"silent_frac={r['silent_eval_frac']:.2f} ({elapsed:.0f}s)",
                      flush=True)
    print("DONE - all topology/seed combinations complete.")


if __name__ == '__main__':
    main()
