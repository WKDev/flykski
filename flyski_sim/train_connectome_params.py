"""업무5 착수: LIF ConnectomeController + motor_decoder 파라미터의 첫 최적화 루프.

RESEARCH_NOTES.md 18/19/20/21번 요약: 위상은 고정, tanh rate-model은 리듬이
안 남 -> LIF로 바꾸니 리듬은 생김 -> 다리는 움직이지만 협응된 삼각보행은 아님
-> 랜덤 12개 파라미터 탐색으로도 안 풀림. 이제 진짜 최적화(간단한 (mu,lambda)
진화전략)를 시도한다.

**범위를 의도적으로 좁혔다**: 05_training_evaluation.md가 그리는 전체 그림
(냄새 게이트 기반 세션/에피소드, train/eval seed 분리, real/shuffled/random
3조건 비교)은 훨씬 크다. 지금은 그 중 "위상 고정 + 소수 파라미터 학습"이라는
핵심 아이디어가 조종 가능한 파라미터 수준(4개 전역 스칼라: weight_scale,
v_thresh, tau_mem, decoder gain)에서 성립하는지만 빠르게 검증한다 — 뉴런별
개별 파라미터(1568차원) 최적화나 냄새 게이트 통합은 다음 단계.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.connectome_controller_lif import LIFConnectomeController
from flyski_sim.motor_decoder import build_motor_neuron_dof_map, decode_to_flybody_action

_CONN_DIR = os.path.join(os.path.dirname(__file__), 'connectome')
_MOTOR_META = pd.read_csv(os.path.join(_CONN_DIR, '03_leg_motor_neurons.csv'))
_DOF_MAP = build_motor_neuron_dof_map(_MOTOR_META)
_MOTOR_BODY_IDS = _MOTOR_META['bodyid'].tolist()

_JOINT_NAMES = ['walker/femur_T1_left', 'walker/femur_T2_left', 'walker/femur_T3_left',
               'walker/femur_T1_right', 'walker/femur_T2_right', 'walker/femur_T3_right']


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / denom)


def evaluate_params(log_params: np.ndarray, n_steps: int = 180, warmup: int = 40,
                    seed: int = 2, topology_mode: str = 'real',
                    topology_seed: int = 0) -> float:
    """log_params = [log(weight_scale), log(v_thresh), log(tau_mem), log(gain)].

    Args:
        topology_mode: 'real' | 'shuffled' | 'random' — 업무5 대조군 비교용.
        topology_seed: shuffled/random 위상 생성 시드(같은 시드면 같은 대조군
            그래프를 매 평가마다 재사용 — 안 그러면 매 스텝 다른 그래프가
            나와서 파라미터 최적화 자체가 의미 없어진다).

    Returns: 적합도(fitness) 점수. 높을수록 좋음. 불안정하면 큰 음수.
    """
    weight_scale, v_thresh, tau_mem, gain = np.exp(log_params)
    weight_scale = np.clip(weight_scale, 5., 80.)
    v_thresh = np.clip(v_thresh, 0.1, 3.0)
    tau_mem = np.clip(tau_mem, 1., 20.)
    gain = np.clip(gain, 0.1, 5.0)

    arena = floors.Floor()
    task = SlopeSmokeTask(walker=fruitfly.FruitFly, arena=arena, time_limit=6.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=6.0, task=task,
                               random_state=np.random.RandomState(seed),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    actuator_names = [env.physics.model.id2name(i, 'actuator')
                      for i in range(env.physics.model.nu)]
    ctrl = LIFConnectomeController(topology_mode=topology_mode, seed=topology_seed,
                                   weight_scale=weight_scale, v_thresh=v_thresh,
                                   tau_mem_init=tau_mem)
    ctrl.reset()

    qpos_adrs = {n: env.physics.model.jnt_qposadr[env.physics.model.name2id(n, 'joint')]
                for n in _JOINT_NAMES}
    thorax_id = env.physics.model.name2id('walker/thorax', 'body')

    traces = {n: [] for n in _JOINT_NAMES}
    z_trace = []
    for _ in range(n_steps):
        efferent = ctrl.step(afferent_drive=0.6, dt_steps=3)
        action = decode_to_flybody_action(efferent, _MOTOR_BODY_IDS, _DOF_MAP,
                                          actuator_names, gain=gain)
        env.step(action.astype(np.float32))
        if not np.all(np.isfinite(env.physics.data.qpos)):
            return -10.0
        for n in _JOINT_NAMES:
            traces[n].append(float(env.physics.data.qpos[qpos_adrs[n]]))
        z_trace.append(float(env.physics.data.xpos[thorax_id][2]))

    warm = {n: np.array(v[warmup:]) for n, v in traces.items()}
    z_mean = float(np.mean(z_trace[warmup:]))

    # 삼각보행 점수: 동측 앞-뒷다리(T1-T3) 동위상 + 인접다리(T1-T2) 반위상
    # + 좌우 대측 다리(T1L-T1R) 반위상. 세 항목을 좌/우 양쪽에서 평균.
    tripod = 0.0
    for side_a, side_b in [('left', 'left'), ('right', 'right')]:
        t1 = warm[f'walker/femur_T1_{side_a}']
        t2 = warm[f'walker/femur_T2_{side_a}']
        t3 = warm[f'walker/femur_T3_{side_a}']
        tripod += _corr(t1, t3) - _corr(t1, t2)
    contra = -_corr(warm['walker/femur_T1_left'], warm['walker/femur_T1_right'])
    activity_bonus = min(np.mean([w.std() for w in warm.values()]) * 10, 1.0)
    posture_bonus = np.clip(z_mean / 0.13, 0., 1.0)  # 주저앉지 않을수록 가점.

    fitness = tripod + 0.5 * contra + 0.3 * activity_bonus + 0.2 * posture_bonus
    return float(fitness)


def run_es(n_generations: int = 10, population: int = 6, sigma: float = 0.3,
          seed: int = 0,
          init_log_params: np.ndarray | None = None,
          sigma_decay: float = 0.9,
          topology_mode: str = 'real',
          topology_seed: int = 0,
          verbose: bool = True) -> dict:
    """(1+lambda) 진화전략, elitism 포함.

    22번(RESEARCH_NOTES) 교훈 반영: truncation-selection(상위 절반 평균)은
    잡음 많은 물리 적합도 지형에서 최고점 근처에 못 머물고 표류했다. 이번엔
    항상 "지금까지 최고 개체"를 다음 세대의 평균으로 쓰고(elitism), 세대마다
    sigma를 줄여(sigma_decay) 점점 국소 탐색으로 수렴하게 한다.
    """
    rng = np.random.RandomState(seed)
    if init_log_params is None:
        init_log_params = np.log([33.7, 0.26, 9.9, 0.76])

    def ev(p):
        return evaluate_params(p, topology_mode=topology_mode,
                               topology_seed=topology_seed)

    best_params = init_log_params.copy()
    best_score = ev(best_params)
    history = [best_score]
    cur_sigma = sigma

    for gen in range(n_generations):
        candidates = [best_params + rng.normal(0, cur_sigma, size=4)
                     for _ in range(population)]
        scores = [ev(c) for c in candidates]
        gen_best_idx = int(np.argmax(scores))
        if scores[gen_best_idx] > best_score:
            best_score = scores[gen_best_idx]
            best_params = candidates[gen_best_idx].copy()
        cur_sigma *= sigma_decay
        history.append(best_score)
        if verbose:
            print(f"  [{topology_mode}] gen {gen}: best_so_far={best_score:.4f} "
                 f"sigma={cur_sigma:.3f} gen_scores={[round(s, 3) for s in scores]}")

    return {
        'topology_mode': topology_mode,
        'best_score': best_score,
        'best_params_exp': np.exp(best_params).tolist(),
        'history': history,
    }


if __name__ == '__main__':
    print("=== 초기 파라미터(랜덤탐색 최고점) 적합도 ===")
    init = np.log([33.7, 0.26, 9.9, 0.76])
    print(f"  fitness = {evaluate_params(init):.4f}")

    print("\n=== (mu,lambda) 진화전략 실행 ===")
    result = run_es(n_generations=8, population=5, seed=0)
    print(f"\nbest_score={result['best_score']:.4f}")
    print(f"best_params [weight_scale, v_thresh, tau_mem, gain] = "
         f"{[round(x, 3) for x in result['best_params_exp']]}")
