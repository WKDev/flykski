"""업무5 2차: 역할군(role-group) 단위 파라미터로 확장한 학습 루프.

RESEARCH_NOTES.md 26번: 전역 스칼라 4개짜리 1차 실험에서는 real/shuffled/random
사이 유의미한 차이가 없었다. 완전 뉴런별(392차원) 파라미터는 이 세션의 ES로
탐색하기엔 너무 비싸므로, 뉴런 역할군 4개(afferent_command/
intrinsic_inhibitory/intrinsic_excitatory/efferent) 단위로 v_thresh/tau_mem을
다르게 주는 중간 지점을 시도한다 — FlyGM의 '뉴런별 소수 파라미터'에 조금 더
가깝다(완전히 같지는 않음, 역할군 내부는 여전히 동질적).

파라미터 벡터(전부 log-space, 10차원):
  [weight_scale, gain,
   v_thresh_afferent, v_thresh_inhib, v_thresh_excit, v_thresh_efferent,
   tau_afferent, tau_inhib, tau_excit, tau_efferent]
"""
from __future__ import annotations

import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.connectome_controller_lif import LIFConnectomeController
from flyski_sim.motor_decoder import build_motor_neuron_dof_map, decode_to_flybody_action
from flyski_sim.train_connectome_params import _corr, _MOTOR_META, _DOF_MAP, _MOTOR_BODY_IDS, _JOINT_NAMES

_ROLES = ('afferent_command', 'intrinsic_inhibitory', 'intrinsic_excitatory', 'efferent')
_INIT_V_THRESH = 0.5
_INIT_TAU = 10.


def _mapped_motor_idx(ctrl: LIFConnectomeController) -> np.ndarray:
    """flybody 액추에이터로 실제 디코딩되는(dof_map에 있는) 운동뉴런 인덱스."""
    return np.array([ctrl.node_index[b] for b in _DOF_MAP])


def open_loop_motor_rate(topology_mode: str, topology_seed: int, weight_scale: float,
                         n_steps: int = 450, warmup: int = 45,
                         afferent_drive: float = 0.6) -> float:
    """물리 없이 네트워크만 돌려서, 초기 역할군 파라미터에서 대응 운동뉴런의
    평균 발화율(스파이크/신경스텝)을 잰다."""
    ctrl = LIFConnectomeController(topology_mode=topology_mode, seed=topology_seed,
                                   weight_scale=weight_scale)
    ctrl.set_role_based_params(dict.fromkeys(_ROLES, _INIT_V_THRESH),
                               dict.fromkeys(_ROLES, _INIT_TAU))
    ctrl.reset()
    for _ in range(warmup):
        ctrl.step(afferent_drive, dt_steps=1)
    before = ctrl.spike_count.copy()
    for _ in range(n_steps - warmup):
        ctrl.step(afferent_drive, dt_steps=1)
    idx = _mapped_motor_idx(ctrl)
    return float((ctrl.spike_count[idx] - before[idx]).mean() / (n_steps - warmup))


def calibrate_weight_scale(topology_mode: str, topology_seed: int,
                           target_rate: float = 0.02,
                           bounds: tuple[float, float] = (0.1, 200.),
                           iters: int = 20) -> tuple[float, float]:
    """대응 운동뉴런 평균 발화율이 target_rate가 되는 weight_scale을 찾는다.

    RESEARCH_NOTES.md 29/31번: 예전엔 모든 위상에 같은 초기 weight_scale(40)을
    줬는데, 같은 값에서 real은 발화율 0.023, random은 ~0이었다(random은 ws<1
    이어야 비슷해짐). 위상 비교가 "초기값이 발화 문턱에 가까웠나"를 재지 않도록
    조건별로 같은 발화율에서 출발시킨다. weight_scale은 시냅스 가중치를 나누는
    값이라 발화율은 ws에 대해 (근사적으로) 단조 감소 → log 공간 이분탐색.

    Returns: (weight_scale, 실제 도달한 발화율).
    """
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    if open_loop_motor_rate(topology_mode, topology_seed, bounds[0]) < target_rate:
        ws = bounds[0]
        return ws, open_loop_motor_rate(topology_mode, topology_seed, ws)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if open_loop_motor_rate(topology_mode, topology_seed, np.exp(mid)) > target_rate:
            lo = mid
        else:
            hi = mid
    # 발화율이 ws에 대해 계단식으로 튈 수 있어서(random 위상에서 관측), 중점 대신
    # 구간 양끝 중 목표에 더 가까운 쪽을 쓴다.
    cands = [(abs(r - target_rate), float(np.exp(x)), r) for x in (lo, hi)
             for r in [open_loop_motor_rate(topology_mode, topology_seed, np.exp(x))]]
    _, ws, rate = min(cands)
    return ws, rate


def evaluate_role_params(log_params: np.ndarray, n_steps: int = 150, warmup: int = 40,
                         seed: int = 2, topology_mode: str = 'real',
                         topology_seed: int = 0,
                         base_weight_scale: float | None = None,
                         silence_penalty: bool = False,
                         return_info: bool = False):
    """
    Args:
        base_weight_scale: 주면 log_params[0]을 절대값이 아니라 이 보정값 대비
            log 배율로 해석한다(±10배로 클립). None이면 예전 방식(절대값, [5,80]).
        silence_penalty: True면 대응 운동뉴런이 한 번도 발화하지 않은 경우
            fitness를 [-1, -0.5] 구간(발화한 어떤 해보다도 낮음)으로 보내되, 막전위가
            문턱에 가까울수록 높게 줘서 침묵 고원에서도 기울기를 만든다. 예전엔
            침묵이 0.594(자세 보너스+정착 과도응답의 상관)로 발화 해와 섞였다.
    """
    vals = np.exp(log_params)
    if base_weight_scale is None:
        weight_scale = np.clip(vals[0], 5., 80.)
    else:
        weight_scale = base_weight_scale * np.exp(np.clip(log_params[0], -2.3, 2.3))
    gain = np.clip(vals[1], 0.1, 5.0)
    v_thresh_vals = np.clip(vals[2:6], 0.1, 3.0)
    tau_vals = np.clip(vals[6:10], 1., 20.)
    v_thresh_by_role = dict(zip(_ROLES, v_thresh_vals))
    tau_by_role = dict(zip(_ROLES, tau_vals))

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
                                   weight_scale=weight_scale)
    ctrl.set_role_based_params(v_thresh_by_role, tau_by_role)
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
            return (-10.0, {'silent': False, 'diverged': True}) if return_info else -10.0
        for n in _JOINT_NAMES:
            traces[n].append(float(env.physics.data.qpos[qpos_adrs[n]]))
        z_trace.append(float(env.physics.data.xpos[thorax_id][2]))

    warm = {n: np.array(v[warmup:]) for n, v in traces.items()}
    z_mean = float(np.mean(z_trace[warmup:]))
    tripod = 0.0
    for side in ('left', 'right'):
        t1 = warm[f'walker/femur_T1_{side}']
        t2 = warm[f'walker/femur_T2_{side}']
        t3 = warm[f'walker/femur_T3_{side}']
        tripod += _corr(t1, t3) - _corr(t1, t2)
    contra = -_corr(warm['walker/femur_T1_left'], warm['walker/femur_T1_right'])
    activity_bonus = min(np.mean([w.std() for w in warm.values()]) * 10, 1.0)
    posture_bonus = np.clip(z_mean / 0.13, 0., 1.0)
    fitness = float(tripod + 0.5 * contra + 0.3 * activity_bonus + 0.2 * posture_bonus)

    idx = _mapped_motor_idx(ctrl)
    silent = bool(ctrl.spike_count[idx].sum() == 0)
    if silence_penalty and silent:
        fitness = -1.0 + 0.5 * float(np.clip(ctrl.max_v_ratio[idx].max(), 0., 1.))
    if return_info:
        return fitness, {'silent': silent, 'diverged': False,
                         'weight_scale': float(weight_scale),
                         'motor_rate': float(ctrl.spike_count[idx].mean() /
                                             (n_steps * 3))}
    return fitness


def run_es_role(n_generations: int = 6, population: int = 6, sigma: float = 0.3,
                seed: int = 0, sigma_decay: float = 0.85,
                topology_mode: str = 'real', topology_seed: int = 0,
                calibrate: bool = False, target_rate: float = 0.02,
                silence_penalty: bool = False,
                verbose: bool = True) -> dict:
    """calibrate=True면 위상별로 weight_scale을 발화율 target_rate에 맞춰 보정하고
    ES는 그 대비 배율을 탐색한다. silence_penalty는 evaluate_role_params 참고."""
    rng = np.random.RandomState(seed)
    base_ws, calib_rate = None, None
    if calibrate:
        base_ws, calib_rate = calibrate_weight_scale(topology_mode, topology_seed,
                                                     target_rate=target_rate)
        ws0 = 1.  # log 배율 0 = 보정값 그대로.
    else:
        ws0 = 40.
    # 초기값: weight_scale(또는 배율), gain=1, 전 역할군 v_thresh=0.5, tau=10 (log space).
    init = np.log([ws0, 1., *[_INIT_V_THRESH] * 4, *[_INIT_TAU] * 4])

    def ev(p):
        return evaluate_role_params(p, topology_mode=topology_mode,
                                    topology_seed=topology_seed,
                                    base_weight_scale=base_ws,
                                    silence_penalty=silence_penalty,
                                    return_info=True)

    best_params = init.copy()
    best_score, best_info = ev(best_params)
    cur_sigma = sigma
    history = [best_score]
    n_silent, n_eval = int(best_info['silent']), 1
    for gen in range(n_generations):
        candidates = [best_params + rng.normal(0, cur_sigma, size=10)
                     for _ in range(population)]
        results = [ev(c) for c in candidates]
        scores = [r[0] for r in results]
        n_silent += sum(r[1]['silent'] for r in results)
        n_eval += len(results)
        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score, best_info = scores[idx], results[idx][1]
            best_params = candidates[idx].copy()
        cur_sigma *= sigma_decay
        history.append(best_score)
        if verbose:
            print(f"  [{topology_mode}] gen {gen}: best={best_score:.4f} "
                 f"scores={[round(s, 2) for s in scores]}")
    return {'topology_mode': topology_mode, 'best_score': best_score,
           'best_params_exp': np.exp(best_params).tolist(), 'history': history,
           'base_weight_scale': base_ws, 'calib_rate': calib_rate,
           'best_silent': best_info['silent'],
           'silent_eval_frac': n_silent / n_eval}


if __name__ == '__main__':
    print("=== 역할군 파라미터(10차원) ES, real 위상 ===")
    result = run_es_role(n_generations=6, population=6, seed=0, topology_mode='real')
    print(f"best_score={result['best_score']:.4f}")
    print(f"params [ws, gain, vth_aff, vth_inh, vth_exc, vth_eff, "
         f"tau_aff, tau_inh, tau_exc, tau_eff] = "
         f"{[round(x,3) for x in result['best_params_exp']]}")
