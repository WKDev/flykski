"""업무4 통합 테스트: LIF ConnectomeController -> motor_decoder -> flybody.

RESEARCH_NOTES.md 19번에서 찾은 리듬이, 실제 flybody 다리 관절을 주기적으로
움직이는지 확인한다(지금까지는 ConnectomeController가 flybody와 분리된 채로만
검증됐었음).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly

from flyski_sim.tasks import SlopeSmokeTask
from flyski_sim.connectome_controller_lif import LIFConnectomeController
from flyski_sim.motor_decoder import build_motor_neuron_dof_map, decode_to_flybody_action

import os
_CONN_DIR = os.path.join(os.path.dirname(__file__), 'connectome')


def _zero_crossings(signal: np.ndarray) -> int:
    centered = signal - signal.mean()
    return int(np.sum(np.diff(np.sign(centered)) != 0))


def run_integration(n_steps: int = 400, neural_substeps: int = 3) -> dict:
    # --- flybody 환경 (평지, 스키 없이 다리 움직임만 우선 확인) ---
    arena = floors.Floor()
    task = SlopeSmokeTask(walker=fruitfly.FruitFly, arena=arena, time_limit=4.0,
                          joint_filter=0., claw_friction=1.0)
    env = composer.Environment(time_limit=4.0, task=task,
                               random_state=np.random.RandomState(2),
                               strip_singleton_obs_buffer_dim=True)
    env.reset()
    actuator_names = [env.physics.model.id2name(i, 'actuator')
                      for i in range(env.physics.model.nu)]

    # --- LIF ConnectomeController ---
    ctrl = LIFConnectomeController(topology_mode='real', seed=0,
                                   weight_scale=30., v_thresh=0.5, tau_mem_init=8.0)
    ctrl.reset()

    motor_meta = pd.read_csv(os.path.join(_CONN_DIR, '03_leg_motor_neurons.csv'))
    dof_map = build_motor_neuron_dof_map(motor_meta)
    motor_body_ids = motor_meta['bodyid'].tolist()

    # --- 관절각 기록용: femur_T1_left 관절이 실제로 주기적으로 움직이는지. ---
    jid = env.physics.model.name2id('walker/femur_T1_left', 'joint')
    qpos_adr = env.physics.model.jnt_qposadr[jid]

    joint_trace = []
    action_nonzero_dims = 0
    stable = True
    steps_done = 0
    for steps_done in range(1, n_steps + 1):
        efferent = ctrl.step(afferent_drive=0.6, dt_steps=neural_substeps)
        action = decode_to_flybody_action(efferent, motor_body_ids, dof_map,
                                          actuator_names, gain=1.0)
        action_nonzero_dims = max(action_nonzero_dims, int(np.sum(action != 0)))
        env.step(action.astype(np.float32))
        joint_trace.append(float(env.physics.data.qpos[qpos_adr]))
        if not np.all(np.isfinite(env.physics.data.qpos)):
            stable = False
            break

    joint_trace = np.array(joint_trace)
    warm = joint_trace[50:] if len(joint_trace) > 50 else joint_trace

    return {
        'steps_completed': steps_done,
        'stable': stable,
        'action_nonzero_dims': action_nonzero_dims,
        'action_dim_total': len(actuator_names),
        'femur_T1_left_qpos_std': round(float(warm.std()), 5),
        'femur_T1_left_qpos_range': (round(float(warm.min()), 4),
                                     round(float(warm.max()), 4)),
        'femur_T1_left_zero_crossings': _zero_crossings(warm),
    }


if __name__ == '__main__':
    print("=== ConnectomeController(LIF) -> flybody 통합 테스트 ===")
    print(f"  {run_integration()}")
