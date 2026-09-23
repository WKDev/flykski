# 초파리 다리로 스키 판 하나를 병진(x/y/z)/회전(롤/피치/요) 어디까지 움직일 수 있는지 IK로 스윕하는 도달 범위 검사
"""판 6자유도 도달 범위(RESEARCH_NOTES 39번).

왼쪽 판 하나만 스탠스에서 한 자유도씩 움직인 자세를 IK(ski_stance.solve_plate_pose)로 풀고,
T1/T2/T3 발끝 위치 오차와 T2 발(부츠) 방향 오차가 작으면 "도달 가능"으로 본다. 반대편 판은
스탠스 그대로. 다리 관절 가동범위 안에서만 푼다(least_squares bounds).

    python -m flyski_sim.plate_workspace
"""
from __future__ import annotations

import warnings

import numpy as np
from flybody.fruitfly import fruitfly

from flyski_sim.ski_stance import solve_plate_pose

TIP_OK_CM = 0.005
ORI_OK_DEG = 5.


def reach(walker, dof, value):
    yaw = {'left': 0., 'right': 0.}
    roll = {'left': 0., 'right': 0.}
    pitch = {'left': 0., 'right': 0.}
    shift = {'left': np.zeros(3), 'right': np.zeros(3)}
    if dof in ('x', 'y', 'z'):
        shift['left']['xyz'.index(dof)] = value
    else:
        {'yaw': yaw, 'roll': roll, 'pitch': pitch}[dof]['left'] = value
    _, err = solve_plate_pose(walker, yaw, roll, shift_cm=shift, pitch_deg=pitch)
    tip = max(v for k, v in err.items() if k.endswith('_left'))
    ori = np.degrees(err['T2_left_ori'])
    return tip, ori


def main():
    warnings.filterwarnings('ignore')
    w = fruitfly.FruitFly()
    sweeps = {'x': np.linspace(-0.1, 0.1, 9), 'y': np.linspace(-0.1, 0.1, 9), 'z': np.linspace(-0.1, 0.1, 9),
              'roll': np.linspace(-40, 40, 9), 'pitch': np.linspace(-30, 30, 7), 'yaw': np.linspace(-40, 40, 9)}
    for dof, vals in sweeps.items():
        ok = []
        for v in vals:
            tip, ori = reach(w, dof, float(v))
            if tip < TIP_OK_CM and ori < ORI_OK_DEG:
                ok.append(float(v))
            print(f'  {dof:5s} {v:+7.3f}: tip err {tip:.4f}cm, boot ori err {ori:5.1f}deg', flush=True)
        unit = 'cm' if dof in 'xyz' else 'deg'
        print(f'{dof}: reachable {min(ok) if ok else None}..{max(ok) if ok else None} {unit}', flush=True)


if __name__ == '__main__':
    main()
