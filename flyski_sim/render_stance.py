# 스키 스탠스/플레이트 부착 상태를 평지 근접 샷(측면/정면/위)으로 PNG 저장하는 렌더 스크립트
"""python -m flyski_sim.render_stance [출력 접두어]

평지에 착지시킨 뒤(zero action = 스탠스 유지) 측면/정면/위에서 찍는다. 판 중심에 T2가
오는지, 발끝이 판 위에 있는지 눈으로 확인하는 용도(RESEARCH_NOTES 34번).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from dm_control.mujoco.engine import MovableCamera
from PIL import Image

from flyski_sim.live_view import build_env

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'renders', 'stance')


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'stance'
    os.makedirs(OUT_DIR, exist_ok=True)
    env = build_env('alpine', True, 'all_mountain', 7, 1e9)
    env.reset()
    for _ in range(150):
        env.step(np.zeros(env.action_spec().shape))
    p = env.physics
    th = p.model.name2id('walker/thorax', 'body')
    lookat = p.data.xpos[th].copy()
    lookat[2] *= 0.5
    views = {'side': (90, -5), 'front': (180, -5), 'top': (90, -89), 'persp': (135, -25)}
    tiles = []
    for name, (az, el) in views.items():
        cam = MovableCamera(p, height=480, width=640)
        cam.set_pose(lookat, 0.9, az, el)
        img = cam.render()
        Image.fromarray(img).save(os.path.join(OUT_DIR, f'{prefix}_{name}.png'))
        tiles.append(img)
    grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    Image.fromarray(grid).save(os.path.join(OUT_DIR, f'{prefix}_grid.png'))
    print('saved', os.path.join(OUT_DIR, f'{prefix}_grid.png'))


if __name__ == '__main__':
    main()
