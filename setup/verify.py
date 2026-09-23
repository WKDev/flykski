"""새 PC에서 설치가 제대로 됐는지 한 번에 확인한다.

    python setup/verify.py            # 빠른 검사(지형/스키/텔레메트리/항력/커넥톰 통합)
    python setup/verify.py --all      # + LIF 리듬 비교, rate-model 컨트롤러 검사

각 검사 모듈을 별도 프로세스로 실행하고 종료 코드만 모아서 보여준다(출력은 그대로
흘려보냄). 헤드리스 Linux라면 먼저 `export MUJOCO_GL=egl`(또는 osmesa).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUICK = ['smoke_test', 'ski_test', 'telemetry_test', 'drag_test',
         'connectome_to_flybody_test']
EXTRA = ['connectome_controller_lif_test', 'connectome_controller_test']


def main():
    mods = QUICK + (EXTRA if '--all' in sys.argv else [])
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''),
               PYTHONIOENCODING='utf-8')
    results = []
    for mod in mods:
        print(f'\n===== flyski_sim.{mod} =====', flush=True)
        t0 = time.time()
        code = subprocess.call([sys.executable, '-m', f'flyski_sim.{mod}'], cwd=ROOT, env=env)
        results.append((mod, code, time.time() - t0))
    print('\n===== 요약 =====')
    for mod, code, dt in results:
        print(f"  {'OK  ' if code == 0 else 'FAIL'} {mod} ({dt:.0f}s)")
    sys.exit(0 if all(c == 0 for _, c, _ in results) else 1)


if __name__ == '__main__':
    main()
