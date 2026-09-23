"""업무1: 슬라롬형 냄새 게이트 필드.

01_infinite_slope_terrain.md 4절 요구사항: 슬로프 진행 방향(+X)을 따라 좌우로
번갈아 배치된 냄새 게이트, 순차 점화(한 번에 하나만 활성 — 동시 다중 활성화 시
기울기 간섭이 생긴다는 codex 교차검증 지적 반영, RESEARCH_NOTES.md 7번 참고).

이 게이트 시퀀스가 업무5 학습/평가의 ground-truth 경로다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Gate:
    gate_id: int
    order_index: int
    center_xy: tuple[float, float]
    pass_radius: float
    required_side: str  # 'left' | 'right' — 게이트 중심이 폴라인 기준 어느 쪽인지.


class OdorField:
    """순차 점화되는 슬라롬형 냄새 게이트 필드 (정적 가우시안 플룸)."""

    def __init__(self,
                gate_spacing_cm: float,
                gate_amplitude_cm: float,
                arena_dim: tuple[float, float],
                pass_radius_cm: float = 1.5,
                plume_sigma_cm: float | None = None,
                alternation_mode: str = 'strict',
                seed: int = 0):
        """
        Args:
            gate_spacing_cm: 게이트 간 x 간격.
            gate_amplitude_cm: 게이트 좌우 진폭(y 오프셋 크기).
            arena_dim: (radius_x, radius_y) — 업무1 SlopedMoguls.dim과 동일 규약.
            pass_radius_cm: 게이트 통과 판정 반경.
            plume_sigma_cm: 가우시안 플룸 표준편차. None이면 gate_spacing의 절반.
            alternation_mode: 'strict'(완전 좌우 교대) 또는 'stochastic'(확률적 변주).
            seed: 재현성을 위한 시드.
        """
        self._gate_spacing = gate_spacing_cm
        self._gate_amplitude = gate_amplitude_cm
        self._arena_dim = arena_dim
        self._pass_radius = pass_radius_cm
        self._plume_sigma = plume_sigma_cm or (gate_spacing_cm / 2)
        self._alternation_mode = alternation_mode
        self._random_state = np.random.RandomState(seed)

        self._gates: list[Gate] = self._build_gate_sequence()
        self._active_idx: int = 0

    def _build_gate_sequence(self) -> list[Gate]:
        size_x, size_y = self._arena_dim
        # 슬로프 시작(-size_x)에서 약간 떨어진 지점부터 게이트 시작.
        xs = np.arange(-size_x + self._gate_spacing, size_x, self._gate_spacing)
        max_amp = min(self._gate_amplitude, size_y * 0.8)

        gates = []
        sign = 1
        for i, x in enumerate(xs):
            if self._alternation_mode == 'strict':
                y = sign * max_amp
                sign *= -1
            elif self._alternation_mode == 'stochastic':
                # 완전 교대에서 크게 벗어나지 않도록, 같은 쪽 연속을 낮은 확률로만 허용.
                if self._random_state.uniform() < 0.85:
                    sign *= -1
                y = sign * max_amp
            else:
                raise ValueError(
                    f"Unknown alternation_mode: {self._alternation_mode!r}")
            side = 'left' if y > 0 else 'right'
            gates.append(Gate(gate_id=i, order_index=i, center_xy=(float(x), float(y)),
                              pass_radius=self._pass_radius, required_side=side))
        return gates

    def get_gate_sequence(self) -> list[Gate]:
        return list(self._gates)

    @property
    def active_gate(self) -> Gate | None:
        if 0 <= self._active_idx < len(self._gates):
            return self._gates[self._active_idx]
        return None  # 모든 게이트 통과 완료.

    def advance_gate(self, gate_id: int) -> None:
        """지정한 게이트를 통과 처리하고 다음 게이트를 활성화한다."""
        if self.active_gate is not None and self.active_gate.gate_id == gate_id:
            self._active_idx += 1

    def try_advance(self, position_xy: tuple[float, float]) -> bool:
        """현재 위치가 활성 게이트의 pass_radius 안이면 자동으로 다음 게이트로 넘어간다.

        Returns:
            게이트를 통과(전진)시켰으면 True.
        """
        gate = self.active_gate
        if gate is None:
            return False
        dist = np.hypot(position_xy[0] - gate.center_xy[0],
                        position_xy[1] - gate.center_xy[1])
        if dist <= gate.pass_radius:
            self.advance_gate(gate.gate_id)
            return True
        return False

    def sample(self, position_xyz: tuple[float, float, float]
              ) -> tuple[float, tuple[float, float, float]]:
        """현재 활성 게이트 기준 냄새 농도와 국소 기울기를 반환한다.

        비활성(이미 통과했거나 아직 안 켜진) 게이트는 기울기 간섭을 피하기 위해
        전혀 고려하지 않는다 — 01/04 문서에서 codex 교차검증으로 확정한 설계.
        """
        gate = self.active_gate
        if gate is None:
            return 0.0, (0.0, 0.0, 0.0)
        dx = position_xyz[0] - gate.center_xy[0]
        dy = position_xyz[1] - gate.center_xy[1]
        r2 = dx * dx + dy * dy
        sigma2 = self._plume_sigma ** 2
        concentration = float(np.exp(-r2 / (2 * sigma2)))
        # 농도 기울기 = -dC/d(dx,dy) 방향, 즉 게이트를 향하는 방향.
        grad_x = -concentration * dx / sigma2
        grad_y = -concentration * dy / sigma2
        return concentration, (float(grad_x), float(grad_y), 0.0)

    def reset(self) -> None:
        self._active_idx = 0

    @property
    def all_passed(self) -> bool:
        return self._active_idx >= len(self._gates)
