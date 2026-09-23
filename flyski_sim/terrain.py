"""업무1 1단계: 유한 경사+모글 슬로프 아레나 (flybody Hills 확장).

flykski/prompts/01_infinite_slope_terrain.md 의 1단계(유한 긴 코스) 요구사항을 구현한다.
flybody의 Hills/SineBumps 패턴(hfield 런타임 재생성 + mjr_uploadHField)을 그대로 재사용한다.
"""
from __future__ import annotations

import numpy as np
from dm_control.mujoco.wrapper import mjbindings

from flybody.tasks.arenas.hills import Hills

from flyski_sim.snow import SNOW_PRESETS

mjlib = mjbindings.mjlib


def _bump_shape(nrow: int,
                ncol: int,
                size_x: float,
                size_y: float,
                terrain_type: str,
                mogul_wavelength: float,
                random_state: np.random.RandomState) -> np.ndarray:
    """국소 요철의 '모양'만 [0, 1]로 정규화해서 반환한다 (높이는 아직 미적용).

    높이 스케일링을 분리해두면, `mogul_height`만 바꿔서 여러 후보를 시도하는
    `calibrate_mogul_height_for_max_slope`가 매번 다른 난수를 뽑지 않고
    동일한 요철 모양에 대해 단조적으로 탐색할 수 있다.
    """
    x_axis = np.linspace(-size_x, size_x, ncol)
    xv, _ = np.meshgrid(x_axis, np.linspace(-size_y, size_y, nrow))

    if terrain_type == 'alpine':
        corduroy_wavelength = 0.5 * mogul_wavelength if mogul_wavelength > 0 else 1.0
        shape = 0.5 * (1 + np.sin(2 * np.pi * xv / max(corduroy_wavelength, 1e-6)))
        return shape  # 이미 [0, 1].
    elif terrain_type == 'mogul':
        from scipy import ndimage
        low_res = max(4, int(round(2 * size_x / max(mogul_wavelength, 1e-6))))
        low_res_y = max(4, int(round(2 * size_y / max(mogul_wavelength, 1e-6))))
        noise = random_state.uniform(0, 1, (low_res_y, low_res))
        shape = ndimage.zoom(noise, (nrow / low_res_y, ncol / low_res), order=1)
        shape = ndimage.gaussian_filter(shape, sigma=max(1.0, ncol / (2 * low_res)))
        shape -= shape.min()
        if shape.max() > 0:
            shape /= shape.max()
        return shape
    else:
        raise ValueError(f"Unknown terrain_type: {terrain_type!r}")


def generate_slope_terrain(
    nrow: int,
    ncol: int,
    size: tuple[float, float],
    mean_slope_deg: float,
    terrain_type: str,
    mogul_wavelength: float,
    mogul_height: float,
    random_state: np.random.RandomState,
) -> np.ndarray:
    """+X 방향으로 내려가는 경사 램프 위에 모글/알파인 표면을 얹은 지형(cm)을 생성한다.

    Args:
        nrow, ncol: hfield 그리드 해상도 (행=Y, 열=X).
        size: (radius_x, radius_y), cm. MuJoCo hfield size 규약과 동일한 순서.
        mean_slope_deg: 종단(진행방향) 평균 경사, degrees.
        terrain_type: 'alpine'(정설) 또는 'mogul'(둔덕밭).
        mogul_wavelength: 둔덕 평균 간격, cm.
        mogul_height: 둔덕 높이(진폭), cm. 'alpine'에서는 실제 코듀로이 높이의
            1/100로 축소해서 쓴다(정설은 경사에 거의 영향 없어야 함).
        random_state: 재현성을 위한 난수 상태.

    Returns:
        terrain: (nrow, ncol) 배열, cm 단위 실제 높이 (flybody Hills 관례상
            hfield_elevation_z=1로 두고 이 배열을 그대로 hfield_data에 쓴다).
    """
    size_x, size_y = size
    x_axis = np.linspace(-size_x, size_x, ncol)
    xv, _ = np.meshgrid(x_axis, np.linspace(-size_y, size_y, nrow))

    # 종단 경사 램프: +X(하강 방향)로 갈수록 낮아짐. 최고점(시작)이 x=-size_x.
    ramp = (size_x - xv) * np.tan(np.deg2rad(mean_slope_deg))

    shape = _bump_shape(nrow, ncol, size_x, size_y, terrain_type,
                        mogul_wavelength, random_state)
    height = mogul_height * 0.01 if terrain_type == 'alpine' else mogul_height
    bumps = shape * height

    return ramp + bumps


def calibrate_mogul_height_for_max_slope(
    nrow: int,
    ncol: int,
    size: tuple[float, float],
    mean_slope_deg: float,
    max_slope_deg: float,
    mogul_wavelength: float,
    random_state: np.random.RandomState,
    tol_deg: float = 2.0,
    max_iters: int = 15,
    height_bounds: tuple[float, float] = (1e-3, 30.0),
) -> tuple[float, np.ndarray]:
    """목표 max_slope_deg에 근접하도록 mogul_height를 이분탐색으로 보정한다.

    01_infinite_slope_terrain.md 수용 기준: "mean_slope_deg=20, max_slope_deg=40
    설정 시 ... 최댓값이 40±3도 이내" 대응. 'mogul' 타입에서만 의미가 있다
    ('alpine'은 정의상 max≈mean이어야 하므로 별도 캘리브레이션을 하지 않음).

    동일 요철 '모양'을 고정해두고(난수를 한 번만 뽑음) 높이만 스케일링하므로
    탐색이 단조적으로 수렴한다.

    Returns:
        (calibrated_mogul_height, terrain) — 최종 지형 배열도 함께 반환해
        재생성 없이 바로 쓸 수 있게 한다.
    """
    size_x, size_y = size
    x_axis = np.linspace(-size_x, size_x, ncol)
    xv, _ = np.meshgrid(x_axis, np.linspace(-size_y, size_y, nrow))
    ramp = (size_x - xv) * np.tan(np.deg2rad(mean_slope_deg))
    shape = _bump_shape(nrow, ncol, size_x, size_y, 'mogul', mogul_wavelength,
                        random_state)

    lo, hi = height_bounds
    best_h, best_terrain = hi, ramp + shape * hi
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        terrain = ramp + shape * mid
        _, measured_max = slope_stats_deg(terrain, size)
        best_h, best_terrain = mid, terrain
        if abs(measured_max - max_slope_deg) <= tol_deg:
            break
        if measured_max < max_slope_deg:
            lo = mid
        else:
            hi = mid
    return best_h, best_terrain


def slope_stats_deg(terrain: np.ndarray,
                    size: tuple[float, float]) -> tuple[float, float]:
    """생성된 지형(cm)의 실측 평균/최대 경사(degrees)를 계산한다.

    01_infinite_slope_terrain.md의 "생성 후 통계 검증" 수용 기준에 대응.
    """
    nrow, ncol = terrain.shape
    size_x, size_y = size
    dx = (2 * size_x) / (ncol - 1)
    dy = (2 * size_y) / (nrow - 1)
    grad_y, grad_x = np.gradient(terrain, dy, dx)
    local_slope_deg = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))
    return float(local_slope_deg.mean()), float(local_slope_deg.max())


class SlopedMoguls(Hills):
    """경사 + 모글/알파인 슬로프 아레나 (업무1 1단계: 유한 긴 코스).

    flybody의 `Hills` hfield 인프라(런타임 재생성 + mjr_uploadHField 재업로드)를
    그대로 재사용하고, bowl 대신 종단 경사 램프 + 모글/알파인 표면을 얹는다.
    """

    def _build(self,
              name: str = 'sloped_moguls',
              dim: tuple[float, float] = (30., 5.),
              mean_slope_deg: float = 20.,
              max_slope_deg: float | None = None,
              terrain_type: str = 'mogul',
              mogul_wavelength: float = 3.0,
              mogul_height: float = 1.0,
              aesthetic: str = 'outdoor_natural',
              grid_density: float = 20,
              hfield_elevation_z: float | None = None,
              hfield_base_z: float = 0.5,
              snow_distribution: dict[str, float] | None = None,
              n_snow_zones: int = 5):
        """
        Args:
            dim: (radius_x, radius_y) cm. x가 진행(하강) 방향, 전체 슬로프 길이는
                2*radius_x cm, 폭은 2*radius_y cm.
            mean_slope_deg: 종단 평균 경사, degrees.
            max_slope_deg: 목표 국소 최대 경사, degrees. None이면 `mogul_height`를
                그대로 사용(수동 제어). 값을 주면 `mogul_height`를 무시하고
                이분탐색으로 자동 보정한다 — `terrain_type='mogul'`에서만 의미가
                있음('alpine'은 정의상 max≈mean).
            terrain_type: 'alpine' 또는 'mogul'.
            mogul_wavelength, mogul_height: 모글 파라미터, cm.
            grid_density: 단위 길이당 hfield 격자점 수.
            snow_distribution: {snow_type: weight} 딕셔너리. None이면 전체
                'packed_powder' 단일 설질(기존 동작과 동일, 하위호환).
            n_snow_zones: 슬로프를 x축으로 나눌 설질 구간 개수.
        """
        # MuJoCo hfield는 hfield_data를 [0,1]로 보고 size[2](elevation_z)를 곱해
        # 높이를 만든다. 충돌 경계박스(aabb)도 컴파일 시 elevation_z로 정해지므로,
        # 예전처럼 elevation_z=1에 cm 높이(최대 20+)를 그대로 쓰면 레이캐스트엔
        # 지형이 보이지만 z>1 위의 물체와는 충돌 판정이 안 된다(RESEARCH_NOTES.md
        # 30번). 가능한 최대 높이로 elevation_z를 잡고 데이터는 정규화해서 쓴다.
        if hfield_elevation_z is None:
            size_x = dim[0] if isinstance(dim, tuple) else dim
            ramp_max = 2 * size_x * np.tan(np.deg2rad(mean_slope_deg))
            if terrain_type == 'alpine':
                bump_max = mogul_height * 0.01
            elif max_slope_deg is not None:
                bump_max = 30.0  # calibrate_mogul_height_for_max_slope의 height_bounds 상한.
            else:
                bump_max = mogul_height
            hfield_elevation_z = ramp_max + bump_max + 1.0
        self._hfield_elevation_z = hfield_elevation_z

        # Hills._build를 그대로 호출해 hfield 애셋/텍스처/geom을 만든다.
        super()._build(name=name,
                      dim=dim,
                      aesthetic=aesthetic,
                      hfield_elevation_z=hfield_elevation_z,
                      hfield_base_z=hfield_base_z,
                      grid_density=grid_density,
                      elevation_z_range=(0., 0.))  # 사용 안 함(bowl 미사용)

        # 다리 6개 + adhesion claw가 모두 hfield와 접촉 가능해서 기본 njmax/nconmax로는
        # 제약조건 스택이 부족해 mj_stackAlloc 오류가 났다(fruitfly.xml이 이미
        # njmax/nconmax를 쓰므로 'memory' 속성과 섞어 쓸 수 없음 — 같은 계열로 확대).
        self._mjcf_root.size.njmax = 6000
        self._mjcf_root.size.nconmax = 3000

        self._mean_slope_deg = mean_slope_deg
        self._max_slope_deg = max_slope_deg
        self._terrain_type = terrain_type
        self._mogul_wavelength = mogul_wavelength
        self._mogul_height = mogul_height
        self._dim = dim if isinstance(dim, tuple) else (dim, dim)
        self._last_terrain = None  # 마지막 생성 지형 (cm), 통계 검증/높이 조회용.
        self._last_calibrated_mogul_height = None  # 자동 보정된 값(디버그용).
        self._snow_distribution = snow_distribution
        self._n_snow_zones = n_snow_zones
        self._zone_bounds_x: list[float] | None = None
        self._zone_types: list[str] | None = None

    def initialize_episode(self, physics, random_state):
        if self._regenerate:
            self._regenerate = False

            nrow = physics.bind(self._hfield).nrow
            ncol = physics.bind(self._hfield).ncol

            if self._max_slope_deg is not None and self._terrain_type == 'mogul':
                mogul_height, terrain = calibrate_mogul_height_for_max_slope(
                    nrow=nrow,
                    ncol=ncol,
                    size=self._dim,
                    mean_slope_deg=self._mean_slope_deg,
                    max_slope_deg=self._max_slope_deg,
                    mogul_wavelength=self._mogul_wavelength,
                    random_state=random_state,
                )
                self._last_calibrated_mogul_height = mogul_height
            else:
                terrain = generate_slope_terrain(
                    nrow=nrow,
                    ncol=ncol,
                    size=self._dim,
                    mean_slope_deg=self._mean_slope_deg,
                    terrain_type=self._terrain_type,
                    mogul_wavelength=self._mogul_wavelength,
                    mogul_height=self._mogul_height,
                    random_state=random_state,
                )
            self._last_terrain = terrain
            self._generate_snow_zones(random_state)

            if terrain.min() < 0 or terrain.max() > self._hfield_elevation_z:
                raise ValueError(
                    f'terrain range [{terrain.min():.2f}, {terrain.max():.2f}] cm '
                    f'exceeds hfield elevation_z={self._hfield_elevation_z:.2f}')
            start_idx = physics.bind(self._hfield).adr
            physics.model.hfield_data[start_idx:start_idx +
                                      nrow * ncol] = (terrain.ravel() /
                                                      self._hfield_elevation_z)

            # physics.contexts는 접근만 해도 GL 컨텍스트(숨은 GLFW 창)를 새로 만든다 —
            # 그러면 mujoco.viewer의 GLFW 창과 충돌해 화면이 검게 나왔다. 이미 있는
            # 컨텍스트에만 재업로드하면 충분하다(나중에 생기는 컨텍스트는 생성 시점의
            # hfield_data를 그대로 올린다).
            if getattr(physics, '_contexts', None):
                with physics.contexts.gl.make_current() as ctx:
                    ctx.call(mjlib.mjr_uploadHField, physics.model.ptr,
                            physics.contexts.mujoco.ptr,
                            physics.bind(self._hfield).element_id)

    def height_at(self, x: float, y: float) -> float:
        """마지막으로 생성된 지형에서 (x, y)에 가장 가까운 격자점의 높이(cm)를 반환.

        스폰 위치 보정 등에 쓰는 근사치 — 정밀한 버전은 업무3의
        TerrainQuery.get_height_and_normal()에서 보간으로 구현한다.
        """
        if self._last_terrain is None:
            return 0.
        nrow, ncol = self._last_terrain.shape
        size_x, size_y = self._dim
        idx_x = int(round((x + size_x) / (2 * size_x) * (ncol - 1)))
        idx_y = int(round((y + size_y) / (2 * size_y) * (nrow - 1)))
        idx_x = int(np.clip(idx_x, 0, ncol - 1))
        idx_y = int(np.clip(idx_y, 0, nrow - 1))
        return float(self._last_terrain[idx_y, idx_x])

    def _generate_snow_zones(self, random_state: np.random.RandomState) -> None:
        """슬로프를 x축으로 `n_snow_zones` 구간으로 나눠 설질을 배정한다."""
        size_x = self._dim[0]
        edges = np.linspace(-size_x, size_x, self._n_snow_zones + 1)
        self._zone_bounds_x = list(edges[1:])  # 각 구간의 오른쪽(하류) 경계.

        if self._snow_distribution is None:
            self._zone_types = ['packed_powder'] * self._n_snow_zones
            return

        types = list(self._snow_distribution.keys())
        weights = np.array(list(self._snow_distribution.values()), dtype=float)
        weights = weights / weights.sum()
        chosen = random_state.choice(types, size=self._n_snow_zones, p=weights)
        self._zone_types = list(chosen)

    def get_snow_type(self, x: float) -> str:
        """x 위치(cm)가 속한 설질 구간의 타입명을 반환한다."""
        if self._zone_types is None:
            return 'packed_powder'
        for bound, snow_type in zip(self._zone_bounds_x, self._zone_types):
            if x <= bound:
                return snow_type
        return self._zone_types[-1]

    def get_snow_params(self, x: float):
        """x 위치(cm)의 SnowParams(friction/solref/solimp)를 반환한다."""
        return SNOW_PRESETS[self.get_snow_type(x)]

    def apply_snow_friction_at(self, physics, x: float) -> str:
        """x 위치의 설질에 맞춰 지형 geom의 마찰/솔버 파라미터를 런타임으로 갱신한다.

        MuJoCo geom은 하나의 friction/solref/solimp만 가지므로, hfield 전체에
        여러 설질을 동시에 물리적으로 표현하지는 못한다(업무3/RESEARCH_NOTES 참고).
        대신 초파리가 현재 위치한 구간의 설질을 매 컨트롤 스텝마다 반영한다 —
        "정적 XML 값이 아니라 런타임 갱신"이라는 01 요구사항의 실용적 근사.

        Returns:
            적용된 snow_type 이름.
        """
        snow_type = self.get_snow_type(x)
        params = SNOW_PRESETS[snow_type]
        bound = physics.bind(self._terrain_geom)
        bound.friction = [params.friction, 0.005, 0.0001]
        bound.solref = list(params.solref)
        bound.solimp = list(params.solimp)
        return snow_type

    def normal_at(self, x: float, y: float) -> tuple[float, float, float]:
        """(x, y)에서의 지형 법선(단위벡터)을 유한차분으로 근사한다.

        업무3 텔레메트리(effective_edge_angle_deg)의 `terrain_normal_fn`으로 쓴다.
        """
        if self._last_terrain is None:
            return (0., 0., 1.)
        nrow, ncol = self._last_terrain.shape
        size_x, size_y = self._dim
        dx = (2 * size_x) / (ncol - 1)
        dy = (2 * size_y) / (nrow - 1)
        idx_x = int(np.clip(round((x + size_x) / (2 * size_x) * (ncol - 1)), 1, ncol - 2))
        idx_y = int(np.clip(round((y + size_y) / (2 * size_y) * (nrow - 1)), 1, nrow - 2))
        dzdx = (self._last_terrain[idx_y, idx_x + 1] -
               self._last_terrain[idx_y, idx_x - 1]) / (2 * dx)
        dzdy = (self._last_terrain[idx_y + 1, idx_x] -
               self._last_terrain[idx_y - 1, idx_x]) / (2 * dy)
        normal = np.array([-dzdx, -dzdy, 1.])
        return tuple(normal / np.linalg.norm(normal))

    @property
    def last_terrain(self):
        return self._last_terrain

    @property
    def last_calibrated_mogul_height(self):
        return self._last_calibrated_mogul_height

    @property
    def zone_types(self):
        return self._zone_types

    @property
    def zone_bounds_x(self):
        return self._zone_bounds_x

    @property
    def dim(self):
        return self._dim
