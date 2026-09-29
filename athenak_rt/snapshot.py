"""AthenaK ``.bin`` snapshot access for the TDE radiative post-processing.

Everything here streams the MeshBlock records of an ``Athena binary output
version=1.1`` file; nothing loads the full snapshot into memory.  The only
dense arrays are the uniform ray grids (``los_steps`` along the line of sight,
``image_size`` across it) onto which the finest available AMR level is
resampled by nearest-cell lookup.

The functions are the ones used by the validated scripts
(``light_curve_tde_cartesian.py``); their numerics are unchanged.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .constants import SAFE_POSITIVE

SNAPSHOT_RE = re.compile(r"(\d+)\.bin$")
BINARY_VERSION_LINE = "Athena binary output version=1.1\n"


@dataclass(frozen=True)
class UnitSystem:
    length_cgs: float
    mass_cgs: float
    time_cgs: float
    mu: float = 1.0

    @property
    def velocity_cgs(self) -> float:
        return self.length_cgs / self.time_cgs

    @property
    def density_cgs(self) -> float:
        return self.mass_cgs / self.length_cgs**3

    @property
    def pressure_cgs(self) -> float:
        return self.mass_cgs / (self.length_cgs * self.time_cgs**2)


@dataclass(frozen=True)
class SnapshotHeader:
    path: Path
    time_code: float
    cycle: int
    location_size: int
    variable_size: int
    variable_names: Tuple[str, ...]
    input_data: Dict[str, Dict[str, str]]
    data_offset: int
    location_format: str
    variable_dtype: np.dtype
    snapshot_index: int

    @property
    def dfloor_code(self) -> float:
        return float(self.input_data.get("hydro", {}).get("dfloor", "nan"))

    @property
    def eos_table_in_header(self) -> Optional[str]:
        """The ``<hydro>/table`` (or ``<mhd>/table``) path recorded in the dump."""
        for block in ("hydro", "mhd"):
            table = self.input_data.get(block, {}).get("table")
            if table:
                return table
        return None


@dataclass(frozen=True)
class RuntimeState:
    """Live black-hole / translating-frame state recorded in the dump header."""

    bh_x: float
    bh_y: float
    bh_z: float
    bh_vx: float
    bh_vy: float
    bh_vz: float
    bh_mass: float
    newton_g: float
    bh_softening: float
    bh_excise_radius: float
    frame_x: float
    frame_y: float
    frame_z: float
    use_translating_frame: bool

    @property
    def bh_inertial_xyz(self) -> Tuple[float, float, float]:
        return (
            self.frame_x + self.bh_x,
            self.frame_y + self.bh_y,
            self.frame_z + self.bh_z,
        )


@dataclass(frozen=True)
class RTBox:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class DenseGasBoxInfo:
    box: RTBox
    raw_box: RTBox
    threshold_code: float
    selected_cells: int
    max_density_code: float


@dataclass(frozen=True)
class BoxSettings:
    """How the ray-grid bounding box is chosen.

    ``box`` fixes it explicitly.  Otherwise the box is the padded bounding box of
    all cells with ``rho_code > threshold_code`` (strict), clipped to the mesh.
    """

    box: Optional[Tuple[float, float, float, float, float, float]] = None
    padding_code: float = 0.25
    padding_fraction: float = 0.10
    min_width_code: float = 1.0
    clip_to_mesh: bool = True


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def snapshot_index_from_path(path: Path) -> int:
    match = SNAPSHOT_RE.search(path.name)
    return int(match.group(1)) if match else -1


def read_snapshot_header(path: Path) -> SnapshotHeader:
    path = Path(path)
    with path.open("rb") as fp:
        version = fp.readline().decode("ascii")
        if version != BINARY_VERSION_LINE:
            raise RuntimeError(f"Unsupported binary format in {path}.")

        _ = fp.readline().decode("ascii")
        time_line = fp.readline().decode("ascii").strip()
        cycle_line = fp.readline().decode("ascii").strip()

        line = fp.readline().decode("ascii")
        if not line.startswith("  size of location="):
            raise RuntimeError(f"Could not read location size in {path}.")
        location_size = int(line.split("=", 1)[1].strip())

        line = fp.readline().decode("ascii")
        if not line.startswith("  size of variable="):
            raise RuntimeError(f"Could not read variable size in {path}.")
        variable_size = int(line.split("=", 1)[1].strip())

        line = fp.readline().decode("ascii")
        if not line.startswith("  number of variables="):
            raise RuntimeError(f"Could not read number of variables in {path}.")
        _ = int(line.split("=", 1)[1].strip())

        line = fp.readline().decode("ascii")
        if not line.startswith("  variables:"):
            raise RuntimeError(f"Could not read variable list in {path}.")
        variable_names = tuple(line[12:].split())

        line = fp.readline().decode("ascii")
        if not line.startswith("  header offset="):
            raise RuntimeError(f"Could not read header offset in {path}.")
        header_offset = int(line.split("=", 1)[1].strip())

        input_data: Dict[str, Dict[str, str]] = {}
        section_name: Optional[str] = None
        start_of_data = fp.tell() + header_offset
        while fp.tell() < start_of_data:
            raw = fp.readline().decode("ascii")
            if not raw:
                break
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("<") and stripped.endswith(">"):
                section_name = stripped[1:-1]
                input_data[section_name] = {}
                continue
            if section_name is None or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            input_data[section_name][key.strip()] = value.split("#", 1)[0].strip()

    location_format = "f" if location_size == 4 else "d"
    variable_dtype = np.float32 if variable_size == 4 else np.float64
    return SnapshotHeader(
        path=path,
        time_code=float(time_line.split("=", 1)[1].strip()),
        cycle=int(cycle_line.split("=", 1)[1].strip()),
        location_size=location_size,
        variable_size=variable_size,
        variable_names=variable_names,
        input_data=input_data,
        data_offset=start_of_data,
        location_format=location_format,
        variable_dtype=np.dtype(variable_dtype),
        snapshot_index=snapshot_index_from_path(path),
    )


def build_units(input_data: Dict[str, Dict[str, str]]) -> UnitSystem:
    units = input_data.get("units", {})
    if not units:
        raise RuntimeError(
            "The snapshot header has no <units> block; cgs units are required."
        )
    return UnitSystem(
        length_cgs=float(units["length_cgs"]),
        mass_cgs=float(units["mass_cgs"]),
        time_cgs=float(units["time_cgs"]),
        mu=float(units.get("mu", "1.0")),
    )


def build_runtime(header: SnapshotHeader) -> RuntimeState:
    problem = header.input_data.get("problem", {})
    gravity = header.input_data.get("gravity", {})
    if "external_newton_g" in problem:
        newton_g = float(problem["external_newton_g"])
    else:
        newton_g = float(gravity.get("four_pi_G", "1.0")) / (4.0 * math.pi)
    return RuntimeState(
        bh_x=float(problem.get("bh_live_x", "0.0")),
        bh_y=float(problem.get("bh_live_y", "0.0")),
        bh_z=float(problem.get("bh_live_z", "0.0")),
        bh_vx=float(problem.get("bh_live_vx", "0.0")),
        bh_vy=float(problem.get("bh_live_vy", "0.0")),
        bh_vz=float(problem.get("bh_live_vz", "0.0")),
        bh_mass=float(problem.get("external_bh_mass", problem.get("mass_ratio", "0.0"))),
        newton_g=newton_g,
        bh_softening=float(
            problem.get("external_bh_softening", problem.get("bh_softening", "0.0"))
        ),
        bh_excise_radius=float(problem.get("bh_excise_radius", "0.0")),
        frame_x=float(problem.get("frame_live_x", "0.0")),
        frame_y=float(problem.get("frame_live_y", "0.0")),
        frame_z=float(problem.get("frame_live_z", "0.0")),
        use_translating_frame=is_true(problem.get("use_translating_frame", "false")),
    )


def mesh_box_from_header(header: SnapshotHeader) -> RTBox:
    mesh = header.input_data.get("mesh", {})
    return RTBox(
        x_min=float(mesh["x1min"]),
        x_max=float(mesh["x1max"]),
        y_min=float(mesh["x2min"]),
        y_max=float(mesh["x2max"]),
        z_min=float(mesh["x3min"]),
        z_max=float(mesh["x3max"]),
    )


# ---------------------------------------------------------------------------
# Density thresholds.  Two rules are kept on purpose: they are the ones the
# validated runs used.  Both reduce to ``factor * dfloor`` when the problem
# block carries no separate rho_floor.
# ---------------------------------------------------------------------------
def choose_density_threshold(
    header: SnapshotHeader,
    factor: float,
    threshold_code: Optional[float] = None,
) -> float:
    """Resampling threshold: cells with ``rho_code > threshold`` enter the grid."""
    if threshold_code is not None:
        return float(threshold_code)
    hydro = header.input_data.get("hydro", {})
    dfloor = float(hydro.get("dfloor", "0.0"))
    return max(float(factor) * dfloor, 0.0)


def choose_auto_box_density_threshold(
    header: SnapshotHeader,
    factor: float,
    threshold_code: Optional[float] = None,
) -> float:
    """Bounding-box threshold: ``factor * max(hydro/dfloor, problem/rho_floor)``."""
    if threshold_code is not None:
        return float(threshold_code)
    hydro = header.input_data.get("hydro", {})
    problem = header.input_data.get("problem", {})
    floor = max(
        float(hydro.get("dfloor", "0.0")),
        float(problem.get("rho_floor", "0.0")),
    )
    return max(float(factor) * floor, 0.0)


def expand_box_axis(
    lo: float,
    hi: float,
    mesh_lo: float,
    mesh_hi: float,
    settings: BoxSettings,
) -> Tuple[float, float]:
    width = max(hi - lo, SAFE_POSITIVE)
    pad = max(float(settings.padding_code), float(settings.padding_fraction) * width)
    target_width = max(width + 2.0 * pad, float(settings.min_width_code))
    center = 0.5 * (lo + hi)
    out_lo = center - 0.5 * target_width
    out_hi = center + 0.5 * target_width

    if settings.clip_to_mesh:
        mesh_width = mesh_hi - mesh_lo
        if target_width >= mesh_width:
            return mesh_lo, mesh_hi
        if out_lo < mesh_lo:
            out_hi += mesh_lo - out_lo
            out_lo = mesh_lo
        if out_hi > mesh_hi:
            out_lo -= out_hi - mesh_hi
            out_hi = mesh_hi
        out_lo = max(out_lo, mesh_lo)
        out_hi = min(out_hi, mesh_hi)
    return out_lo, out_hi


def expand_dense_gas_box(raw_box: RTBox, mesh_box: RTBox, settings: BoxSettings) -> RTBox:
    x_min, x_max = expand_box_axis(
        raw_box.x_min, raw_box.x_max, mesh_box.x_min, mesh_box.x_max, settings
    )
    y_min, y_max = expand_box_axis(
        raw_box.y_min, raw_box.y_max, mesh_box.y_min, mesh_box.y_max, settings
    )
    z_min, z_max = expand_box_axis(
        raw_box.z_min, raw_box.z_max, mesh_box.z_min, mesh_box.z_max, settings
    )
    return RTBox(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
    )


def _read_exact(fp, nbytes: int, what: str, path: Path) -> bytes:
    raw = fp.read(nbytes)
    if len(raw) != nbytes:
        raise RuntimeError(f"Truncated {what} in {path}.")
    return raw


def dense_gas_box_from_snapshot(
    header: SnapshotHeader,
    threshold_code: float,
    settings: BoxSettings,
) -> DenseGasBoxInfo:
    """Stream the density field once and return the padded dense-gas bounding box."""
    if "dens" not in header.variable_names:
        raise RuntimeError(
            f"{header.path} must contain dens for automatic RT box selection."
        )

    dens_index = header.variable_names.index("dens")
    mesh_box = mesh_box_from_header(header)
    selected_cells = 0
    max_density_code = 0.0
    x_min = math.inf
    x_max = -math.inf
    y_min = math.inf
    y_max = -math.inf
    z_min = math.inf
    z_max = -math.inf

    with header.path.open("rb") as fp:
        fp.seek(0, 2)
        file_size = fp.tell()
        fp.seek(header.data_offset, 0)

        while fp.tell() < file_size:
            raw = fp.read(24)
            if not raw:
                break
            if len(raw) != 24:
                raise RuntimeError(f"Truncated MeshBlock index record in {header.path}.")
            ois, oie, ojs, oje, oks, oke = struct.unpack("@6i", raw)
            block_nx = oie - ois + 1
            block_ny = oje - ojs + 1
            block_nz = oke - oks + 1
            cells_per_block = block_nx * block_ny * block_nz
            variable_bytes = cells_per_block * header.variable_size

            _read_exact(fp, 16, "MeshBlock metadata", header.path)
            lims_raw = _read_exact(
                fp, 6 * header.location_size, "MeshBlock limits", header.path
            )
            x1min, x1max, x2min, x2max, x3min, x3max = struct.unpack(
                "=" + 6 * header.location_format, lims_raw
            )

            if dens_index:
                fp.seek(dens_index * variable_bytes, 1)
            payload = _read_exact(fp, variable_bytes, "density payload", header.path)
            remaining_variables = len(header.variable_names) - dens_index - 1
            if remaining_variables:
                fp.seek(remaining_variables * variable_bytes, 1)

            dens = np.frombuffer(payload, dtype=header.variable_dtype).reshape(
                (block_nz, block_ny, block_nx)
            )
            block_max = float(np.max(dens)) if dens.size else 0.0
            max_density_code = max(max_density_code, block_max)
            mask = dens > threshold_code
            if not np.any(mask):
                continue

            i_has_dense = np.any(mask, axis=(0, 1))
            j_has_dense = np.any(mask, axis=(0, 2))
            k_has_dense = np.any(mask, axis=(1, 2))
            ii = np.flatnonzero(i_has_dense)
            jj = np.flatnonzero(j_has_dense)
            kk = np.flatnonzero(k_has_dense)
            selected_cells += int(mask.sum())
            dx1 = (x1max - x1min) / block_nx
            dx2 = (x2max - x2min) / block_ny
            dx3 = (x3max - x3min) / block_nz
            x_min = min(x_min, x1min + int(ii.min()) * dx1)
            x_max = max(x_max, x1min + (int(ii.max()) + 1) * dx1)
            y_min = min(y_min, x2min + int(jj.min()) * dx2)
            y_max = max(y_max, x2min + (int(jj.max()) + 1) * dx2)
            z_min = min(z_min, x3min + int(kk.min()) * dx3)
            z_max = max(z_max, x3min + (int(kk.max()) + 1) * dx3)

    if selected_cells == 0:
        return DenseGasBoxInfo(
            box=mesh_box,
            raw_box=mesh_box,
            threshold_code=threshold_code,
            selected_cells=0,
            max_density_code=max_density_code,
        )

    raw_box = RTBox(
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max
    )
    return DenseGasBoxInfo(
        box=expand_dense_gas_box(raw_box, mesh_box, settings),
        raw_box=raw_box,
        threshold_code=threshold_code,
        selected_cells=selected_cells,
        max_density_code=max_density_code,
    )


def determine_rt_box(
    header: SnapshotHeader,
    settings: BoxSettings,
    auto_threshold_code: float,
) -> Tuple[RTBox, Optional[DenseGasBoxInfo]]:
    if settings.box is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = (float(v) for v in settings.box)
        return RTBox(x_min, x_max, y_min, y_max, z_min, z_max), None
    info = dense_gas_box_from_snapshot(header, auto_threshold_code, settings)
    if info.selected_cells == 0:
        raise RuntimeError(
            f"{header.path.name} has no cells above the adaptive RT threshold "
            f"rho_code>{info.threshold_code:.6e}; refusing a full-mesh empty-box "
            "fallback (pass --box to force one)."
        )
    return info.box, info


# ---------------------------------------------------------------------------
# Ray grid geometry.
# ---------------------------------------------------------------------------
def direction_code_and_sign(direction: str) -> Tuple[int, bool]:
    """``"z"`` -> (2, True): rays enter from +z and travel toward -z."""
    axis_name = direction.lstrip("-")
    if axis_name == "x":
        direction_code = 0
    elif axis_name == "y":
        direction_code = 1
    elif axis_name == "z":
        direction_code = 2
    else:
        raise RuntimeError(f"Unsupported direction {direction!r}.")
    positive = not direction.startswith("-")
    return direction_code, positive


def axes_for_direction(box: RTBox, direction: str, image_size: int, los_steps: int):
    if direction.lstrip("-") not in {"x", "y", "z"}:
        raise RuntimeError(f"Unsupported direction {direction!r}.")

    nx = los_steps if direction.lstrip("-") == "x" else image_size
    ny = los_steps if direction.lstrip("-") == "y" else image_size
    nz = los_steps if direction.lstrip("-") == "z" else image_size

    def centers(lo: float, hi: float, n: int) -> np.ndarray:
        dx = (hi - lo) / n
        return np.linspace(lo + 0.5 * dx, hi - 0.5 * dx, n, dtype=np.float64)

    x = centers(box.x_min, box.x_max, nx)
    y = centers(box.y_min, box.y_max, ny)
    z = centers(box.z_min, box.z_max, nz)
    return x, y, z


def axis_slice(axis: np.ndarray, lo: float, hi: float) -> slice:
    start = int(np.searchsorted(axis, lo, side="left"))
    stop = int(np.searchsorted(axis, hi, side="left"))
    start = max(start, 0)
    stop = min(stop, axis.size)
    return slice(start, stop)


def image_geometry_for_direction(
    direction: str,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Image axes (u = column, v = row), LOS step and pixel area in code units."""
    axis_name = direction.lstrip("-")
    if axis_name == "x":
        image_u = y_axis
        image_v = z_axis
        los_axis = x_axis
        du = (y_axis[-1] - y_axis[0]) / max(y_axis.size - 1, 1)
        dv = (z_axis[-1] - z_axis[0]) / max(z_axis.size - 1, 1)
    elif axis_name == "y":
        image_u = x_axis
        image_v = z_axis
        los_axis = y_axis
        du = (x_axis[-1] - x_axis[0]) / max(x_axis.size - 1, 1)
        dv = (z_axis[-1] - z_axis[0]) / max(z_axis.size - 1, 1)
    else:
        image_u = x_axis
        image_v = y_axis
        los_axis = z_axis
        du = (x_axis[-1] - x_axis[0]) / max(x_axis.size - 1, 1)
        dv = (y_axis[-1] - y_axis[0]) / max(y_axis.size - 1, 1)
    ds_code = abs(los_axis[1] - los_axis[0]) if los_axis.size > 1 else 0.0
    area_code = abs(du * dv)
    return image_u, image_v, ds_code, area_code


def grid_storage_dtypes(image_size: int, los_steps: int, grid_dtype: str):
    """Storage precision of the resampled (rho, T) cubes.

    ``"auto"`` keeps the validated rule: single precision once any axis reaches
    1024 samples (the paper's production runs), double precision below.  EOS
    recovery and all ray sums are always carried out in double precision.
    """
    if grid_dtype == "float32":
        compact = True
    elif grid_dtype == "float64":
        compact = False
    elif grid_dtype == "auto":
        compact = max(int(image_size), int(los_steps)) >= 1024
    else:
        raise RuntimeError(f"Unknown grid dtype {grid_dtype!r}.")
    if compact:
        return np.float32, np.int8
    return np.float64, np.int16


def resample_snapshot_to_grid(
    header: SnapshotHeader,
    eos,
    units: UnitSystem,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    density_threshold_code: float,
    grid_dtype=np.float64,
    level_dtype=np.int16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-cell resampling of (rho, T) [cgs] onto the ray grid.

    Blocks are streamed in file order; a grid cell takes the value of the block
    with the highest refinement level covering it (ties: later block wins), and
    only cells with ``rho_code > density_threshold_code`` are filled.  Empty cells
    keep rho = T = 0 and are transparent to every transfer kernel.
    """
    if "dens" not in header.variable_names or "eint" not in header.variable_names:
        raise RuntimeError(f"{header.path} must contain dens and eint for LTE RT.")

    rho_grid = np.zeros((z_axis.size, y_axis.size, x_axis.size), dtype=grid_dtype)
    temp_grid = np.zeros_like(rho_grid)
    level_grid = np.full(rho_grid.shape, -1, dtype=level_dtype)
    dens_index = header.variable_names.index("dens")
    eint_index = header.variable_names.index("eint")

    with header.path.open("rb") as fp:
        fp.seek(0, 2)
        file_size = fp.tell()
        fp.seek(header.data_offset, 0)

        while fp.tell() < file_size:
            raw = fp.read(24)
            if not raw:
                break
            if len(raw) != 24:
                raise RuntimeError(f"Truncated MeshBlock index record in {header.path}.")
            ois, oie, ojs, oje, oks, oke = struct.unpack("@6i", raw)
            block_nx = oie - ois + 1
            block_ny = oje - ojs + 1
            block_nz = oke - oks + 1
            cells_per_block = block_nx * block_ny * block_nz
            variable_bytes = cells_per_block * header.variable_size

            meta = _read_exact(fp, 16, "MeshBlock metadata", header.path)
            _, _, _, block_level = struct.unpack("@4i", meta)

            lims_raw = _read_exact(
                fp, 6 * header.location_size, "MeshBlock limits", header.path
            )
            x1min, x1max, x2min, x2max, x3min, x3max = struct.unpack(
                "=" + 6 * header.location_format, lims_raw
            )

            xs = axis_slice(x_axis, x1min, x1max)
            ys = axis_slice(y_axis, x2min, x2max)
            zs = axis_slice(z_axis, x3min, x3max)
            overlaps_grid = (
                xs.start < xs.stop and ys.start < ys.stop and zs.start < zs.stop
            )
            if not overlaps_grid:
                fp.seek(len(header.variable_names) * variable_bytes, 1)
                continue

            dens = None
            eint = None
            for ivar, _name in enumerate(header.variable_names):
                payload = _read_exact(fp, variable_bytes, "variable payload", header.path)
                if ivar == dens_index:
                    dens = (
                        np.frombuffer(payload, dtype=header.variable_dtype)
                        .astype(np.float64)
                        .reshape((block_nz, block_ny, block_nx))
                    )
                elif ivar == eint_index:
                    eint = (
                        np.frombuffer(payload, dtype=header.variable_dtype)
                        .astype(np.float64)
                        .reshape((block_nz, block_ny, block_nx))
                    )

            if dens is None or eint is None:
                raise RuntimeError(f"Could not read dens/eint from {header.path}.")

            dx1 = (x1max - x1min) / block_nx
            dx2 = (x2max - x2min) / block_ny
            dx3 = (x3max - x3min) / block_nz
            ii = np.clip(((x_axis[xs] - x1min) / dx1).astype(np.int64), 0, block_nx - 1)
            jj = np.clip(((y_axis[ys] - x2min) / dx2).astype(np.int64), 0, block_ny - 1)
            kk = np.clip(((z_axis[zs] - x3min) / dx3).astype(np.int64), 0, block_nz - 1)

            dens_values_code = dens[np.ix_(kk, jj, ii)]
            update = block_level >= level_grid[zs, ys, xs]
            update &= dens_values_code > density_threshold_code
            if not np.any(update):
                continue

            eint_values_code = eint[np.ix_(kk, jj, ii)]
            dens_update_code = dens_values_code[update]
            eint_update_code = eint_values_code[update]
            temp_code = eos.temperature_from_rho_eint(
                np.maximum(dens_update_code, SAFE_POSITIVE),
                np.maximum(eint_update_code, SAFE_POSITIVE),
            )
            temp_values_cgs = temp_code * eos.temp_unit_cgs
            rho_values_cgs = dens_update_code * units.density_cgs

            rho_view = rho_grid[zs, ys, xs]
            temp_view = temp_grid[zs, ys, xs]
            level_view = level_grid[zs, ys, xs]
            rho_view[update] = rho_values_cgs
            temp_view[update] = temp_values_cgs
            level_view[update] = block_level

    return rho_grid, temp_grid, level_grid


def apply_bh_excision_mask(
    rho_grid: np.ndarray,
    temp_grid: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    runtime: RuntimeState,
    radius: Optional[float] = None,
) -> float:
    """Zero rho and T inside the BH excision sphere.  Returns the radius used."""
    if radius is None:
        radius = runtime.bh_excise_radius
    radius = float(radius)
    if radius <= 0.0:
        return 0.0
    bx, by, bz = runtime.bh_inertial_xyz

    # Restrict the temporary distance array to the small index cube that can
    # intersect the excision sphere.  Constructing a full-domain r^2 array
    # would add 8 GiB at 1024^3 and defeats the memory-controlled RT path.
    ix = np.flatnonzero(np.abs(x_axis - bx) <= radius)
    iy = np.flatnonzero(np.abs(y_axis - by) <= radius)
    iz = np.flatnonzero(np.abs(z_axis - bz) <= radius)
    if ix.size == 0 or iy.size == 0 or iz.size == 0:
        return radius
    local_r2 = (
        (x_axis[ix][None, None, :] - bx) ** 2
        + (y_axis[iy][None, :, None] - by) ** 2
        + (z_axis[iz][:, None, None] - bz) ** 2
    )
    local_mask = local_r2 <= radius**2
    rho_local = rho_grid[np.ix_(iz, iy, ix)]
    temp_local = temp_grid[np.ix_(iz, iy, ix)]
    rho_local[local_mask] = 0.0
    temp_local[local_mask] = 0.0
    rho_grid[np.ix_(iz, iy, ix)] = rho_local
    temp_grid[np.ix_(iz, iy, ix)] = temp_local
    return radius
