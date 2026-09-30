"""End-to-end test on a synthetic two-level AMR snapshot and a synthetic EOS table."""

import math
import struct
from pathlib import Path

import h5py
import numpy as np
import pytest

from athenak_rt.config import RTSettings
from athenak_rt.constants import HYDROGEN_MASS_CGS, K_B_CGS, SIGMA_SB_CGS
from athenak_rt.eos import TabulatedLteTable, resolve_eos_table
from athenak_rt.opacity import MesaOpacityModel
from athenak_rt.pipeline import EosCache, process_snapshot
from athenak_rt.snapshot import (
    BoxSettings,
    axes_for_direction,
    build_runtime,
    build_units,
    determine_rt_box,
    read_snapshot_header,
    resample_snapshot_to_grid,
)

LENGTH_CGS = 1.0e10
MASS_CGS = 1.0e33
TIME_CGS = 1.0e3
DENSITY_UNIT = MASS_CGS / LENGTH_CGS**3  # 1e3 g/cm^3
PRESSURE_UNIT = MASS_CGS / (LENGTH_CGS * TIME_CGS**2)
EPS_UNIT = PRESSURE_UNIT / DENSITY_UNIT
NB = 8  # cells per block edge
DFLOOR = 1.0e-14
RHO_ROOT_CODE = 1.0e-9  # 1e-6 g/cm^3
RHO_FINE_CODE = 4.0e-9
T_ROOT = 2.0e4
T_FINE = 5.0e4
BH_EXCISE_RADIUS = 0.35


def eps_of_temperature(temp):
    """Ideal monatomic H gas, mu = 1: eps = 1.5 k T / m_H."""
    return 1.5 * K_B_CGS * temp / HYDROGEN_MASS_CGS


def write_eos_table(path: Path) -> None:
    logrho = np.linspace(math.log(1e-12), math.log(1e2), 40)
    logtemp = np.linspace(math.log(1e3), math.log(1e8), 60)
    rho = np.exp(logrho)[:, None]
    temp = np.exp(logtemp)[None, :]
    fields = {
        "logpress": np.log(rho * K_B_CGS * temp / HYDROGEN_MASS_CGS),
        "logeps": np.log(eps_of_temperature(temp)) + 0.0 * rho,
        "logcs2": np.log(5.0 / 3.0 * K_B_CGS * temp / HYDROGEN_MASS_CGS) + 0.0 * rho,
        "gamma1": np.full((40, 60), 5.0 / 3.0),
        "gamma3m1": np.full((40, 60), 2.0 / 3.0),
        "xion": np.zeros((40, 60)),
    }
    header = (
        "<metadatabegin>\ntable_type = test_ideal_h\neos_name = test\nlog_axis_base = e\n"
        "endianness = little\n<metadataend>\n<scalarsbegin>\n<scalarsend>\n"
        "<pointsbegin>\nlogrho = 40\nlogtemp = 60\n<pointsend>\n<fieldsbegin>\n"
        + "".join(f"{name}\n" for name in fields)
        + "<fieldsend>\n"
    )
    with path.open("wb") as fp:
        fp.write(header.encode("ascii"))
        fp.write(logrho.astype("<f8").tobytes())
        fp.write(logtemp.astype("<f8").tobytes())
        for values in fields.values():
            fp.write(np.ascontiguousarray(values, dtype="<f8").tobytes())


def write_snapshot(
    path: Path, eos_table, time: float = 1.5, cycle: int = 7, t_scale: float = 1.0
) -> None:
    """Root block on [-1,1]^3 (level 0) plus one level-1 block on [0,1]^3.

    ``t_scale`` multiplies both temperatures (a different dump of a series).
    """
    input_text = (
        "<mesh>\nx1min = -1.0\nx1max = 1.0\nx2min = -1.0\nx2max = 1.0\n"
        "x3min = -1.0\nx3max = 1.0\n"
        f"<units>\nlength_cgs = {LENGTH_CGS}\nmass_cgs = {MASS_CGS}\n"
        f"time_cgs = {TIME_CGS}\n"
        f"<hydro>\neos = lte_table_test\ntable = {eos_table}\ndfloor = {DFLOOR}\n"
        f"<problem>\nbh_live_x = 0.5\nbh_live_y = -0.5\nbh_live_z = 0.0\n"
        f"bh_excise_radius = {BH_EXCISE_RADIUS}\n"
    ).encode("ascii")

    def block(level, lims, rho_code, temp):
        temp = temp * t_scale
        dens = np.full((NB, NB, NB), rho_code, dtype=np.float32)
        eint = np.full(
            (NB, NB, NB), rho_code * eps_of_temperature(temp) / EPS_UNIT, dtype=np.float32
        )
        out = struct.pack("@6i", 0, NB - 1, 0, NB - 1, 0, NB - 1)
        out += struct.pack("@4i", 0, 0, 0, level)
        out += struct.pack("=6d", *lims)
        out += dens.tobytes() + eint.tobytes()
        return out

    with path.open("wb") as fp:
        fp.write(b"Athena binary output version=1.1\n")
        fp.write(b"  preheader size=9\n")
        fp.write(f"  time={time!r}\n".encode("ascii"))
        fp.write(f"  cycle={cycle}\n".encode("ascii"))
        fp.write(b"  size of location=8\n")
        fp.write(b"  size of variable=4\n")
        fp.write(b"  number of variables=2\n")
        fp.write(b"  variables:  dens  eint\n")
        fp.write(f"  header offset={len(input_text)}\n".encode("ascii"))
        fp.write(input_text)
        fp.write(block(0, (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), RHO_ROOT_CODE, T_ROOT))
        fp.write(block(1, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), RHO_FINE_CODE, T_FINE))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    eos_table = root / "test_ideal_h.table"
    snapshot = root / "TDETest.hydro_w.00042.bin"
    write_eos_table(eos_table)
    write_snapshot(snapshot, eos_table)
    return snapshot, eos_table


def test_eos_table_round_trip(synthetic):
    _, eos_table = synthetic
    units = build_units(read_snapshot_header(synthetic[0]).input_data)
    eos = TabulatedLteTable.from_units(eos_table, units)
    for temp in (3.0e3, T_ROOT, T_FINE, 2.0e7):
        dens_code = np.array([RHO_ROOT_CODE])
        eint_code = dens_code * eps_of_temperature(temp) / EPS_UNIT
        t_code = eos.temperature_from_rho_eint(dens_code, eint_code)
        assert t_code[0] * eos.temp_unit_cgs == pytest.approx(temp, rel=1e-10)


def test_header_and_resampling(synthetic):
    snapshot, eos_table = synthetic
    header = read_snapshot_header(snapshot)
    assert header.variable_names == ("dens", "eint")
    assert header.snapshot_index == 42 and header.cycle == 7 and header.time_code == 1.5
    assert header.eos_table_in_header == str(eos_table)
    assert resolve_eos_table(None, header.eos_table_in_header) == eos_table.resolve()
    units = build_units(header.input_data)
    runtime = build_runtime(header)
    assert runtime.bh_inertial_xyz == (0.5, -0.5, 0.0)
    assert runtime.bh_excise_radius == BH_EXCISE_RADIUS

    box, info = determine_rt_box(header, BoxSettings(), 10.0 * DFLOOR)
    assert info is not None and info.selected_cells == 2 * NB**3
    assert box.as_array().tolist() == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0]

    eos = TabulatedLteTable.from_units(eos_table, units)
    x, y, z = axes_for_direction(box, "z", 8, 16)
    rho, temp, level = resample_snapshot_to_grid(
        header, eos, units, x, y, z, 10.0 * DFLOOR
    )
    assert rho.shape == (16, 8, 8)
    fine = (z[:, None, None] > 0) & (y[None, :, None] > 0) & (x[None, None, :] > 0)
    assert np.allclose(rho[fine], RHO_FINE_CODE * DENSITY_UNIT, rtol=1e-6)
    assert np.allclose(rho[~fine], RHO_ROOT_CODE * DENSITY_UNIT, rtol=1e-6)
    assert np.allclose(temp[fine], T_FINE, rtol=1e-6)
    assert np.allclose(temp[~fine], T_ROOT, rtol=1e-6)
    assert np.all(level[fine] == 1) and np.all(level[~fine] == 0)
    # A threshold above the root density keeps only the refined octant.
    rho2, _, _ = resample_snapshot_to_grid(
        header, eos, units, x, y, z, 2.0 * RHO_ROOT_CODE
    )
    assert np.all(rho2[~fine] == 0.0) and np.all(rho2[fine] > 0.0)


@pytest.mark.parametrize("direction", ["z", "-y"])
def test_tau1_pipeline_end_to_end(synthetic, tmp_path, direction):
    snapshot, eos_table = synthetic
    kappa = 1.0e5  # each cell is optically thick
    settings = RTSettings(
        mode="tau1",
        direction=direction,
        image_size=8,
        los_steps=8,
        output_dir=tmp_path / "out",
        eos_table=eos_table,
        bh_mask=False,
        threads=1,
        hdf5_compression=None,
    )
    result = process_snapshot(
        snapshot,
        settings,
        MesaOpacityModel.constant_kappa(kappa),
        EosCache(),
        log=lambda *_: None,
    )
    # The first cell along every ray is the photosphere.  Seen from +z the
    # refined octant (x>0, y>0) shows T_FINE; seen from -y nothing refined is in front.
    area = (2.0 / 8) ** 2 * LENGTH_CGS**2
    tph = result.maps["photosphere_temp_K"]
    if direction == "z":
        expected_hot = 16
    else:
        expected_hot = 0
    assert np.count_nonzero(np.isclose(tph, T_FINE, rtol=1e-6)) == expected_hot
    assert result.valid_pixels == 64
    expected_l = (
        4.0
        * SIGMA_SB_CGS
        * area
        * (expected_hot * T_FINE**4 + (64 - expected_hot) * T_ROOT**4)
    )
    assert result.luminosity == pytest.approx(expected_l, rel=1e-6)
    assert result.hdf5_path is not None and result.hdf5_path.is_file()
    with h5py.File(result.hdf5_path, "r") as h5:
        assert h5.attrs["mode"] == "tau1"
        assert h5.attrs["direction"] == direction
        assert h5.attrs["luminosity_bolometric_erg_s"] == pytest.approx(
            expected_l, rel=1e-6
        )
        assert h5.attrs["density_threshold_code"] == pytest.approx(10.0 * DFLOOR)
        assert h5.attrs["eos_table_resolved"] == str(eos_table.resolve())
        assert h5["maps/valid"][...].all()
        assert h5["bands/lnu_erg_s_hz"].shape == (14,)
        assert np.allclose(
            np.sum(h5["bands/lnu_pixel_erg_s_hz"][...], axis=(1, 2)),
            h5["bands/lnu_erg_s_hz"][...],
        )


def test_bh_mask_and_multifreq_pipeline(synthetic, tmp_path):
    snapshot, eos_table = synthetic
    settings = RTSettings(
        mode="multifreq",
        direction="z",
        image_size=8,
        los_steps=8,
        nfreq=24,
        output_dir=tmp_path / "out",
        eos_table=eos_table,
        bh_mask=True,
        threads=1,
        hdf5_compression=None,
    )
    result = process_snapshot(
        snapshot,
        settings,
        MesaOpacityModel.constant_kappa(0.4),
        EosCache(),
        log=lambda *_: None,
    )
    assert result.bh_mask_radius_code == BH_EXCISE_RADIUS
    assert result.spectrum is not None and result.spectrum["lnu_erg_s_hz"].shape == (24,)
    assert result.luminosity > 0.0
    assert result.luminosity == pytest.approx(
        4.0
        * math.pi
        * result.area_cm2
        * float(np.sum(result.maps["intensity_erg_s_cm2_sr"]))
    )
    with h5py.File(result.hdf5_path, "r") as h5:
        assert h5.attrs["bh_mask_radius_code"] == BH_EXCISE_RADIUS
        assert h5.attrs["scattering"]
        assert h5["spectra/energy_ev"].shape == (24,)
        assert h5["bands/lnu_erg_s_hz"].shape == (14,)
        teff = h5["maps/effective_temp_K"][...]
        assert np.all(np.isfinite(teff)) and np.all(teff > 0.0)
