"""Parameter file: parsing, errors, dump selection and end-to-end runs."""

import csv
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
import pytest

from athenak_rt import cli
from athenak_rt.config import RTSettings
from athenak_rt.params import (
    SCHEMA,
    ParameterError,
    format_dump_selection,
    load_parameter_file,
    parse_blocks,
    parse_dump_selection,
    parse_parameters,
    run_parameter_file,
)
from test_pipeline import TIME_CGS, write_eos_table, write_snapshot

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MINIMAL = "<input>\ndump_dir = .\ndumps = 1\n"


def write(directory: Path, text: str, name: str = "run.rtin") -> Path:
    path = directory / name
    path.write_text(text)
    return path


def quiet(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# Parsing and defaults.
# ---------------------------------------------------------------------------
def test_defaults_match_rtsettings(tmp_path):
    config = load_parameter_file(write(tmp_path, MINIMAL))
    default = RTSettings()
    for item in fields(RTSettings):
        got = getattr(config.settings, item.name)
        if item.name == "output_dir":
            assert got == (tmp_path / "athenak_rt_output").resolve()
        else:
            assert got == getattr(default, item.name), item.name
    assert config.directions == ("z",)
    assert config.dump_dir == tmp_path.resolve()
    assert config.dumps == (1,)
    assert (config.basename, config.variable) == ("TDEExternalLTEPrad", "hydro_w")
    assert config.missing_dumps == "error" and config.skip_existing is False
    assert config.dump_path(1).name == "TDEExternalLTEPrad.hydro_w.00001.bin"


def test_every_setting_has_a_key(tmp_path):
    for name in ("eos.table", "high.data", "low.data"):
        (tmp_path / name).write_text("")
    text = """
# every key with a non-default value
<input>
dump_dir      = dumps           # relative to the parameter file
basename      = Other
variable      = prim
dumps         = 5, 7:11:2
missing_dumps = skip
<output>
dir              = out
hdf5_compression = none
bands            = false
<transfer>
mode            = tau1
directions      = -x, +y
tau_photosphere = 2.0
tau_stop        = 12
nfreq           = 20
emin_ev         = 1
emax_ev         = 50
scattering      = no
populations     = saha
<image>
image_size                = 64
los_steps                 = 32
grid_dtype                = float32
box                       = -1, 1, -2, 2, -3, 3
auto_box_padding_code     = 0.5
auto_box_padding_fraction = 0.2
auto_box_min_width        = 2
auto_box_clip_to_mesh     = false
<selection>
density_threshold_factor = 3
density_threshold_code   = 1e-12
bh_mask                  = off
bh_mask_radius           = 0.1
<tables>
eos_table   = eos.table
mesa_high_t = high.data
mesa_low_t  = low.data
<run>
threads       = 2
skip_existing = true
"""
    config = load_parameter_file(write(tmp_path, text))
    default = RTSettings()
    for item in fields(RTSettings):
        assert getattr(config.settings, item.name) != getattr(default, item.name), item.name
    s = config.settings
    assert config.directions == ("-x", "y") and s.direction == "-x"
    assert config.dumps == (5, 7, 9, 11)
    assert config.dump_dir == (tmp_path / "dumps").resolve()
    assert s.output_dir == (tmp_path / "out").resolve()
    assert s.eos_table == (tmp_path / "eos.table").resolve()
    assert s.box.box == (-1.0, 1.0, -2.0, 2.0, -3.0, 3.0)
    assert s.box.clip_to_mesh is False and s.box.min_width_code == 2.0
    assert s.observation_bands == () and s.hdf5_compression is None
    assert s.threads == 2 and config.skip_existing and config.missing_dumps == "skip"

    # The resolved text is itself a parameter file that gives the same run.
    again = parse_parameters(config.resolved_text(), base_dir="/")
    assert again.settings == config.settings
    assert again.directions == config.directions and again.dumps == config.dumps
    assert again.dump_dir == config.dump_dir and again.basename == "Other"


def test_resolved_text_round_trips_dumps_all():
    config = parse_parameters("<input>\ndump_dir = .\ndumps = all\n", base_dir="/")
    assert config.dumps is None
    resolved = config.resolved_text()
    assert "dumps         = all" in resolved
    assert parse_parameters(resolved, base_dir="/").dumps is None
    # ``auto`` is still written for the keys that have it
    assert any(
        line.startswith("bh_mask_radius") and line.endswith("= auto")
        for line in resolved.splitlines()
    )


def test_examples_list_every_key_and_use_the_paper_settings():
    for name in ("tde_snapshot.rtin", "tde_lightcurve.rtin"):
        path = EXAMPLES / name
        blocks = parse_blocks(path.read_text(), str(path))
        assert {b: set(k) for b, k in blocks.items()} == {
            b: set(k) for b, k in SCHEMA.items()
        }
        config = load_parameter_file(path)
        s = config.settings
        assert s.mode == "multifreq" and config.directions == ("z", "-y")
        assert (s.nfreq, s.emin_ev, s.emax_ev) == (73, 0.1, 1000.0)
        assert s.density_threshold_factor == 10.0 and s.density_threshold_code is None
        assert s.populations == "eos" and s.scattering and s.bh_mask
    single = load_parameter_file(EXAMPLES / "tde_snapshot.rtin")
    assert single.dumps == (327,) and single.settings.image_size == 1024
    series = load_parameter_file(EXAMPLES / "tde_lightcurve.rtin")
    assert series.dumps == tuple(range(300, 328)) and series.skip_existing


@pytest.mark.parametrize(
    "text,match",
    [
        ("<inptu>\n", r"line 1: unknown block <inptu> \(did you mean 'input'\?\)"),
        ("<input\n", "cannot read block header"),
        ("dumps = 1\n", "before the first <block>"),
        (
            MINIMAL + "<image>\nimgae_size = 4\n",
            r"line 5: unknown key 'imgae_size' in <image> \(did you mean 'image_size'\?\)",
        ),
        (MINIMAL + "<image>\nthreads = 4\n", r"'threads' belongs to <run>"),
        (MINIMAL + "<image>\nimage_size 64\n", "expected 'key = value'"),
        (MINIMAL + "<image>\n= 64\n", "missing key"),
        (
            MINIMAL + "<image>\nimage_size = 64\nimage_size = 32\n",
            "line 6: <image>/image_size is already set on line 5",
        ),
        (MINIMAL + "<image>\nimage_size =\n", "has no value"),
        (MINIMAL + "<image>\nimage_size = 64.5\n", "expected an integer"),
        (MINIMAL + "<image>\nimage_size = 1\n", ">= 2"),
        (MINIMAL + "<image>\nbox = 0, 1, 0, 1, 1, 0\n", "zmin must be smaller than zmax"),
        (MINIMAL + "<image>\nbox = 0, 1, 0, 1\n", "six numbers"),
        (
            MINIMAL + "<transfer>\nmode = mulitfreq\n",
            "must be one of tau1, grey, grey-therm, multifreq",
        ),
        (MINIMAL + "<transfer>\ndirections = z, w\n", "unknown direction 'w'"),
        (MINIMAL + "<transfer>\ndirections = z, +z\n", "given twice"),
        (MINIMAL + "<transfer>\nscattering = maybe\n", "expected true or false"),
        (MINIMAL + "<transfer>\nemin_ev = 10\nemax_ev = 1\n", "emin_ev < emax_ev"),
        (MINIMAL + "<transfer>\nnfreq = 1\n", "nfreq must be >= 2"),
        (MINIMAL + "<selection>\ndensity_threshold_factor = nan\n", "finite"),
        (MINIMAL + "<selection>\nbh_mask_radius = -1\n", "bh_mask_radius must be >= 0"),
        (MINIMAL + "<tables>\neos_table = missing.table\n", "eos_table: file not found"),
        (MINIMAL + "<run>\nthreads = 0\n", "threads must be >= 1"),
        (MINIMAL + "<input>\ndumps = 2\n", "<input>/dumps is already set on line 3"),
        ("<input>\ndump_dir = .\n", "<input>/dumps is required"),
        ("<input>\ndumps = 3\n", "<input>/dump_dir is required"),
        ("<input>\ndump_dir = .\ndumps = 5:1\n", "ends before it starts"),
        ("<input>\ndump_dir = .\ndumps = 0:99999999999\n", "more than 1000000 dumps"),
        ("<input>\ndump_dir = .\ndumps = 0:1999999, 5\n", "more than 1000000 dumps"),
        ("<input>\ndump_dir = .\ndumps = 1_0\n", "cannot read '1_0'"),
        ("<input>\ndump_dir = .\ndumps = \uff11\uff12\n", "cannot read"),
        (MINIMAL + "<image>\nimage_size = 1_0\n", "line 5.*expected an integer"),
        (MINIMAL + "<image>\nimage_size = \u0661\u0662\n", "expected an integer"),
        (MINIMAL + "<transfer>\ntau_stop = 1_0.5\n", "expected a number"),
        (MINIMAL + "<transfer>\ntau_stop = \uff11\n", "expected a number"),
        (MINIMAL + "<image>\nbox = 0_0, 1, 0, 1, 0, 1\n", "expected a number"),
        # range errors of RTSettings.validate name the line of the key
        (MINIMAL + "<transfer>\nnfreq = 1\n", r"run.rtin, line 5: nfreq must be >= 2"),
        (MINIMAL + "<image>\nimage_size = 1\n", r"line 5: image_size and los_steps"),
        (MINIMAL + "<transfer>\nemin_ev = 10\nemax_ev = 1\n", r"line 5: The photon"),
        (MINIMAL + "<run>\nthreads = 0\n", r"line 5: threads must be >= 1"),
        (MINIMAL + "<selection>\nbh_mask_radius = -1\n", r"line 5: <selection>/bh_mask_radius"),
        (MINIMAL + "<input>\nbasename = a/b\n", r"line 5: <input>/basename"),
    ],
)
def test_parameter_errors(tmp_path, text, match):
    with pytest.raises(ParameterError, match=match):
        load_parameter_file(write(tmp_path, text))


def test_blocks_can_reopen_and_comments_are_ignored():
    text = (
        "<input>  # inputs\ndump_dir = .\n\n<run>\nthreads = 3 # three\n"
        "<input>\ndumps = 4\n"
    )
    config = parse_parameters(text)
    assert config.dumps == (4,) and config.settings.threads == 3


# ---------------------------------------------------------------------------
# Dump selection.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("327", (327,)),
        ("00327", (327,)),
        ("310, 327", (310, 327)),
        ("310 327", (310, 327)),
        ("300:303", (300, 301, 302, 303)),
        ("300:306:2", (300, 302, 304, 306)),
        ("300:305:2", (300, 302, 304)),
        ("300 : 304 : 2, 301", (300, 301, 302, 304)),
        ("5, 3, 5", (3, 5)),
        ("7:7", (7,)),
        ("all", None),
        ("ALL", None),
    ],
)
def test_dump_selection(text, expected):
    assert parse_dump_selection(text) == expected


@pytest.mark.parametrize(
    "text", ["", "306:300", "300:306:0", "a", "-3", "300:", ":5", "1:2:3:4", "3.5", "1-5"]
)
def test_dump_selection_errors(text):
    with pytest.raises(ValueError):
        parse_dump_selection(text)


@pytest.mark.parametrize(
    "dumps,text",
    [
        (tuple(range(300, 328)), "300:327"),
        ((300, 302, 304, 306, 327), "300:306:2, 327"),
        ((1, 5), "1, 5"),
        ((1, 2, 4, 5, 6), "1, 2, 4:6"),
        (None, "all"),
    ],
)
def test_format_dump_selection(dumps, text):
    assert format_dump_selection(dumps) == text
    assert parse_dump_selection(text) == dumps


def test_find_dumps_missing_and_all(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for index in (10, 11, 13):
        (bin_dir / f"X.hydro_w.{index:05d}.bin").write_bytes(b"")
    (bin_dir / "Y.hydro_w.00012.bin").write_bytes(b"")
    (bin_dir / "X.hydro_u.00012.bin").write_bytes(b"")
    (bin_dir / "X.hydro_w.00014.bin.part").write_bytes(b"")
    base = "<input>\ndump_dir = bin\nbasename = X\n"

    config = load_parameter_file(write(tmp_path, base + "dumps = 10:14\n"))
    match = r"2 of 5 selected dumps .* 12, 14\. .*missing_dumps = skip"
    with pytest.raises(ParameterError, match=match):
        config.find_dumps(log=quiet)

    messages = []
    config = load_parameter_file(
        write(tmp_path, base + "dumps = 10:14\nmissing_dumps = skip\n")
    )
    found = config.find_dumps(log=messages.append)
    assert [p.name for p in found] == [f"X.hydro_w.{i:05d}.bin" for i in (10, 11, 13)]
    assert len(messages) == 1 and "skipping 2 missing dump(s)" in messages[0]
    assert messages[0].endswith("12, 14")

    config = load_parameter_file(
        write(tmp_path, base + "dumps = 20:22\nmissing_dumps = skip\n")
    )
    with pytest.raises(ParameterError, match="none of the 3 selected dumps"):
        config.find_dumps(log=quiet)

    config = load_parameter_file(write(tmp_path, base + "dumps = all\n"))
    assert [p.name for p in config.find_dumps(log=quiet)] == [
        f"X.hydro_w.{i:05d}.bin" for i in (10, 11, 13)
    ]

    config = load_parameter_file(write(tmp_path, "<input>\ndump_dir = nowhere\ndumps = 1\n"))
    with pytest.raises(ParameterError, match="is not a directory"):
        config.find_dumps(log=quiet)


def test_cli_takes_only_a_parameter_file(tmp_path, capsys):
    bad = write(tmp_path, MINIMAL + "<image>\nimgae_size = 4\n")
    assert cli.main([str(bad)]) == 2
    assert "unknown key 'imgae_size'" in capsys.readouterr().err
    assert cli.main([str(tmp_path / "absent.rtin")]) == 2
    assert "parameter file not found" in capsys.readouterr().err
    for argv in ([], [str(bad), "--mode", "tau1"], [str(bad), str(bad)]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# End-to-end runs on the synthetic AMR dump.
# ---------------------------------------------------------------------------
def make_run(tmp_path: Path, series=((42, 1.5, 1.0),)) -> Path:
    """<tmp>/run/bin/TDETest.hydro_w.NNNNN.bin plus the EOS table in <tmp>.

    The dump header records the table as ../test_ideal_h.table, relative to
    the run directory <tmp>/run, as the example decks do.
    """
    write_eos_table(tmp_path / "test_ideal_h.table")
    bin_dir = tmp_path / "run" / "bin"
    bin_dir.mkdir(parents=True)
    for index, time, t_scale in series:
        write_snapshot(
            bin_dir / f"TDETest.hydro_w.{index:05d}.bin",
            "../test_ideal_h.table",
            time=time,
            cycle=10 * index,
            t_scale=t_scale,
        )
    return bin_dir


def test_single_dump_end_to_end(tmp_path, capsys):
    make_run(tmp_path)
    text = """<input>
dump_dir = run/bin
basename = TDETest
dumps    = 42

<output>
dir              = out
hdf5_compression = none

<transfer>
mode       = tau1
directions = z, -y

<image>
image_size = 8
los_steps  = 8

<selection>
bh_mask = false

<run>
threads = 1
"""
    path = write(tmp_path, text)
    assert cli.main([str(path)]) == 0
    stdout = capsys.readouterr().out
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == [
        "TDETest.hydro_w.00042.rt_tau1_-y.h5",
        "TDETest.hydro_w.00042.rt_tau1_z.h5",
    ]  # one dump: no light curve, no partial files
    config = load_parameter_file(path)
    for direction in ("z", "-y"):
        product = out / f"TDETest.hydro_w.00042.rt_tau1_{direction}.h5"
        assert f"tau1 {direction}: L_bol,iso" in stdout
        with h5py.File(product, "r") as h5:
            assert h5.attrs["direction"] == direction
            assert h5.attrs["valid_pixels"] == 64
            assert h5.attrs["luminosity_bolometric_erg_s"] > 0.0
            assert h5.attrs["time_code"] == 1.5 and h5.attrs["snapshot_index"] == 42
            assert h5.attrs["eos_table_resolved"] == str(
                (tmp_path / "test_ideal_h.table").resolve()
            )
            assert h5["parameters"].attrs["path"] == str(path.resolve())
            assert h5["parameters/parameter_file"].asstr()[()] == text
            resolved = h5["parameters/resolved"].asstr()[()]
        assert resolved == config.resolved_text()
        assert parse_parameters(resolved).settings == config.settings


def lightcurve_text(mode: str, dumps: str, extra: str = "") -> str:
    return f"""<input>
dump_dir      = run/bin
basename      = TDETest
dumps         = {dumps}
missing_dumps = skip

<output>
dir = out

<transfer>
mode       = {mode}
directions = z, -y
nfreq      = 8

<image>
image_size = 8
los_steps  = 8

<run>
threads = 1
{extra}"""


def test_lightcurve_end_to_end(tmp_path):
    make_run(tmp_path, series=((40, 1.0, 1.0), (42, 2.0, 1.1), (44, 3.0, 1.2)))
    path = write(tmp_path, lightcurve_text("multifreq", "40:46:2"))
    messages = []
    results = run_parameter_file(path, log=messages.append)
    assert len(results) == 6
    assert any("skipping 1 missing dump(s)" in m and m.endswith("46") for m in messages)

    out = tmp_path / "out"
    csv_path, h5_path = out / "rt_lightcurve_multifreq.csv", out / "rt_lightcurve_multifreq.h5"
    with csv_path.open() as fp:
        rows = list(csv.DictReader(fp))
    header = list(rows[0])
    assert header[:6] == ["dump", "cycle", "time_code", "time_s", "L_bol_z", "L_bol_minus_y"]
    assert header[6] == "nuLnu_nir_J_z" and header[-1] == "nuLnu_xray_2keV_minus_y"
    assert len(header) == 6 + 2 * 14
    assert [int(r["dump"]) for r in rows] == [40, 42, 44]
    assert [int(r["cycle"]) for r in rows] == [400, 420, 440]
    assert [float(r["time_code"]) for r in rows] == [1.0, 2.0, 3.0]
    assert [float(r["time_s"]) for r in rows] == [t * TIME_CGS for t in (1.0, 2.0, 3.0)]
    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    assert table.dtype.names == tuple(header)

    for row in rows:
        for direction, label in (("z", "z"), ("-y", "minus_y")):
            stem = f"TDETest.hydro_w.{int(row['dump']):05d}"
            product = out / f"{stem}.rt_multifreq_{direction}.h5"
            with h5py.File(product, "r") as h5:
                assert float(row[f"L_bol_{label}"]) == h5.attrs["luminosity_bolometric_erg_s"]
                assert float(row[f"nuLnu_optical_g_{label}"]) == h5["bands/nu_lnu_erg_s"][4]
                assert h5.attrs["populations_used"] == "saha"  # synthetic table: fallback

    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["mode"] == "multifreq"
        assert list(h5.attrs["directions"]) == ["z", "-y"]
        assert h5["dump"][...].tolist() == [40, 42, 44]
        assert h5["time_code"][...].tolist() == [1.0, 2.0, 3.0]
        assert h5["bands/label"].shape == (14,) and h5["spectra/energy_ev"].shape == (8,)
        for direction, label in (("z", "z"), ("-y", "minus_y")):
            lbol = h5[f"{direction}/luminosity_bolometric_erg_s"][...]
            assert lbol.tolist() == [float(r[f"L_bol_{label}"]) for r in rows]
            assert h5[f"{direction}/bands/nu_lnu_erg_s"].shape == (3, 14)
            assert h5[f"{direction}/spectra/lnu_erg_s_hz"].shape == (3, 8)
            assert h5[f"{direction}/bh_xyz_code"].shape == (3, 3)
        assert h5["parameters/parameter_file"].asstr()[()] == path.read_text()
    assert not list(out.glob("*.part"))


def test_skip_existing_reuses_matching_outputs(tmp_path, capsys):
    make_run(tmp_path, series=((40, 1.0, 1.0), (41, 2.0, 1.1)))
    path = write(tmp_path, lightcurve_text("tau1", "40, 41", "skip_existing = true\n"))
    assert len(run_parameter_file(path, log=quiet)) == 4
    csv_path = tmp_path / "out" / "rt_lightcurve_tau1.csv"
    first = csv_path.read_text()
    assert first.startswith("dump,cycle,time_code,time_s,L_bol_z,L_bol_minus_y,nuLnu_")

    # Remove one product: only that one is recomputed.
    (tmp_path / "out" / "TDETest.hydro_w.00041.rt_tau1_-y.h5").unlink()
    messages = []
    results = run_parameter_file(path, log=messages.append)
    assert [(r.snapshot_index, r.direction) for r in results] == [(41, "-y")]
    assert sum(m.endswith("(reused)") for m in messages) == 3
    assert csv_path.read_text() == first

    # A changed setting must not silently reuse the old products.
    path.write_text(path.read_text().replace("image_size = 8", "image_size = 4"))
    match = r"does not match the current settings or input dump \(image_size = 8 there, 4 now\)"
    with pytest.raises(RuntimeError, match=match):
        run_parameter_file(path, log=quiet)
    capsys.readouterr()
    assert cli.main([str(path)]) == 1
    assert "skip_existing = false" in capsys.readouterr().err


def test_single_dump_without_skip_overwrites(tmp_path):
    make_run(tmp_path)
    text = lightcurve_text("grey", "42")
    path = write(tmp_path, text)
    assert len(run_parameter_file(path, log=quiet)) == 2
    assert len(run_parameter_file(path, log=quiet)) == 2
    assert not (tmp_path / "out" / "rt_lightcurve_grey.csv").exists()


# ---------------------------------------------------------------------------
# Restart checks, light-curve merge, threads, API.
# ---------------------------------------------------------------------------
def product_path(tmp_path: Path, mode: str, direction: str, dump: int) -> Path:
    return tmp_path / "out" / f"TDETest.hydro_w.{dump:05d}.rt_{mode}_{direction}.h5"


def lightcurve_dumps(tmp_path: Path, mode: str) -> list:
    with h5py.File(tmp_path / "out" / f"rt_lightcurve_{mode}.h5", "r") as h5:
        return h5["dump"][...].tolist()


def test_skip_existing_checks_the_input_dump(tmp_path):
    bin_dir = make_run(tmp_path, series=((40, 1.0, 1.0), (41, 2.0, 1.1)))
    path = write(tmp_path, lightcurve_text("tau1", "40, 41", "skip_existing = true\n"))
    assert len(run_parameter_file(path, log=quiet)) == 4
    assert len(run_parameter_file(path, log=quiet)) == 0

    # Same name, other content: a different time, then a different cycle.
    dump = bin_dir / "TDETest.hydro_w.00041.bin"
    write_snapshot(dump, "../test_ideal_h.table", time=2.5, cycle=410, t_scale=1.1)
    with pytest.raises(RuntimeError, match=r"input dump differs: time_code = 2.0 there, 2.5 now"):
        run_parameter_file(path, log=quiet)
    write_snapshot(dump, "../test_ideal_h.table", time=2.0, cycle=999, t_scale=1.1)
    with pytest.raises(RuntimeError, match=r"input dump differs: cycle = 410 there, 999 now"):
        run_parameter_file(path, log=quiet)
    write_snapshot(dump, "../test_ideal_h.table", time=2.0, cycle=410, t_scale=1.1)
    assert len(run_parameter_file(path, log=quiet)) == 0

    # The same dumps in another directory (same names) are the same input.
    other = tmp_path / "copy" / "bin"
    other.mkdir(parents=True)
    for name in ("TDETest.hydro_w.00040.bin", "TDETest.hydro_w.00041.bin"):
        (other / name).write_bytes((bin_dir / name).read_bytes())
    (tmp_path / "copy" / "test_ideal_h.table").write_bytes(
        (tmp_path / "test_ideal_h.table").read_bytes()
    )
    text = path.read_text().replace("run/bin", "copy/bin")
    assert len(run_parameter_file(write(tmp_path, text, "copy.rtin"), log=quiet)) == 0
    # ... but not when the time differs.
    write_snapshot(other / "TDETest.hydro_w.00040.bin", "../test_ideal_h.table",
                   time=1.25, cycle=400, t_scale=1.0)
    with pytest.raises(RuntimeError, match="input dump differs"):
        run_parameter_file(tmp_path / "copy.rtin", log=quiet)


def test_skip_existing_ignores_settings_the_mode_does_not_use(tmp_path):
    make_run(tmp_path)
    base = lightcurve_text("tau1", "42", "skip_existing = true\n")
    path = write(tmp_path, base)
    assert len(run_parameter_file(path, log=quiet)) == 2
    unused = base.replace("nfreq      = 8", "nfreq      = 9\npopulations = saha\nemin_ev = 0.5\nemax_ev = 5\n"
                          "scattering = false\ntau_stop = 3")
    assert "nfreq      = 9" in unused
    assert len(run_parameter_file(write(tmp_path, unused), log=quiet)) == 0
    # tau_photosphere does matter in tau1 mode
    changed = base.replace("nfreq      = 8", "nfreq = 8\ntau_photosphere = 2")
    with pytest.raises(RuntimeError, match=r"tau_photosphere = 1.0 there, 2.0 now"):
        run_parameter_file(write(tmp_path, changed), log=quiet)

    # multifreq: tau_photosphere is unused, nfreq is not
    text = lightcurve_text("multifreq", "42", "skip_existing = true\n")
    assert len(run_parameter_file(write(tmp_path, text), log=quiet)) == 2
    same = text.replace("nfreq      = 8", "nfreq = 8\ntau_photosphere = 5")
    assert len(run_parameter_file(write(tmp_path, same), log=quiet)) == 0
    with pytest.raises(RuntimeError, match=r"nfreq = 8 there, 9 now"):
        run_parameter_file(write(tmp_path, text.replace("nfreq      = 8", "nfreq = 9")), log=quiet)


def test_skip_existing_compares_table_hashes_not_paths(tmp_path):
    make_run(tmp_path)
    text = lightcurve_text("tau1", "42", "skip_existing = true\n")
    table = tmp_path / "test_ideal_h.table"
    moved = tmp_path / "elsewhere.table"
    moved.write_bytes(table.read_bytes())
    path = write(tmp_path, text + f"\n<tables>\neos_table = {table}\n")
    assert len(run_parameter_file(path, log=quiet)) == 2
    with h5py.File(product_path(tmp_path, "tau1", "z", 42), "r") as h5:
        assert len(h5.attrs["eos_table_sha256"]) == 64
        assert len(h5.attrs["mesa_high_t_sha256"]) == 64

    # Same content under another path: reused.
    path = write(tmp_path, text + f"\n<tables>\neos_table = {moved}\n")
    assert len(run_parameter_file(path, log=quiet)) == 0
    # Other content: refused.
    moved.write_bytes(table.read_bytes() + b"\0")
    with pytest.raises(RuntimeError, match="eos_table file differs"):
        run_parameter_file(path, log=quiet)

    # A product from an older version has no hashes: the paths are compared.
    moved.write_bytes(table.read_bytes())
    for direction in ("z", "-y"):
        with h5py.File(product_path(tmp_path, "tau1", direction, 42), "r+") as h5:
            for name in ("eos_table_sha256", "mesa_high_t_sha256", "mesa_low_t_sha256"):
                del h5.attrs[name]
    with pytest.raises(RuntimeError, match="eos_table_setting"):
        run_parameter_file(path, log=quiet)
    path = write(tmp_path, text + f"\n<tables>\neos_table = {table}\n")
    assert len(run_parameter_file(path, log=quiet)) == 0


def test_lightcurve_merges_earlier_matching_products(tmp_path):
    make_run(tmp_path, series=((40, 1.0, 1.0), (41, 2.0, 1.1), (42, 3.0, 1.2), (43, 4.0, 1.3)))
    messages = []
    run_parameter_file(write(tmp_path, lightcurve_text("tau1", "40")), log=messages.append)
    assert not (tmp_path / "out" / "rt_lightcurve_tau1.csv").exists()  # one row only
    assert not any(m.startswith("Light curve") for m in messages)

    run_parameter_file(write(tmp_path, lightcurve_text("tau1", "43")), log=quiet)
    assert lightcurve_dumps(tmp_path, "tau1") == [40, 43]
    run_parameter_file(write(tmp_path, lightcurve_text("tau1", "41, 42")), log=quiet)
    assert lightcurve_dumps(tmp_path, "tau1") == [40, 41, 42, 43]
    with (tmp_path / "out" / "rt_lightcurve_tau1.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert [int(r["dump"]) for r in rows] == [40, 41, 42, 43]
    assert [float(r["time_code"]) for r in rows] == [1.0, 2.0, 3.0, 4.0]

    # Products made with other settings are left out.
    other = lightcurve_text("tau1", "41, 42").replace("image_size = 8", "image_size = 4")
    run_parameter_file(write(tmp_path, other), log=quiet)
    assert lightcurve_dumps(tmp_path, "tau1") == [41, 42]

    # A dump that lacks one of the directions is left out.
    product_path(tmp_path, "tau1", "-y", 40).unlink()
    run_parameter_file(write(tmp_path, lightcurve_text("tau1", "42, 43")), log=quiet)
    assert lightcurve_dumps(tmp_path, "tau1") == [42, 43]

    # Other modes have their own light curve.
    run_parameter_file(write(tmp_path, lightcurve_text("grey", "40, 41")), log=quiet)
    assert lightcurve_dumps(tmp_path, "grey") == [40, 41]
    assert not list((tmp_path / "out").glob("*.part"))


def test_lightcurve_skips_products_of_another_dump_with_the_same_name(tmp_path):
    bin_dir = make_run(tmp_path, series=((40, 1.0, 1.0), (41, 2.0, 1.1), (42, 3.0, 1.2)))
    run_parameter_file(write(tmp_path, lightcurve_text("tau1", "40, 41")), log=quiet)
    assert lightcurve_dumps(tmp_path, "tau1") == [40, 41]
    # Dump 40 of this directory is another snapshot than the one behind the product.
    write_snapshot(bin_dir / "TDETest.hydro_w.00040.bin", "../test_ideal_h.table",
                   time=9.0, cycle=1, t_scale=1.0)
    run_parameter_file(write(tmp_path, lightcurve_text("tau1", "41, 42")), log=quiet)
    assert lightcurve_dumps(tmp_path, "tau1") == [41, 42]


def test_reused_products_are_reported(tmp_path, capsys):
    make_run(tmp_path)
    path = write(tmp_path, lightcurve_text("tau1", "42", "skip_existing = true\n"))
    assert cli.main([str(path)]) == 0
    first = capsys.readouterr().out
    assert first.count("L_bol,iso") == 2 and "(reused)" not in first
    assert cli.main([str(path)]) == 0
    second = capsys.readouterr().out
    lines = [l for l in second.splitlines() if "L_bol,iso" in l]
    assert len(lines) == 2 and all(l.endswith("(reused)") for l in lines)
    assert lines[0].startswith("TDETest.hydro_w.00042.bin tau1 z: L_bol,iso = ")
    computed = [l for l in first.splitlines() if "L_bol,iso" in l]
    assert [l + " (reused)" for l in computed] == lines


def test_threads_are_capped_at_the_numba_limit():
    import numba

    from athenak_rt.pipeline import configure_threads

    cap = numba.config.NUMBA_NUM_THREADS
    messages = []
    assert configure_threads(cap + 100, messages.append) == cap
    assert len(messages) == 1 and "at most" in messages[0] and str(cap) in messages[0]
    messages.clear()
    assert configure_threads(1, messages.append) == 1 and not messages
    assert 1 <= configure_threads(None, messages.append) <= cap
    configure_threads(cap, messages.append)
