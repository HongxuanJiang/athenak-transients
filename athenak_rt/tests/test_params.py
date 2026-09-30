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
    assert sum(m.startswith("Skipping") for m in messages) == 3
    assert csv_path.read_text() == first

    # A changed setting must not silently reuse the old products.
    path.write_text(path.read_text().replace("image_size = 8", "image_size = 4"))
    match = r"does not match the current settings \(image_size = 8 there, 4 now\)"
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
