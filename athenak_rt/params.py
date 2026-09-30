"""Parameter file of ``python -m athenak_rt PARAMETER_FILE``.

The format is that of AthenaK input files::

    <input>
    dump_dir = /path/to/run/bin     # directory of the .bin dumps
    dumps    = 300:327              # one dump, a list, or an inclusive range

    <transfer>
    mode       = multifreq
    directions = z, -y

Blocks are written in angle brackets, every other line is ``key = value`` and
``#`` starts a comment anywhere on a line (a path containing ``#`` cannot be
given).  Unknown blocks or keys, keys given twice and values that cannot be
read are errors.  A key that is left out takes its default
(``SCHEMA`` below).  Relative paths are relative to the directory of the
parameter file; ``~`` and ``$VARIABLES`` are expanded.
"""

from __future__ import annotations

import difflib
import math
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import __version__
from .bands import DEFAULT_OBSERVATION_BANDS
from .config import GRID_DTYPES, MODES, POPULATIONS, RTSettings, SettingsError
from .opacity import DEFAULT_MESA_HIGH_T, DEFAULT_MESA_LOW_T
from .snapshot import BoxSettings

DIRECTIONS = ("x", "y", "z", "-x", "-y", "-z")
MISSING_DUMP_POLICIES = ("error", "skip")
HDF5_COMPRESSIONS = ("lzf", "gzip", "none")


class ParameterError(ValueError):
    """A parameter file that cannot be used (syntax, unknown key, bad value)."""


@dataclass(frozen=True)
class Key:
    """One parameter: value type, default and whether ``auto`` is accepted.

    ``kind`` is one of str, path, int, float, bool, choice, directions, box,
    dumps.  ``auto`` keys accept the word ``auto`` (stored as None), which
    means "derive the value" as described in ``doc``.
    """

    kind: str
    default: object = None
    required: bool = False
    auto: bool = False
    choices: Tuple[str, ...] = ()
    doc: str = ""


_S = RTSettings()
_B = BoxSettings()

SCHEMA: Dict[str, Dict[str, Key]] = {
    "input": {
        "dump_dir": Key("path", required=True, doc="directory with the .bin dumps"),
        "basename": Key(
            "str", "TDEExternalLTEPrad", doc="<job>/basename of the AthenaK run"
        ),
        "variable": Key("str", "hydro_w", doc="output variable of the dumps"),
        "dumps": Key(
            "dumps",
            required=True,
            doc="N, a list N1, N2, an inclusive range A:B or A:B:STEP, or all",
        ),
        "missing_dumps": Key(
            "choice",
            "error",
            choices=MISSING_DUMP_POLICIES,
            doc="selected dumps that do not exist: error, or skip with a warning",
        ),
    },
    "output": {
        "dir": Key("path", _S.output_dir, doc="output directory"),
        "hdf5_compression": Key(
            "choice", "lzf", choices=HDF5_COMPRESSIONS, doc="filter for the maps"
        ),
        "bands": Key("bool", True, doc="NIR/optical/UV/X-ray band products"),
    },
    "transfer": {
        "mode": Key("choice", _S.mode, choices=MODES, doc="transfer mode"),
        "directions": Key(
            "directions", (_S.direction,), doc="lines of sight among x y z -x -y -z"
        ),
        "tau_photosphere": Key(
            "float", _S.tau_photosphere, doc="tau1: tau of the photosphere"
        ),
        "tau_stop": Key("float", _S.tau_stop, doc="stop a ray past this optical depth"),
        "nfreq": Key("int", _S.nfreq, doc="multifreq: number of photon energies"),
        "emin_ev": Key("float", _S.emin_ev, doc="multifreq: lowest photon energy [eV]"),
        "emax_ev": Key("float", _S.emax_ev, doc="multifreq: highest photon energy [eV]"),
        "scattering": Key("bool", _S.scattering, doc="multifreq: thermalization depth"),
        "populations": Key(
            "choice", _S.populations, choices=POPULATIONS, doc="H/He populations"
        ),
    },
    "image": {
        "image_size": Key("int", _S.image_size, doc="pixels across the image"),
        "los_steps": Key("int", _S.los_steps, doc="samples along the line of sight"),
        "grid_dtype": Key(
            "choice", _S.grid_dtype, choices=GRID_DTYPES, doc="storage of the ray grid"
        ),
        "box": Key(
            "box", None, auto=True, doc="xmin, xmax, ymin, ymax, zmin, zmax, or auto"
        ),
        "auto_box_padding_code": Key("float", _B.padding_code, doc="auto box padding"),
        "auto_box_padding_fraction": Key(
            "float", _B.padding_fraction, doc="auto box fractional padding"
        ),
        "auto_box_min_width": Key("float", _B.min_width_code, doc="auto box min width"),
        "auto_box_clip_to_mesh": Key("bool", _B.clip_to_mesh, doc="clip box to mesh"),
    },
    "selection": {
        "density_threshold_factor": Key(
            "float", _S.density_threshold_factor, doc="rho_code > factor * dfloor"
        ),
        "density_threshold_code": Key(
            "float", None, auto=True, doc="explicit threshold, overrides the factor"
        ),
        "bh_mask": Key("bool", _S.bh_mask, doc="empty the BH excision sphere"),
        "bh_mask_radius": Key(
            "float", None, auto=True, doc="excision radius; auto: header value"
        ),
    },
    "tables": {
        "eos_table": Key("path", None, auto=True, doc="auto: path in the dump header"),
        "mesa_high_t": Key("path", None, auto=True, doc="auto: bundled OPAL table"),
        "mesa_low_t": Key("path", None, auto=True, doc="auto: bundled low-T table"),
    },
    "run": {
        "threads": Key("int", None, auto=True, doc="numba threads; auto: all CPUs"),
        "skip_existing": Key(
            "bool", False, doc="reuse per-dump outputs that already exist"
        ),
    },
}

_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}
_BLOCK_RE = re.compile(r"<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>")
_INT_RE = re.compile(r"[0-9]+")  # ASCII digits only
MAX_DUMP_SELECTION = 1_000_000


# ---------------------------------------------------------------------------
# Value parsers.  Each raises ValueError with a short reason.
# ---------------------------------------------------------------------------
def _split_list(text: str) -> List[str]:
    return [item for item in re.split(r"[\s,]+", text.strip()) if item]


def parse_dump_selection(text: str) -> Optional[Tuple[int, ...]]:
    """Dump numbers selected by ``text``, sorted and unique; None for ``all``.

    Items are separated by commas or blanks: ``N``, ``A:B`` (A to B inclusive)
    or ``A:B:STEP``.  A selection of more than ``MAX_DUMP_SELECTION`` dumps is
    refused (use ``all`` to take every dump present).
    """
    text = text.strip()
    if text.lower() == "all":
        return None
    items = _split_list(re.sub(r"\s*:\s*", ":", text))
    if not items:
        raise ValueError("empty dump selection")
    selected = set()
    for item in items:
        parts = item.split(":")
        if len(parts) > 3 or not all(_INT_RE.fullmatch(p) for p in parts):
            raise ValueError(
                f"cannot read {item!r}; use N, A:B or A:B:STEP with "
                "non-negative integers, or all"
            )
        numbers = [int(p) for p in parts]
        if len(numbers) == 1:
            selected.add(numbers[0])
            continue
        start, stop = numbers[0], numbers[1]
        step = numbers[2] if len(numbers) == 3 else 1
        if step < 1:
            raise ValueError(f"range {item!r} needs a step >= 1")
        if stop < start:
            raise ValueError(f"range {item!r} ends before it starts")
        if (stop - start) // step + 1 > MAX_DUMP_SELECTION:
            raise ValueError(
                f"range {item!r} selects more than {MAX_DUMP_SELECTION} dumps; "
                "narrow the range or use all"
            )
        selected.update(range(start, stop + 1, step))
        if len(selected) > MAX_DUMP_SELECTION:
            raise ValueError(
                f"the selection has more than {MAX_DUMP_SELECTION} dumps; "
                "narrow it or use all"
            )
    return tuple(sorted(selected))


def format_dump_selection(dumps: Optional[Sequence[int]]) -> str:
    """Compact text for a dump list: runs of three or more become A:B[:STEP]."""
    if dumps is None:
        return "all"
    values = sorted(set(int(d) for d in dumps))
    items: List[str] = []
    i = 0
    while i < len(values):
        j = i
        if i + 1 < len(values):
            step = values[i + 1] - values[i]
            while j + 1 < len(values) and values[j + 1] - values[j] == step:
                j += 1
        if j - i >= 2:
            items.append(
                f"{values[i]}:{values[j]}" + ("" if step == 1 else f":{step}")
            )
            i = j + 1
        else:
            items.append(str(values[i]))
            i += 1
    return ", ".join(items)


def parse_directions(text: str) -> Tuple[str, ...]:
    out: List[str] = []
    for item in _split_list(text):
        direction = item[1:] if item.startswith("+") else item
        if direction not in DIRECTIONS:
            raise ValueError(f"unknown direction {item!r}; use {', '.join(DIRECTIONS)}")
        if direction in out:
            raise ValueError(f"direction {direction!r} is given twice")
        out.append(direction)
    if not out:
        raise ValueError("no direction given")
    return tuple(out)


def _ascii_number(text: str, what: str) -> None:
    """Reject underscores and non-ASCII digits, which int() and float() accept."""
    if not text.isascii() or "_" in text:
        raise ValueError(what)


def _parse_float(text: str) -> float:
    _ascii_number(text, "expected a number")
    try:
        value = float(text)
    except ValueError:
        raise ValueError("expected a number") from None
    if not math.isfinite(value):
        raise ValueError("expected a finite number")
    return value


def _parse_int(text: str) -> int:
    _ascii_number(text, "expected an integer")
    try:
        return int(text)
    except ValueError:
        raise ValueError("expected an integer") from None


def _parse_bool(text: str) -> bool:
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError("expected true or false")


def parse_box(text: str) -> Tuple[float, float, float, float, float, float]:
    items = _split_list(text)
    if len(items) != 6:
        raise ValueError("expected auto or six numbers xmin, xmax, ymin, ymax, zmin, zmax")
    values = tuple(_parse_float(v) for v in items)
    for axis, (lo, hi) in zip("xyz", zip(values[0::2], values[1::2])):
        if not lo < hi:
            raise ValueError(f"{axis}min must be smaller than {axis}max")
    return values  # type: ignore[return-value]


def _parse_path(text: str, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _convert(spec: Key, text: str, base_dir: Path):
    if spec.auto and text.lower() == "auto":
        return None
    kind = spec.kind
    if kind == "str":
        return text
    if kind == "path":
        return _parse_path(text, base_dir)
    if kind == "int":
        return _parse_int(text)
    if kind == "float":
        return _parse_float(text)
    if kind == "bool":
        return _parse_bool(text)
    if kind == "choice":
        if text not in spec.choices:
            raise ValueError(f"must be one of {', '.join(spec.choices)}")
        return text
    if kind == "directions":
        return parse_directions(text)
    if kind == "box":
        return parse_box(text)
    if kind == "dumps":
        return parse_dump_selection(text)
    raise AssertionError(f"unknown key kind {kind!r}")


def _format_value(spec: Key, value) -> str:
    if spec.kind == "dumps":  # None means ``all`` here, not ``auto``
        return format_dump_selection(value)
    if value is None:
        return "auto"
    if spec.kind == "bool":
        return "true" if value else "false"
    if spec.kind == "float":
        return repr(float(value))
    if spec.kind == "directions":
        return ", ".join(value)
    if spec.kind == "box":
        return ", ".join(repr(float(v)) for v in value)
    return str(value)


def _suggestion(name: str, candidates) -> str:
    close = difflib.get_close_matches(name, list(candidates), n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


# ---------------------------------------------------------------------------
# Reading the file.
# ---------------------------------------------------------------------------
def parse_blocks(text: str, source: str = "<string>") -> Dict[str, Dict[str, Tuple[str, int]]]:
    """Blocks -> {key: (raw value, line number)}; checks names against SCHEMA."""
    blocks: Dict[str, Dict[str, Tuple[str, int]]] = {}
    current: Optional[str] = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{source}, line {lineno}"
        if line.startswith("<"):
            match = _BLOCK_RE.fullmatch(line)
            if match is None:
                raise ParameterError(f"{where}: cannot read block header {line!r}")
            name = match.group(1)
            if name not in SCHEMA:
                raise ParameterError(
                    f"{where}: unknown block <{name}>{_suggestion(name, SCHEMA)}; "
                    f"valid blocks are {', '.join(f'<{b}>' for b in SCHEMA)}"
                )
            current = name
            blocks.setdefault(name, {})
            continue
        if "=" not in line:
            raise ParameterError(f"{where}: expected 'key = value', got {line!r}")
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise ParameterError(f"{where}: missing key before '='")
        if current is None:
            raise ParameterError(f"{where}: {key!r} appears before the first <block>")
        if key not in SCHEMA[current]:
            owners = [b for b, keys in SCHEMA.items() if key in keys]
            hint = (
                f" ({key!r} belongs to <{owners[0]}>)"
                if owners
                else _suggestion(key, SCHEMA[current])
            )
            raise ParameterError(
                f"{where}: unknown key {key!r} in <{current}>{hint}; valid keys are "
                f"{', '.join(SCHEMA[current])}"
            )
        if key in blocks[current]:
            first = blocks[current][key][1]
            raise ParameterError(
                f"{where}: <{current}>/{key} is already set on line {first}"
            )
        if not value:
            raise ParameterError(
                f"{where}: <{current}>/{key} has no value (leave the line out to "
                "use the default)"
            )
        blocks[current][key] = (value, lineno)
    return blocks


def _where(raw, source: str, block: str, key: str) -> str:
    """``source, line N`` if the key was given in the file, else just ``source``."""
    entry = raw.get(block, {}).get(key)
    return f"{source}, line {entry[1]}" if entry else source


# RTSettings field -> (block, key) for range errors reported by validate().
_FIELD_KEYS = {"direction": ("transfer", "directions")}
for _block, _keys in SCHEMA.items():
    for _key in _keys:
        _FIELD_KEYS.setdefault(_key, (_block, _key))


@dataclass
class RunConfig:
    """A parsed parameter file: the dumps to process and the run settings."""

    settings: RTSettings  # direction = directions[0]
    directions: Tuple[str, ...]
    dump_dir: Path
    basename: str
    variable: str
    dumps: Optional[Tuple[int, ...]]  # None: every dump found in dump_dir
    missing_dumps: str
    skip_existing: bool
    values: Dict[str, Dict[str, object]]  # every key, defaults filled in
    parameter_text: str = ""
    parameter_file: Optional[Path] = None

    def settings_for(self, direction: str) -> RTSettings:
        return replace(self.settings, direction=direction)

    def dump_path(self, index: int) -> Path:
        return self.dump_dir / f"{self.basename}.{self.variable}.{int(index):05d}.bin"

    def find_dumps(self, log: Callable[[str], None] = print) -> List[Path]:
        """Paths of the selected dumps in index order (see <input>/missing_dumps)."""
        if not self.dump_dir.is_dir():
            raise ParameterError(f"<input>/dump_dir {self.dump_dir} is not a directory")
        if self.dumps is None:
            pattern = re.compile(
                rf"{re.escape(self.basename)}\.{re.escape(self.variable)}\.(\d+)\.bin"
            )
            found = []
            for path in self.dump_dir.iterdir():
                match = pattern.fullmatch(path.name)
                if match and path.is_file():
                    found.append((int(match.group(1)), path))
            if not found:
                raise ParameterError(
                    f"no dumps {self.basename}.{self.variable}.NNNNN.bin in {self.dump_dir}"
                )
            return [path for _, path in sorted(found)]
        paths = [self.dump_path(i) for i in self.dumps]
        missing = [i for i, p in zip(self.dumps, paths) if not p.is_file()]
        if missing:
            listing = format_dump_selection(missing)
            pattern = f"{self.basename}.{self.variable}.NNNNN.bin"
            if len(missing) == len(paths):
                raise ParameterError(
                    f"none of the {len(paths)} selected dumps ({pattern}) exists in "
                    f"{self.dump_dir}: {listing}. Check <input>/dump_dir, basename, "
                    "variable and dumps."
                )
            if self.missing_dumps == "error":
                raise ParameterError(
                    f"{len(missing)} of {len(paths)} selected dumps ({pattern}) are "
                    f"missing in {self.dump_dir}: {listing}. Correct <input>/dumps, or "
                    "set <input>/missing_dumps = skip to process the others."
                )
            log(
                f"WARNING: skipping {len(missing)} missing dump(s) in "
                f"{self.dump_dir}: {listing}"
            )
            paths = [p for p in paths if p.is_file()]
        return paths

    def resolved_text(self) -> str:
        """The parameter file with every key and its value used (paths absolute)."""
        lines = [f"# athenak_rt {__version__}: resolved parameters"]
        for block, keys in SCHEMA.items():
            lines.append(f"<{block}>")
            width = max(len(k) for k in keys)
            for key, spec in keys.items():
                value = _format_value(spec, self.values[block][key])
                lines.append(f"{key:<{width}} = {value}")
            lines.append("")
        return "\n".join(lines)

    def provenance(self) -> Dict[str, str]:
        """What every output records about the parameter file."""
        return {
            "path": "" if self.parameter_file is None else str(self.parameter_file),
            "parameter_file": self.parameter_text,
            "resolved": self.resolved_text(),
        }


def parse_parameters(
    text: str, base_dir: os.PathLike = ".", source: str = "<string>"
) -> RunConfig:
    """Parse parameter-file text; relative paths are taken relative to ``base_dir``."""
    base_dir = Path(base_dir).expanduser().resolve()
    raw = parse_blocks(text, source)
    values: Dict[str, Dict[str, object]] = {}
    for block, keys in SCHEMA.items():
        values[block] = {}
        given = raw.get(block, {})
        for key, spec in keys.items():
            if key not in given:
                if spec.required:
                    raise ParameterError(f"{source}: <{block}>/{key} is required")
                values[block][key] = spec.default
                continue
            text_value, lineno = given[key]
            try:
                values[block][key] = _convert(spec, text_value, base_dir)
            except ValueError as exc:
                raise ParameterError(
                    f"{source}, line {lineno}: <{block}>/{key} = {text_value}: {exc}"
                ) from None

    tables = values["tables"]
    if tables["mesa_high_t"] is None:
        tables["mesa_high_t"] = DEFAULT_MESA_HIGH_T
    if tables["mesa_low_t"] is None:
        tables["mesa_low_t"] = DEFAULT_MESA_LOW_T
    for key, value in tables.items():
        if value is not None and not Path(value).is_file():
            raise ParameterError(f"{source}: <tables>/{key}: file not found: {value}")

    inp, out = values["input"], values["output"]
    tr, img = values["transfer"], values["image"]
    sel, run = values["selection"], values["run"]
    directions = tr["directions"]
    settings = RTSettings(
        mode=tr["mode"],
        direction=directions[0],
        image_size=img["image_size"],
        los_steps=img["los_steps"],
        output_dir=_parse_path(str(out["dir"]), base_dir),
        box=BoxSettings(
            box=img["box"],
            padding_code=img["auto_box_padding_code"],
            padding_fraction=img["auto_box_padding_fraction"],
            min_width_code=img["auto_box_min_width"],
            clip_to_mesh=img["auto_box_clip_to_mesh"],
        ),
        eos_table=tables["eos_table"],
        mesa_high_t=tables["mesa_high_t"],
        mesa_low_t=tables["mesa_low_t"],
        density_threshold_factor=sel["density_threshold_factor"],
        density_threshold_code=sel["density_threshold_code"],
        bh_mask=sel["bh_mask"],
        bh_mask_radius=sel["bh_mask_radius"],
        tau_photosphere=tr["tau_photosphere"],
        tau_stop=tr["tau_stop"],
        nfreq=tr["nfreq"],
        emin_ev=tr["emin_ev"],
        emax_ev=tr["emax_ev"],
        scattering=tr["scattering"],
        populations=tr["populations"],
        grid_dtype=img["grid_dtype"],
        observation_bands=DEFAULT_OBSERVATION_BANDS if out["bands"] else (),
        hdf5_compression=None
        if out["hdf5_compression"] == "none"
        else out["hdf5_compression"],
        threads=run["threads"],
    )
    values["output"]["dir"] = settings.output_dir
    for direction in directions:
        try:
            replace(settings, direction=direction).validate()
        except SettingsError as exc:
            where = source
            for name in exc.fields:
                block, key = _FIELD_KEYS.get(name, ("", name))
                if key in raw.get(block, {}):
                    where = _where(raw, source, block, key)
                    break
            raise ParameterError(f"{where}: {exc}") from None
        except RuntimeError as exc:
            raise ParameterError(f"{source}: {exc}") from None
    for block, key in (
        ("image", "auto_box_padding_code"),
        ("image", "auto_box_padding_fraction"),
        ("selection", "density_threshold_code"),
        ("selection", "bh_mask_radius"),
    ):
        value = values[block][key]
        if value is not None and value < 0.0:
            raise ParameterError(
                f"{_where(raw, source, block, key)}: <{block}>/{key} must be >= 0"
            )
    if img["auto_box_min_width"] <= 0.0:
        where = _where(raw, source, "image", "auto_box_min_width")
        raise ParameterError(f"{where}: <image>/auto_box_min_width must be > 0")
    for name in ("basename", "variable"):
        if "/" in inp[name] or not inp[name]:
            where = _where(raw, source, "input", name)
            raise ParameterError(f"{where}: <input>/{name} must be a plain file-name part")

    return RunConfig(
        settings=settings,
        directions=directions,
        dump_dir=inp["dump_dir"],
        basename=inp["basename"],
        variable=inp["variable"],
        dumps=inp["dumps"],
        missing_dumps=inp["missing_dumps"],
        skip_existing=run["skip_existing"],
        values=values,
        parameter_text=text,
    )


def load_parameter_file(path: os.PathLike) -> RunConfig:
    """Read a parameter file; relative paths in it are relative to its directory."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ParameterError(f"parameter file not found: {path}")
    path = path.resolve()
    text = path.read_text(encoding="utf-8")
    config = parse_parameters(text, base_dir=path.parent, source=str(path))
    config.parameter_file = path
    return config


def run_parameter_file(path: os.PathLike, log: Callable[[str], None] = print):
    """Process every selected dump and direction of a parameter file.

    Returns the ``RTResult`` of every (dump, direction) that was computed
    (products reused through ``<run>/skip_existing`` are only logged).
    """
    config = load_parameter_file(path)
    snapshots = config.find_dumps(log=log)
    from .pipeline import run

    return run(
        config.settings,
        snapshots,
        log,
        directions=config.directions,
        skip_existing=config.skip_existing,
        provenance=config.provenance(),
    )
