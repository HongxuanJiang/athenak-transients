"""Tabulated LTE equation-of-state reader: temperature from (rho, e_int).

``TabulatedLteTable`` is the native AthenaK table reader/interpolator taken
from ``plot_slice.py`` (only the parts needed here: the binary table format,
bilinear field evaluation in (ln rho, ln T), and the bisection that inverts the
``logeps`` field for the temperature).  The interpolation and inversion are
unchanged from the validated version.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .constants import HYDROGEN_MASS_CGS, K_B_CGS, SAFE_POSITIVE


class TabulatedLteTable:
    """Native AthenaK table reader/interpolator for tabulated LTE EOS tables."""

    required_fields = ("logpress", "logeps", "logcs2", "gamma1", "gamma3m1", "xion")

    def __init__(
        self, table_path, density_unit_cgs, pressure_unit_cgs, velocity_unit_cgs
    ):
        self.table_path = Path(table_path)
        metadata, point_info, fields = self._read_native_table(self.table_path)

        if list(point_info.keys()) != ["logrho", "logtemp"]:
            raise RuntimeError(
                "Tabulated LTE table axes must be ordered as logrho, logtemp."
            )
        for name in self.required_fields:
            if name not in fields:
                raise RuntimeError(
                    f'Tabulated LTE table is missing required field "{name}".'
                )

        self.metadata = metadata
        log_axis_base = metadata.get("log_axis_base", "e")
        if log_axis_base not in ("", "e"):
            raise RuntimeError(
                f"Only native tables with log_axis_base = e are supported, "
                f'got "{log_axis_base}".'
            )
        self.table_type = metadata.get("table_type", "")
        self.eos_name = metadata.get("eos_name", "")
        self.h2_enabled = metadata.get("h2_enabled", "off").lower() == "on"
        self.radiation_enabled = metadata.get("radiation_pressure", "off").lower() == "on"
        self.logrho = point_info["logrho"]
        self.logtemp = point_info["logtemp"]
        self.fields = fields
        self.nrho = self.logrho.size
        self.ntemp = self.logtemp.size
        self.inv_dlogrho = 1.0 / self._uniform_spacing(self.logrho, "logrho")
        self.inv_dlogtemp = 1.0 / self._uniform_spacing(self.logtemp, "logtemp")

        self.density_unit_cgs = density_unit_cgs
        self.pressure_unit_cgs = pressure_unit_cgs
        self.velocity_unit_cgs = velocity_unit_cgs
        self.specific_eint_unit_cgs = pressure_unit_cgs / density_unit_cgs
        self.temp_unit_cgs = HYDROGEN_MASS_CGS * velocity_unit_cgs**2 / K_B_CGS
        self.tmin_code = math.exp(self.logtemp[0]) / self.temp_unit_cgs
        self.tmax_code = math.exp(self.logtemp[-1]) / self.temp_unit_cgs

    @classmethod
    def from_units(cls, table_path, units):
        """Build from a :class:`athenak_rt.snapshot.UnitSystem`."""
        return cls(table_path, units.density_cgs, units.pressure_cgs, units.velocity_cgs)

    # ------------------------------------------------------------------ I/O
    @staticmethod
    def _read_header_block(fp, name):
        begin = f"<{name}begin>"
        end = f"<{name}end>"
        line = fp.readline().decode("ascii").strip()
        if line != begin:
            raise RuntimeError(f'Table header is missing block "{name}".')

        lines = []
        while True:
            line = fp.readline().decode("ascii")
            if not line:
                raise RuntimeError(f'Unexpected EOF while reading "{name}" block.')
            line = line.strip()
            if line == end:
                return lines
            lines.append(line)

    @classmethod
    def _read_native_table(cls, path):
        with open(path, "rb") as fp:
            metadata_lines = cls._read_header_block(fp, "metadata")
            scalar_lines = cls._read_header_block(fp, "scalars")
            point_lines = cls._read_header_block(fp, "points")
            field_lines = cls._read_header_block(fp, "fields")

            metadata = cls._parse_key_value_block(metadata_lines)
            scalars = cls._parse_key_value_block(scalar_lines, cast=float)
            point_counts = cls._parse_key_value_block(point_lines, cast=int)
            field_names = [line.strip() for line in field_lines if line.strip()]

            endianness = metadata.get("endianness", "little")
            if endianness not in ("little", "big"):
                raise RuntimeError(f'Unsupported table endianness "{endianness}".')
            dtype = np.dtype("<f8" if endianness == "little" else ">f8")
            raw = np.frombuffer(fp.read(), dtype=dtype)

        offset = 0
        point_arrays = {}
        for name, npts in point_counts.items():
            point_arrays[name] = np.array(
                raw[offset : offset + npts], dtype=np.float64, copy=True
            )
            offset += npts

        npoints = 1
        for npts in point_counts.values():
            npoints *= npts

        fields = {}
        shape = tuple(point_counts.values())
        for name in field_names:
            values = raw[offset : offset + npoints]
            if values.size != npoints:
                raise RuntimeError(f'Table field "{name}" is truncated.')
            fields[name] = np.array(values, dtype=np.float64, copy=True).reshape(shape)
            offset += npoints

        if offset != raw.size:
            raise RuntimeError("Table contains trailing or incomplete binary data.")

        metadata.update({key: f"{value}" for key, value in scalars.items()})
        return metadata, point_arrays, fields

    @staticmethod
    def _parse_key_value_block(lines, cast=str):
        parsed = {}
        for line in lines:
            if not line:
                continue
            if "=" not in line:
                raise RuntimeError(f'Invalid table header line "{line}".')
            key, value = line.split("=", 1)
            parsed[key.strip()] = cast(value.strip())
        return parsed

    @staticmethod
    def _uniform_spacing(values, name):
        if values.size < 2:
            raise RuntimeError(
                f'Tabulated LTE axis "{name}" must contain at least 2 points.'
            )
        delta = values[1] - values[0]
        if delta <= 0.0:
            raise RuntimeError(
                f'Tabulated LTE axis "{name}" must be strictly increasing.'
            )
        tol = 1.0e-10 * max(1.0, abs(delta))
        if not np.all(np.abs(np.diff(values) - delta) <= tol):
            raise RuntimeError(f'Tabulated LTE axis "{name}" must be uniformly spaced.')
        return delta

    # ---------------------------------------------------------- interpolation
    @staticmethod
    def _safe_positive(values):
        return np.maximum(np.asarray(values, dtype=np.float64), SAFE_POSITIVE)

    def _rho_weights(self, log_rho):
        log_rho_clamped = np.clip(log_rho, self.logrho[0], self.logrho[-1])
        ir = ((log_rho_clamped - self.logrho[0]) * self.inv_dlogrho).astype(np.int64)
        ir = np.clip(ir, 0, self.nrho - 2)
        wr1 = (log_rho_clamped - self.logrho[ir]) * self.inv_dlogrho
        wr0 = 1.0 - wr1
        return ir, wr0, wr1

    def _temp_weights(self, log_temp):
        log_temp_clamped = np.clip(log_temp, self.logtemp[0], self.logtemp[-1])
        it = ((log_temp_clamped - self.logtemp[0]) * self.inv_dlogtemp).astype(np.int64)
        it = np.clip(it, 0, self.ntemp - 2)
        wt1 = (log_temp_clamped - self.logtemp[it]) * self.inv_dlogtemp
        wt0 = 1.0 - wt1
        return it, wt0, wt1

    def _field_at_temp_index_from_weights(self, field_name, ir, wr0, wr1, itemp):
        field = self.fields[field_name]
        return wr0 * field[ir, itemp] + wr1 * field[ir + 1, itemp]

    def _eval_field(self, field_name, log_rho, log_temp):
        ir, wr0, wr1 = self._rho_weights(log_rho)
        it, wt0, wt1 = self._temp_weights(log_temp)
        field = self.fields[field_name]
        return wr0 * (wt0 * field[ir, it] + wt1 * field[ir, it + 1]) + wr1 * (
            wt0 * field[ir + 1, it] + wt1 * field[ir + 1, it + 1]
        )

    def _temperature_from_monotonic_field(self, field_name, rho_cgs, target_log):
        log_rho = np.log(self._safe_positive(rho_cgs))
        ir, wr0, wr1 = self._rho_weights(log_rho)

        ilo = np.zeros_like(ir)
        ihi = np.full_like(ir, self.ntemp - 1)
        vlo = self._field_at_temp_index_from_weights(field_name, ir, wr0, wr1, ilo)
        vhi = self._field_at_temp_index_from_weights(field_name, ir, wr0, wr1, ihi)
        target = np.clip(target_log, vlo, vhi)

        for _ in range(int(np.ceil(np.log2(self.ntemp))) + 1):
            imid = (ilo + ihi) // 2
            vmid = self._field_at_temp_index_from_weights(field_name, ir, wr0, wr1, imid)
            use_hi = target <= vmid
            ihi = np.where(use_hi, imid, ihi)
            vhi = np.where(use_hi, vmid, vhi)
            ilo = np.where(use_hi, ilo, imid)
            vlo = np.where(use_hi, vlo, vmid)

        log_t_lo = self.logtemp[ilo]
        log_t_hi = self.logtemp[ihi]
        denom = vhi - vlo
        frac = np.zeros_like(target)
        np.divide(target - vlo, denom, out=frac, where=np.abs(denom) > 0.0)
        log_t = log_t_lo + frac * (log_t_hi - log_t_lo)
        t_code = np.exp(log_t) / self.temp_unit_cgs
        return np.clip(t_code, self.tmin_code, self.tmax_code)

    # ---------------------------------------------------------------- public
    def temperature_from_rho_eint(self, dens_code, eint_code):
        """Temperature [code units, T_unit = m_H v_unit^2 / k_B] from rho and e_int."""
        dens_safe = self._safe_positive(dens_code)
        rho_cgs = dens_safe * self.density_unit_cgs
        eps_cgs = self._safe_positive(eint_code / dens_safe) * self.specific_eint_unit_cgs
        return self._temperature_from_monotonic_field("logeps", rho_cgs, np.log(eps_cgs))

    def pressure_from_rho_t(self, dens_code, temp_code):
        rho_cgs = self._safe_positive(dens_code) * self.density_unit_cgs
        temp_cgs = np.clip(temp_code, self.tmin_code, self.tmax_code) * self.temp_unit_cgs
        log_p = self._eval_field("logpress", np.log(rho_cgs), np.log(temp_cgs))
        return np.exp(log_p) / self.pressure_unit_cgs

    def population_table(self):
        """Ionization populations for the continuum transfer kernels.

        Returns ``(lr0, dlr, lt0, dlt, lnf, nh_per_rho, nhe_per_rho)`` with
        ``lnf[5, nrho, ntemp]`` the natural log of the (H I, H II, He I, He II,
        He III) fractions (H I = 1 - xion - xh2; clipped to [1e-300, 1]) and
        ``n_H = nh_per_rho * rho``, ``n_He = nhe_per_rho * rho``.

        Raises ``RuntimeError`` (message says why) if the table cannot provide
        consistent H/He populations, e.g. hydrogen-only Saha tables that carry
        no ``xh2``/``xhe1``/``xhe2`` fields or the scalars ``x_h, y_he, m_h, m_he``.
        """
        missing = [n for n in ("xion", "xh2", "xhe1", "xhe2") if n not in self.fields]
        if missing:
            raise RuntimeError(
                f"EOS table {self.table_path.name} has no {', '.join(missing)} "
                "field(s) (hydrogen-only or Saha-type table)."
            )
        try:
            x_h, y_he, m_h, m_he = (
                float(self.metadata[k]) for k in ("x_h", "y_he", "m_h", "m_he")
            )
        except KeyError as exc:
            raise RuntimeError(
                f"EOS table {self.table_path.name} lacks the <scalars> entry {exc}."
            ) from exc
        if y_he > 0.0 and not np.any(self.fields["xhe1"] + self.fields["xhe2"] > 0.0):
            raise RuntimeError(
                f"EOS table {self.table_path.name} has y_he > 0 but no helium "
                "ionization (xhe1 = xhe2 = 0 everywhere)."
            )
        xion, xh2 = self.fields["xion"], self.fields["xh2"]
        xhe1, xhe2 = self.fields["xhe1"], self.fields["xhe2"]
        frac = np.stack([1.0 - xion - xh2, xion, 1.0 - xhe1 - xhe2, xhe1, xhe2])
        lnf = np.ascontiguousarray(np.log(np.clip(frac, 1.0e-300, 1.0)), dtype=np.float64)
        return (
            float(self.logrho[0]),
            float(self.logrho[1] - self.logrho[0]),
            float(self.logtemp[0]),
            float(self.logtemp[1] - self.logtemp[0]),
            lnf,
            x_h / m_h,
            y_he / m_he,
        )

    def pressure_from_rho_eint(self, dens_code, eint_code):
        temp_code = self.temperature_from_rho_eint(dens_code, eint_code)
        return self.pressure_from_rho_t(dens_code, temp_code)


def resolve_eos_table(
    explicit: Optional[os.PathLike],
    header_path: Optional[str],
    search_dirs: Iterable[os.PathLike] = (),
) -> Path:
    """Locate the EOS table.

    Order: an explicit ``--eos-table``; the path recorded in the snapshot header
    (``<hydro>/table``) if it exists on this machine; ``$ATHENAK_EOS_TABLE_DIR``
    or any ``search_dirs`` joined with the header path's basename.
    """
    tried = []
    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise RuntimeError(f"EOS table not found: {candidate}")
    if header_path:
        raw = Path(header_path).expanduser()
        candidates = [raw]
        env_dir = os.environ.get("ATHENAK_EOS_TABLE_DIR")
        if env_dir:
            candidates.append(Path(env_dir).expanduser() / raw.name)
        candidates.extend(Path(d).expanduser() / raw.name for d in search_dirs)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
            tried.append(str(candidate))
    raise RuntimeError(
        "Could not locate the EOS table; pass --eos-table explicitly. Tried:\n  "
        + "\n  ".join(tried or ["(snapshot header records no <hydro>/table)"])
    )
