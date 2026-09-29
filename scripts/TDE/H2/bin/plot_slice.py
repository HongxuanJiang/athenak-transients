#! /usr/bin/env python3

"""
Script for plotting a 2D slice from a 2D or 3D AthenaK data dump.

Usage:
[python3] plot_slice.py <input_file> <quantity_to_plot> <output_file> [options]

Example:
~/athenak/vis/python/plot_slice.py basename.prim.00100.bin dens image.png

<input_file> can be any standard AthenaK .bin data dump. <output_file> can have
any extension recognized by Matplotlib (e.g., .png). If <output_file> is simply
"show", the script will open a live Matplotlib viewing window.

Available quantities include anything found in the input file (e.g., dens, velx,
eint, bcc1, r00_ff). If an invalid quantity is requested (e.g.,
"plot_slice.py <input_file> ? show"), the error message will list all available
quantities in the file.

Additional derived quantities can be computed as well. These are identified by
prefixing with "derived:". An invalid request (e.g.,
"plot_slice.py <input_file> derived:? show") will list available options.
Currently, these include the following:
  - Gas pressure and related quantities:
    - pgas: gas pressure
    - pgas_rho: pgas / rho
    - entropy: entropy proxy K = pgas / rho^gamma (gamma-law EOS only)
    - c_s: adiabatic sound speed
    - T: temperature in K
    - xh2: molecular-hydrogen fraction, 2 n_H2 / n_H,tot
    - xion: hydrogen ionization fraction
    - xhe1: singly ionized helium fraction
    - xhe2: doubly ionized helium fraction
    - gamma1: first adiabatic exponent
    - gamma3m1: Gamma3 - 1
    - mu: mean molecular weight
    - beta_rad: radiation-pressure fraction p_rad / (p_gas + p_rad)
    - prad_pgas: (fluid-frame radiation pressure) / (gas pressure)
  - Simple transformed quantities:
    - grav_phi_abs: absolute gravity potential |grav_phi|
  - Non-relativistic velocity:
    - vel_xyz: three-panel plot of velx, vely, velz
    - vel_norm: one-panel speed = (velx^2 + vely^2 + velz^2)^(1/2)
    - mach: Mach number = speed / c_s
    - div_v: block-local velocity divergence nabla . v
    - div_v_cs: dimensionless velocity divergence normalized by tidal radius over c_s
    - div_v_panels: plot of div_v; with
      --orthogonal_triptych, this becomes xoy/yoz panels
    - vr_nr, vth_nr, vph_nr: orthonormal spherical components v^i
  - Relativistic velocity:
    - uut: normal-frame Lorentz factor u^{t'} = tilde{u}^t
    - ut, ux, uy, uz: contravariant CKS 4-velocity components u^mu
    - ur, uth, uph: contravariant SKS 4-velocity components u^i
    - u_t, u_x, u_y, u_z: covariant CKS 4-velocity components u_mu
    - u_r, u_th, u_ph: covariant SKS 4-velocity components u_i
    - vx, vy, vz: CKS 3-velocity components v^i = u^i / u^t
    - vr_rel, vth_rel, vph_rel: SKS 3-velocity components v^i = u^i / u^t
  - Non-relativistic magnetic field and related quantities:
    - Br_nr, Bth_nr, Bph_nr: orthonormal spherical components B^i
    - pmag_nr: magnetic pressure, pmag = B^2 / 2
    - beta_inv_nr: reciprocal of plasma beta, beta^{-1} = pmag / pgas
    - sigma_nr: plasma sigma, sigma = B^2 / rho
  - Relativistic magnetic field and related quantities:
    - bt, bx, by, bz: contravariant CKS 4-field components b^mu
    - br, bth, bph: contravariant SKS 4-field components b^i
    - b_t, b_x, b_y, b_z: covariant CKS 4-field components b_mu
    - b_r, b_th, b_ph: covariant SKS 4-field components b_i
    - Br_rel, Bth_rel, Bph_rel: SKS 3-field components B^i = *F^{it}
    - pmag_rel: fluid-frame magnetic pressure, pmag = (B^2 - E^2) / 2 = b_mu b^mu / 2
    - beta_inv_rel: reciprocal of plasma beta, beta^{-1} = pmag / pgas
    - sigma_rel: cold plasma sigma, sigma = 2 pmag / rho
    - sigmah_rel: hot plasma sigma, sigma_hot = 2 pmag / (rho + ugas + pgas)
    - va_rel: Alfven speed, v_A = (2 pmag / (rho + ugas + pgas + 2 * pmag))^(1/2)
    - pmag_prad: (fluid-frame magnetic pressure) / (fluid-frame radiation pressure)
  - Relativistic radiation quantities:
    - prad: (fluid-frame radiation pressure) = (fluid-frame radiation energy density) / 3
    - Rtr, Rtth, Rtph: contravariant SKS components of radiation flux
    - Rrr, Rthth, Rphph: contravariant SKS components of radiation pressure
    - Rrth, Rrph, Rthph: contravariant SKS components of radiation shear
    - Rtx_Rtt, Rty_Rtt, Rtz_Rtt: contravariant CKS components of Eddington flux
    - Rxx_Rtt, Ryy_Rtt, Rzz_Rtt: contravariant CKS components of Eddington pressure
    - Rxy_Rtt, Rxz_Rtt, Ryz_Rtt: contravariant CKS components of Eddington shear
    - Rtr_Rtt, Rtth_Rtt, Rtph_Rtt: contravariant SKS components of Eddington flux
    - Rrr_Rtt, Rthth_Rtt, Rphph_Rtt: contravariant SKS components of Eddington pressure
    - Rrth_Rtt, Rrph_Rtt, Rthph_Rtt: contravariant SKS components of Eddington shear
    - R01_R00_ff, R02_R00_ff, R03_R00_ff: fluid-frame components of Eddington flux
    - R11_R00_ff, R22_R00_ff, R33_R00_ff: fluid-frame components of Eddington pressure
    - R12_R00_ff, R13_R00_ff, R23_R00_ff: fluid-frame components of Eddington shear
  - Relativistic opacity quantities:
    - kappa_a, kappa_s, kappa_t: absorption, scattering, and total opacities in cm^2/g
    - alpha_a, alpha_s, alpha_t: corresponding absorption coefficients in cm^-1
    - tau_a, tau_s, tau_t: corresponding optical depths per gravitational radius
  - Relativistic enthalpy densities and Bernoulli parameters:
    - wgas: hydrodynamic enthalpy rho + ugas + pgas
    - wmhd: magnetohydrodynamic enthalpy rho + ugas + pgas + 2 * pmag
    - wgasrad: radiation-hydrodynamic enthalpy rho + ugas + pgas + 4 * prad
    - wmhdrad:
        radiation-magnetohydrodynamic enthalpy rho + ugas + pgas + 2 * pmag + 4 * prad
    - Begas: -u_t * wgas / rho - 1
    - Bemhd: -u_t * wmhd / rho - 1
    - Begasrad: -u_t * wgasrad / rho - 1
    - Bemhdrad: -u_t * wmhdrad / rho - 1
  - Non-relativistic conserved quantities
    - cons_hydro_nr_t: pure hydrodynamical energy density
    - cons_hydro_nr_x, cons_hydro_nr_y, cons_hydro_nr_z: momentum density
    - cons_em_nr_t: pure electromagnetic energy density
    - cons_mhd_nr_t: MHD energy density
    - cons_mhd_nr_x, cons_mhd_nr_y, cons_mhd_nr_z: MHD momentum density
  - Relativistic conserved quantities
    - cons_hydro_rel_t, : (T_hydro)^t_t
    - cons_hydro_rel_x, cons_hydro_rel_y, cons_hydro_rel_z: (T_hydro)^t_i
    - cons_em_rel_t, : (T_EM)^t_t
    - cons_em_rel_x, cons_em_rel_y, cons_em_rel_z: (T_EM)^t_i
    - cons_mhd_rel_t, : (T_MHD)^t_t
    - cons_mhd_rel_x, cons_mhd_rel_y, cons_mhd_rel_z: (T_MHD)^t_i

The following quantities are in physical units: T (K), kappa_{a,s,t} (cm g^-1), and
alpha_{a,s,t} (cm^-1). All others are in code units. For raw density plots, the image
range remains in code units and the colorbar shows a secondary cgs scale when unit
metadata are available.

Optional inputs include:
  -d: direction orthogonal to slice of 3D data
  -l: location of slice of 3D data if not 0
  --r_max: half-width of plot in both coordinates, centered at the origin
  --x1_min, --x1_max, --x2_min, --x2_max: horizontal and vertical limits of plot
  -c: colormap recognized by Matplotlib
  -n: colormap normalization (e.g., "-n log") if not linear
  --vmin, --vmax: limits of colorbar if not the full range of data
  --grid: flag for outlining domain decomposition
  --streamlines: flag for overplotting in-plane velocity streamlines on single-panel plots
  --stream_color, --stream_alpha: streamline style choices
  --stream_density, --stream_arrowsize, --stream_resolution: streamline tuning
  --bound_unbound_contour: overplot the E=0 contour separating bound/unbound gas
  --bound_unbound_color, --bound_unbound_alpha, --bound_unbound_linewidth:
      style choices for the bound/unbound contour
  --bh_mask: enable the BH excision-mask overlay read from the dump metadata
  --bh_mask_min_pixels: minimum displayed mask radius for wide views; set 0 for
      the exact physical radius only
  --bh_marker: mark the live BH position with a fixed screen-space point in addition
      to the physical excision mask
  --horizon: flag for outlining outer event horizon of GR simulation
  --horizon_mask: flag for covering black hole of GR simulation
  --ergosphere: flag for outlining boundary of ergosphere in GR simulation
  --horizon_color, --horizon_mask_color, --ergosphere_color: color choices
  --notex: flag to disable Latex typesetting of labels
  --dpi: image resolution

Run "plot_slice.py -h" to see a full description of inputs.
"""

# Python standard modules
import argparse
import collections
import math
from pathlib import Path
import struct
import warnings

# Numerical modules
import numpy as np

# Load plotting modules
import matplotlib


K_B_CGS = 1.380649e-16
HYDROGEN_MASS_CGS = 1.6735575e-24
SAFE_POSITIVE = 1.0e-300
CHABRIER_PLOT_CHEM_RHO_MAX_CGS = 1.0e-1
CHABRIER_PLOT_CHEM_PROXY_TABLE = 'lte_t13_prad_eos.table'

TABULATED_LTE_EOS_NAMES = (
    'saha_table',
    'lte_table_hhe',
    'lte_table_hhe_prad',
    'lte_table_t13',
    'lte_table_t13_prad',
    'lte_table_hybrid_hhe_t13',
    'lte_table_hybrid_hhe_t13_prad',
    'lte_table_scvh_t13_union',
    'lte_table_scvh_t13_union_prad',
    'lte_table_scvh_t13_helm_union',
    'lte_table_scvh_t13_helm_union_prad',
    'lte_table_scvh_t13_cp_union',
    'lte_table_scvh_t13_cp_union_prad',
    'lte_table_scvh_t13_cp_helm_union',
    'lte_table_scvh_t13_cp_helm_union_prad',
    'lte_table_chabrier2021_t13_helm_union',
    'lte_table_chabrier2021_t13_helm_union_prad',
)


def get_density_unit_cgs_from_input(input_data):
    if 'units' in input_data and 'mass_cgs' in input_data['units'] and 'length_cgs' in input_data['units']:
        return (
            float(input_data['units']['mass_cgs']) /
            float(input_data['units']['length_cgs']) ** 3
        )
    if 'units' in input_data and 'density_cgs' in input_data['units']:
        return float(input_data['units']['density_cgs'])
    return None


class TabulatedLteTable:
    """Native AthenaK table reader/interpolator for AthenaK tabulated LTE EOS tables."""

    required_fields = ('logpress', 'logeps', 'logcs2', 'gamma1', 'gamma3m1', 'xion')

    def __init__(self, table_path, density_unit_cgs, pressure_unit_cgs, velocity_unit_cgs):
        self.table_path = Path(table_path)
        metadata, point_info, fields = self._read_native_table(self.table_path)

        if list(point_info.keys()) != ['logrho', 'logtemp']:
            raise RuntimeError('Tabulated LTE table axes must be ordered as logrho, logtemp.')
        for name in self.required_fields:
            if name not in fields:
                raise RuntimeError(f'Tabulated LTE table is missing required field "{name}".')

        self.metadata = metadata
        log_axis_base = metadata.get('log_axis_base', 'e')
        if log_axis_base not in ('', 'e'):
            raise RuntimeError(
                f'Tabulated LTE plotting only supports native tables with '
                f'log_axis_base = e, got "{log_axis_base}".'
            )
        self.table_type = metadata.get('table_type', '')
        self.eos_name = metadata.get('eos_name', '')
        self.h2_enabled = metadata.get('h2_enabled', 'off').lower() == 'on'
        self.radiation_enabled = metadata.get('radiation_pressure', 'off').lower() == 'on'
        self.logrho = point_info['logrho']
        self.logtemp = point_info['logtemp']
        self.fields = fields
        self.nrho = self.logrho.size
        self.ntemp = self.logtemp.size
        self.inv_dlogrho = 1.0 / self._uniform_spacing(self.logrho, 'logrho')
        self.inv_dlogtemp = 1.0 / self._uniform_spacing(self.logtemp, 'logtemp')

        self.density_unit_cgs = density_unit_cgs
        self.pressure_unit_cgs = pressure_unit_cgs
        self.velocity_unit_cgs = velocity_unit_cgs
        self.specific_eint_unit_cgs = pressure_unit_cgs / density_unit_cgs
        self.temp_unit_cgs = HYDROGEN_MASS_CGS * velocity_unit_cgs**2 / K_B_CGS
        self.tmin_code = math.exp(self.logtemp[0]) / self.temp_unit_cgs
        self.tmax_code = math.exp(self.logtemp[-1]) / self.temp_unit_cgs
        self.is_h_only_mode = self.table_type == 'saha_hydrogen_lte'
        self.is_chabrier_union_mode = self.table_type in (
            'lte_chabrier2021_t13_helm_union',
            'lte_chabrier2021_t13_helm_union_prad',
        )
        self.plot_t13_chem_proxy = None
        if self.is_chabrier_union_mode:
            self.plot_t13_chem_proxy = self._build_plot_t13_chem_proxy()

    @classmethod
    def from_input_data(cls, input_data, data_file, variable_names):
        if 'units' not in input_data:
            raise RuntimeError('Tabulated LTE plotting requires a [units] block in the dump '
                               'metadata.')

        units = input_data['units']
        length_cgs = float(units['length_cgs'])
        mass_cgs = float(units['mass_cgs'])
        time_cgs = float(units['time_cgs'])
        density_unit_cgs = mass_cgs / length_cgs**3
        pressure_unit_cgs = mass_cgs / (length_cgs * time_cgs**2)
        velocity_unit_cgs = length_cgs / time_cgs
        block_name, block_data = cls._select_active_fluid_block(input_data, variable_names)
        if 'table' not in block_data:
            raise RuntimeError(
                f'Tabulated LTE plotting requires <{block_name}>/table in the dump metadata.'
            )
        table_path = cls._resolve_table_path(block_data['table'], data_file)
        return cls(table_path, density_unit_cgs, pressure_unit_cgs, velocity_unit_cgs)

    @staticmethod
    def _resolve_table_path(raw_path, data_file):
        raw = Path(raw_path).expanduser()
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend([
                Path.cwd() / raw,
                Path(data_file).resolve().parent / raw,
                Path(__file__).resolve().parent.parent / raw,
            ])
            for parent in Path(__file__).resolve().parents:
                candidates.append(parent / 'eos_tables' / raw.name)

        tried = []
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in tried:
                continue
            tried.append(resolved)
            if resolved.exists():
                return resolved

        tried_str = '\n'.join(str(path) for path in tried)
        raise RuntimeError(
            f'Unable to locate tabulated LTE EOS table "{raw_path}". Tried:\n{tried_str}'
        )

    def _build_plot_t13_chem_proxy(self):
        proxy_path = (self.table_path.parent / CHABRIER_PLOT_CHEM_PROXY_TABLE).resolve()
        if not proxy_path.exists():
            raise RuntimeError(
                f'Chabrier plotting chemistry override requires "{CHABRIER_PLOT_CHEM_PROXY_TABLE}" '
                f'next to "{self.table_path}", but it was not found.'
            )
        return type(self)(
            proxy_path,
            self.density_unit_cgs,
            self.pressure_unit_cgs,
            self.velocity_unit_cgs,
        )

    @staticmethod
    def _select_active_fluid_block(input_data, variable_names):
        """Resolve the active thermodynamic block for a dump.

        The helper must distinguish hydro and MHD dumps symmetrically because the active
        tabulated-EOS table can live under either <hydro>/table or <mhd>/table.
        """
        has_hydro = 'hydro' in input_data and 'eos' in input_data['hydro']
        has_mhd = 'mhd' in input_data and 'eos' in input_data['mhd']
        if not has_hydro and not has_mhd:
            raise RuntimeError('Tabulated LTE plotting requires a <hydro> or <mhd> block '
                               'in the dump metadata.')

        names = set(variable_names)
        looks_mhd = any(name in names for name in ('bcc1', 'bcc2', 'bcc3'))

        if has_hydro and not has_mhd:
            return 'hydro', input_data['hydro']
        if has_mhd and not has_hydro:
            return 'mhd', input_data['mhd']
        if looks_mhd:
            return 'mhd', input_data['mhd']

        raise RuntimeError(
            'Could not determine whether this dump should use <hydro>/table or <mhd>/table. '
            'The dump metadata contains both blocks, but the variable list is not decisive.'
        )

    @classmethod
    def resolve_active_eos_name(cls, input_data, variable_names):
        _, block_data = cls._select_active_fluid_block(input_data, variable_names)
        return block_data.get('eos', '')

    @staticmethod
    def _read_header_block(fp, name):
        begin = f'<{name}begin>'
        end = f'<{name}end>'
        line = fp.readline().decode('ascii').strip()
        if line != begin:
            raise RuntimeError(f'Table header is missing block "{name}".')

        lines = []
        while True:
            line = fp.readline().decode('ascii')
            if not line:
                raise RuntimeError(f'Unexpected EOF while reading "{name}" block.')
            line = line.strip()
            if line == end:
                return lines
            lines.append(line)

    @classmethod
    def _read_native_table(cls, path):
        with open(path, 'rb') as fp:
            metadata_lines = cls._read_header_block(fp, 'metadata')
            scalar_lines = cls._read_header_block(fp, 'scalars')
            point_lines = cls._read_header_block(fp, 'points')
            field_lines = cls._read_header_block(fp, 'fields')

            metadata = cls._parse_key_value_block(metadata_lines)
            scalars = cls._parse_key_value_block(scalar_lines, cast=float)
            point_counts = cls._parse_key_value_block(point_lines, cast=int)
            field_names = [line.strip() for line in field_lines if line.strip()]

            endianness = metadata.get('endianness', 'little')
            if endianness not in ('little', 'big'):
                raise RuntimeError(f'Unsupported table endianness "{endianness}".')
            dtype = np.dtype('<f8' if endianness == 'little' else '>f8')
            raw = np.frombuffer(fp.read(), dtype=dtype)

        offset = 0
        point_arrays = {}
        for name, npts in point_counts.items():
            point_arrays[name] = np.array(raw[offset:offset + npts], dtype=np.float64, copy=True)
            offset += npts

        npoints = 1
        for npts in point_counts.values():
            npoints *= npts

        fields = {}
        shape = tuple(point_counts.values())
        for name in field_names:
            values = raw[offset:offset + npoints]
            if values.size != npoints:
                raise RuntimeError(f'Table field "{name}" is truncated.')
            fields[name] = np.array(values, dtype=np.float64, copy=True).reshape(shape)
            offset += npoints

        if offset != raw.size:
            raise RuntimeError('Table contains trailing or incomplete binary data.')

        metadata.update({key: f'{value}' for key, value in scalars.items()})
        return metadata, point_arrays, fields

    @staticmethod
    def _parse_key_value_block(lines, cast=str):
        parsed = {}
        for line in lines:
            if not line:
                continue
            if '=' not in line:
                raise RuntimeError(f'Invalid table header line "{line}".')
            key, value = line.split('=', 1)
            parsed[key.strip()] = cast(value.strip())
        return parsed

    @staticmethod
    def _uniform_spacing(values, name):
        if values.size < 2:
            raise RuntimeError(f'Tabulated LTE axis "{name}" must contain at least 2 points.')
        delta = values[1] - values[0]
        if delta <= 0.0:
            raise RuntimeError(f'Tabulated LTE axis "{name}" must be strictly increasing.')
        tol = 1.0e-10 * max(1.0, abs(delta))
        if not np.all(np.abs(np.diff(values) - delta) <= tol):
            raise RuntimeError(f'Tabulated LTE axis "{name}" must be uniformly spaced.')
        return delta

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
        return (
            wr0 * (wt0 * field[ir, it] + wt1 * field[ir, it + 1])
            + wr1 * (wt0 * field[ir + 1, it] + wt1 * field[ir + 1, it + 1])
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

    def pressure_from_rho_eint(self, dens_code, eint_code):
        temp_code = self.temperature_from_rho_eint(dens_code, eint_code)
        return self.pressure_from_rho_t(dens_code, temp_code)

    def pressure_from_rho_t(self, dens_code, temp_code):
        rho_cgs = self._safe_positive(dens_code) * self.density_unit_cgs
        temp_cgs = np.clip(temp_code, self.tmin_code, self.tmax_code) * self.temp_unit_cgs
        log_p = self._eval_field('logpress', np.log(rho_cgs), np.log(temp_cgs))
        return np.exp(log_p) / self.pressure_unit_cgs

    def temperature_from_rho_eint(self, dens_code, eint_code):
        dens_safe = self._safe_positive(dens_code)
        rho_cgs = dens_safe * self.density_unit_cgs
        eps_cgs = self._safe_positive(eint_code / dens_safe) * self.specific_eint_unit_cgs
        return self._temperature_from_monotonic_field('logeps', rho_cgs, np.log(eps_cgs))

    def thermo_fields_from_rho_eint(self, dens_code, eint_code, override_plot_chemistry=False):
        # Reconstruction policy:
        #   1. Production LTE diagnostics should come from the active table.
        #   2. The only exact synthetic fallback kept here is the pure-H saha_table case,
        #      where xh2 = xhe1 = xhe2 = beta_rad = 0 and mu = 1 / (1 + xion).
        #   3. Plotting may optionally override chemistry-style diagnostics for the
        #      Chabrier union: use T13 below CHABRIER_PLOT_CHEM_RHO_MAX_CGS and NaN above it.
        #   4. Any other missing diagnostic field is treated as a hard error.
        dens_safe = self._safe_positive(dens_code)
        rho_cgs = dens_safe * self.density_unit_cgs
        temp_code = self.temperature_from_rho_eint(dens_safe, eint_code)
        temp_cgs = temp_code * self.temp_unit_cgs
        log_rho = np.log(rho_cgs)
        log_temp = np.log(temp_cgs)
        xion = self._eval_field('xion', log_rho, log_temp)
        gamma1 = self._eval_field('gamma1', log_rho, log_temp)
        gamma3m1 = self._eval_field('gamma3m1', log_rho, log_temp)
        xh2 = self._load_or_exact_h_only_field('xh2', log_rho, log_temp, xion)
        xhe1 = self._load_or_exact_h_only_field('xhe1', log_rho, log_temp, xion)
        xhe2 = self._load_or_exact_h_only_field('xhe2', log_rho, log_temp, xion)
        mu = self._load_or_exact_h_only_field('mu', log_rho, log_temp, xion)
        beta_rad = self._load_or_exact_h_only_field('beta_rad', log_rho, log_temp, xion)
        beta_rad = np.clip(beta_rad, 0.0, 1.0 - 1.0e-12)
        if override_plot_chemistry and self.plot_t13_chem_proxy is not None:
            chemistry_valid = rho_cgs < CHABRIER_PLOT_CHEM_RHO_MAX_CGS
            (_, _, proxy_xion, proxy_xhe1, proxy_xhe2, _, _, proxy_mu, _) = (
                self.plot_t13_chem_proxy.thermo_fields_from_rho_eint(dens_safe, eint_code)
            )
            nan_field = np.full_like(xion, np.nan, dtype=np.float64)
            xion = np.where(chemistry_valid, proxy_xion, nan_field)
            xhe1 = np.where(chemistry_valid, proxy_xhe1, nan_field)
            xhe2 = np.where(chemistry_valid, proxy_xhe2, nan_field)
            mu = np.where(chemistry_valid, proxy_mu, nan_field)
        return temp_code, xh2, xion, xhe1, xhe2, gamma1, gamma3m1, mu, beta_rad

    def sound_speed2_from_rho_eint(self, dens_code, eint_code):
        dens_safe = self._safe_positive(dens_code)
        rho_cgs = dens_safe * self.density_unit_cgs
        temp_code = self.temperature_from_rho_eint(dens_safe, eint_code)
        temp_cgs = temp_code * self.temp_unit_cgs
        log_cs2 = self._eval_field('logcs2', np.log(rho_cgs), np.log(temp_cgs))
        return np.exp(log_cs2) / self.specific_eint_unit_cgs

    def prad_pgas_from_rho_eint(self, dens_code, eint_code):
        dens_safe = self._safe_positive(dens_code)
        rho_cgs = dens_safe * self.density_unit_cgs
        temp_code = self.temperature_from_rho_eint(dens_safe, eint_code)
        temp_cgs = temp_code * self.temp_unit_cgs
        beta_rad = self._load_or_exact_h_only_field(
            'beta_rad', np.log(rho_cgs), np.log(temp_cgs), np.zeros_like(rho_cgs)
        )
        beta_rad = np.clip(beta_rad, 0.0, 1.0 - 1.0e-12)
        return beta_rad / np.maximum(1.0 - beta_rad, 1.0e-12)

    def _load_or_exact_h_only_field(self, field_name, log_rho, log_temp, xion):
        if field_name in self.fields:
            return self._eval_field(field_name, log_rho, log_temp)
        if self.is_h_only_mode:
            zeros = np.zeros_like(log_rho)
            if field_name in ('xh2', 'xhe1', 'xhe2', 'beta_rad'):
                return zeros
            if field_name == 'mu':
                return 1.0 / np.maximum(1.0 + xion, 1.0e-12)
        raise RuntimeError(
            f'Requested diagnostic field "{field_name}" is not present in EOS table '
            f'"{self.table_path}". Refusing to fabricate it for EOS mode '
            f'"{self.eos_name or self.table_type}".'
        )


def optional_float(value):
    if value is None or value == '':
        return None
    return float(value)


def read_input_metadata(data_file):
    with open(data_file, 'rb') as f:
        line = f.readline().decode('ascii')
        if line != 'Athena binary output version=1.1\n':
            raise RuntimeError('Unrecognized data file format.')
        next(f)
        next(f)
        next(f)
        line = f.readline().decode('ascii')
        if line[:19] != '  size of location=':
            raise RuntimeError('Could not read location size.')
        line = f.readline().decode('ascii')
        if line[:19] != '  size of variable=':
            raise RuntimeError('Could not read variable size.')
        next(f)
        line = f.readline().decode('ascii')
        if line[:12] != '  variables:':
            raise RuntimeError('Could not read variable names.')
        line = f.readline().decode('ascii')
        if line[:16] != '  header offset=':
            raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

        input_data = {}
        start_of_data = f.tell() + header_offset
        while f.tell() < start_of_data:
            line = f.readline().decode('ascii')
            if line[0] == '#':
                continue
            if line[0] == '<':
                section_name = line[1:-2]
                input_data[section_name] = {}
                continue
            key, val = line.split('=', 1)
            input_data[section_name][key.strip()] = val.split('#', 1)[0].strip()
    return input_data


def parse_live_bh_state(input_data):
    problem_data = input_data.get('problem', {})
    bh_state_valid = str(problem_data.get('bh_live_state_valid', 'false')).lower() \
        in ('true', '1', 'yes', 'on')
    if not bh_state_valid:
        return None
    try:
        bh_inner_boundary = str(problem_data.get('bh_inner_boundary', 'true')).lower() \
            in ('true', '1', 'yes', 'on')
        if 'bh_excise_radius' not in problem_data:
            return None
        bh_radius = float(problem_data['bh_excise_radius'])
        if not bh_inner_boundary:
            bh_radius = 0.0
        bh_radius = max(bh_radius, 0.0)
        return {
            'x': float(problem_data['bh_live_x']),
            'y': float(problem_data['bh_live_y']),
            'z': float(problem_data['bh_live_z']),
            'radius': bh_radius,
        }
    except (KeyError, ValueError):
        return None


_ROOT_BLOCK_COUNT_CACHE = {}


def infer_effective_root_blocks(data_file, input_data):
    mesh = input_data.get('mesh', {})
    cache_key = (
        data_file,
        mesh.get('x1min'), mesh.get('x1max'),
        mesh.get('x2min'), mesh.get('x2max'),
        mesh.get('x3min'), mesh.get('x3max'),
    )
    cached = _ROOT_BLOCK_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        domain_widths = {
            'x1': float(mesh['x1max']) - float(mesh['x1min']),
            'x2': float(mesh['x2max']) - float(mesh['x2min']),
            'x3': float(mesh['x3max']) - float(mesh['x3min']),
        }
    except (KeyError, ValueError):
        _ROOT_BLOCK_COUNT_CACHE[cache_key] = {}
        return {}

    inferred_counts = {
        'x1': collections.Counter(),
        'x2': collections.Counter(),
        'x3': collections.Counter(),
    }

    try:
        with open(data_file, 'rb') as f:
            line = f.readline().decode('ascii')
            if line != 'Athena binary output version=1.1\n':
                raise RuntimeError('Unrecognized data file format.')
            next(f)
            next(f)
            next(f)
            location_size = int(f.readline().decode('ascii').split('=', 1)[1])
            variable_size = int(f.readline().decode('ascii').split('=', 1)[1])
            num_variables = int(f.readline().decode('ascii').split('=', 1)[1])
            next(f)
            header_offset = int(f.readline().decode('ascii').split('=', 1)[1])
            start_of_data = f.tell() + header_offset
            while f.tell() < start_of_data:
                f.readline()

            while True:
                block_indices_data = f.read(24)
                if len(block_indices_data) < 24:
                    break
                block_indices = np.array(struct.unpack('@6i', block_indices_data))
                block_meta = f.read(16)
                if len(block_meta) < 16:
                    break
                _, _, _, block_level = struct.unpack('@4i', block_meta)
                block_lims_data = f.read(6 * location_size)
                if len(block_lims_data) < 6 * location_size:
                    break
                block_lims = struct.unpack('=' + 6 * 'd', block_lims_data)

                axis_widths = {
                    'x1': block_lims[1] - block_lims[0],
                    'x2': block_lims[3] - block_lims[2],
                    'x3': block_lims[5] - block_lims[4],
                }
                for axis, block_width in axis_widths.items():
                    if block_width <= 0.0:
                        continue
                    candidate = domain_widths[axis] / (block_width * 2 ** block_level)
                    rounded = int(round(candidate))
                    if rounded < 1:
                        continue
                    if abs(candidate - rounded) > 1.0e-6:
                        continue
                    inferred_counts[axis][rounded] += 1

                block_nx = block_indices[1] - block_indices[0] + 1
                block_ny = block_indices[3] - block_indices[2] + 1
                block_nz = block_indices[5] - block_indices[4] + 1
                if block_nx <= 0 or block_ny <= 0 or block_nz <= 0:
                    raise RuntimeError('Encountered invalid block extents while probing '
                                       'root-block counts.')
                cells_per_block = block_nx * block_ny * block_nz
                f.seek(num_variables * cells_per_block * variable_size, 1)
    except (OSError, RuntimeError, struct.error, ValueError):
        _ROOT_BLOCK_COUNT_CACHE[cache_key] = {}
        return {}

    inferred = {}
    for axis, counter in inferred_counts.items():
        if counter:
            inferred[axis] = counter.most_common(1)[0][0]
    _ROOT_BLOCK_COUNT_CACHE[cache_key] = inferred
    return inferred


def block_velocity_divergence(velx, vely, velz, block_lims):
    nx = velx.shape[2]
    ny = velx.shape[1]
    nz = velx.shape[0]
    dx = (block_lims[1] - block_lims[0]) / max(nx, 1)
    dy = (block_lims[3] - block_lims[2]) / max(ny, 1)
    dz = (block_lims[5] - block_lims[4]) / max(nz, 1)
    edge_order = 2 if min(nx, ny, nz) > 2 else 1
    dvx_dx = np.gradient(velx, dx, axis=2, edge_order=edge_order)
    dvy_dy = np.gradient(vely, dy, axis=1, edge_order=edge_order)
    dvz_dz = np.gradient(velz, dz, axis=0, edge_order=edge_order)
    return dvx_dx + dvy_dy + dvz_dz


def slice_block_quantity(cell_data, dimension, block_ind):
    if dimension == 'x':
        return cell_data[:, :, block_ind]
    if dimension == 'y':
        return cell_data[:, block_ind, :]
    return cell_data[block_ind, :, :]


# Main function
def main(**kwargs):
    panel_state_only = kwargs.pop('_panel_state_only', False)

    # Load additional numerical modules
    if kwargs['ergosphere']:
        from scipy.optimize import brentq

    # Load additional plotting modules
    if kwargs['output_file'] != 'show':
        matplotlib.use('agg')
    if not kwargs['notex']:
        matplotlib.rc('text', usetex=True)
    import matplotlib.colors as colors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    matplotlib.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })

    # Plotting parameters
    grid_line_style = '-'
    grid_line_width = 0.5
    horizon_line_style = '-'
    horizon_line_width = 1.0
    ergosphere_num_points = 129
    ergosphere_line_style = '-'
    ergosphere_line_width = 1.0
    x1_labelpad = 4.0
    x2_labelpad = 4.0

    def format_colorbar_tick(value):
        if not np.isfinite(value):
            return ''
        if value == 0.0:
            return '0'
        abs_value = abs(value)
        if abs_value >= 1.0e3 or abs_value < 1.0e-3:
            return f'{value:.1e}'
        return f'{value:.3g}'

    def add_density_cgs_colorbar_axis(cbar, plot_variable_name, plot_density_unit_cgs):
        if plot_variable_name != 'dens' or plot_density_unit_cgs is None:
            return
        cbar.set_label('')
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _pos: format_colorbar_tick(value))
        )
        cbar.ax.tick_params(axis='y', which='both', direction='out',
                            labelleft=True, labelright=False, pad=6)

        secax = cbar.ax.secondary_yaxis(
            'right',
            functions=(
                lambda code_value: code_value * plot_density_unit_cgs,
                lambda cgs_value: cgs_value / plot_density_unit_cgs,
            ),
        )
        secax.set_ylabel(r'$\mathrm{g}/\mathrm{cm}^{3}$')
        secax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda value, _pos: format_colorbar_tick(value))
        )
        secax.tick_params(axis='y', which='both', direction='out',
                          labelright=True, pad=6)

    def make_colorbar(fig, mappable, cax, label, plot_variable_name=None,
                      plot_density_unit_cgs=None):
        colorbar_label = '' if plot_variable_name == 'dens' else label
        cbar = fig.colorbar(mappable, cax=cax, label=colorbar_label, extend='both')
        cbar.ax.tick_params(labelsize=15)
        add_density_cgs_colorbar_axis(cbar, plot_variable_name, plot_density_unit_cgs)
        return cbar

    # Adjust user inputs
    if kwargs['dimension'] == '1':
        kwargs['dimension'] = 'x'
    if kwargs['dimension'] == '2':
        kwargs['dimension'] = 'y'
    if kwargs['dimension'] == '3':
        kwargs['dimension'] = 'z'

    if (kwargs['orthogonal_triptych'] and not panel_state_only
            and kwargs['variable'] == 'derived:div_v_panels'):
        triptych_input_data = read_input_metadata(kwargs['data_file'])
        triptych_bh_state = parse_live_bh_state(triptych_input_data)
        if triptych_bh_state is None:
            raise RuntimeError(
                'derived:div_v_panels orthogonal triptych requires live BH metadata.'
            )
        if kwargs['norm'] not in (None, 'linear', 'symlog'):
            raise RuntimeError('"derived:div_v_panels" only supports linear or symlog normalization.')
        if kwargs['norm'] is None:
            kwargs['norm'] = 'symlog'

        bhx = triptych_bh_state['x']
        bhy = triptych_bh_state['y']
        bhz = triptych_bh_state['z']

        explicit_ortho = any(
            kwargs.get(f'ortho_{axis}_{edge}') is not None
            for axis in ('x', 'y', 'z')
            for edge in ('min', 'max')
        )
        if explicit_ortho:
            def ortho_bound(axis_name, edge_name, fallback):
                value = kwargs.get(f'ortho_{axis_name}_{edge_name}')
                return fallback if value is None else value

            ortho_x_min = ortho_bound('x', 'min', kwargs['x1_min'])
            ortho_x_max = ortho_bound('x', 'max', kwargs['x1_max'])
            ortho_y_min = ortho_bound('y', 'min', kwargs['x2_min'])
            ortho_y_max = ortho_bound('y', 'max', kwargs['x2_max'])
            ortho_z_min = ortho_bound('z', 'min', ortho_y_min)
            ortho_z_max = ortho_bound('z', 'max', ortho_y_max)
            if not (ortho_x_max > ortho_x_min and ortho_y_max > ortho_y_min
                    and ortho_z_max > ortho_z_min):
                raise RuntimeError('Invalid orthogonal triptych bounds.')
            plane_specs = (
                {
                    'dimension': 'z',
                    'title': 'xoy',
                    'location': bhz,
                    'x1_min': ortho_x_min,
                    'x1_max': ortho_x_max,
                    'x2_min': ortho_y_min,
                    'x2_max': ortho_y_max,
                },
                {
                    'dimension': 'x',
                    'title': 'yoz',
                    'location': bhx,
                    'x1_min': ortho_y_min,
                    'x1_max': ortho_y_max,
                    'x2_min': ortho_z_min,
                    'x2_max': ortho_z_max,
                },
            )
        else:
            triptych_rmax = kwargs['bh_center_rmax']
            if triptych_rmax is None:
                triptych_rmax = kwargs['r_max'] if kwargs['r_max'] is not None else 50.0
            if triptych_rmax <= 0.0:
                raise RuntimeError('BH-centered orthogonal triptych range must be > 0.')
            plane_specs = (
                {
                    'dimension': 'z',
                    'title': 'xoy',
                    'location': bhz,
                    'x1_min': bhx - triptych_rmax,
                    'x1_max': bhx + triptych_rmax,
                    'x2_min': bhy - triptych_rmax,
                    'x2_max': bhy + triptych_rmax,
                },
                {
                    'dimension': 'x',
                    'title': 'yoz',
                    'location': bhx,
                    'x1_min': bhy - triptych_rmax,
                    'x1_max': bhy + triptych_rmax,
                    'x2_min': bhz - triptych_rmax,
                    'x2_max': bhz + triptych_rmax,
                },
            )

        states = []
        for plane_spec in plane_specs:
            panel_kwargs = dict(kwargs)
            panel_kwargs['variable'] = 'derived:div_v'
            panel_kwargs['dimension'] = plane_spec['dimension']
            panel_kwargs['location'] = plane_spec['location']
            panel_kwargs['orthogonal_triptych'] = False
            panel_kwargs['_panel_state_only'] = True
            panel_kwargs['r_max'] = None
            panel_kwargs['bh_center_rmax'] = None
            panel_kwargs['x1_min'] = plane_spec['x1_min']
            panel_kwargs['x1_max'] = plane_spec['x1_max']
            panel_kwargs['x2_min'] = plane_spec['x2_min']
            panel_kwargs['x2_max'] = plane_spec['x2_max']
            try:
                states.append(main(**panel_kwargs))
            except RuntimeError as exc:
                raise RuntimeError(
                    'Orthogonal div-v plotting requires a 3D dataset with '
                    'resolvable xoy and yoz slices.'
                ) from exc

        def div_triptych_symlog_settings():
            linthresh = kwargs['left_linthresh']
            vmax = kwargs['left_vmax']
            if linthresh is None:
                linthresh = kwargs['linthresh'] if kwargs['linthresh'] is not None else 5.0
            if vmax is None:
                vmax = kwargs['vmax'] if kwargs['vmax'] is not None else 30.0
            return linthresh, vmax

        def symmetric_norm_for_states(states):
            finite_chunks = []
            for state in states:
                values = state['quantity_masked']
                values = values[np.isfinite(values)]
                if values.size > 0:
                    finite_chunks.append(values)
            if finite_chunks:
                finite = np.concatenate(finite_chunks)
                data_vabs = np.nanmax(np.abs(finite))
                if not np.isfinite(data_vabs) or data_vabs == 0.0:
                    data_vabs = 1.0
            else:
                data_vabs = 1.0

            if kwargs['norm'] == 'symlog':
                linthresh, explicit_vmax = div_triptych_symlog_settings()
                vabs = abs(float(explicit_vmax)) if explicit_vmax is not None else data_vabs
                return colors.SymLogNorm(linthresh=linthresh, vmin=-vabs, vmax=vabs), None, None

            if kwargs['vmin'] is None and kwargs['vmax'] is None:
                vmin = -data_vabs
                vmax = data_vabs
            else:
                vmin = -data_vabs if kwargs['vmin'] is None else kwargs['vmin']
                vmax = data_vabs if kwargs['vmax'] is None else kwargs['vmax']
            if vmin < 0.0 < vmax:
                return colors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax), None, None
            return colors.Normalize(vmin=vmin, vmax=vmax), None, None

        def set_explicit_symlog_colorbar_ticks(cbar, linthresh, vmax):
            if kwargs['norm'] != 'symlog' or vmax is None:
                return
            vabs = abs(float(vmax))
            linthresh = abs(float(linthresh))
            if not np.isfinite(vabs) or not np.isfinite(linthresh) or vabs <= 0.0:
                return
            positive_ticks = []
            if 0.0 < linthresh < vabs:
                positive_ticks.append(linthresh)
                min_power = int(np.ceil(np.log10(linthresh)))
                max_power = int(np.floor(np.log10(vabs)))
                for power in range(min_power, max_power + 1):
                    tick = 10.0 ** power
                    if linthresh < tick < vabs:
                        positive_ticks.append(tick)
            positive_ticks.append(vabs)
            positive_ticks = sorted(set(positive_ticks))
            ticks = [-tick for tick in reversed(positive_ticks)] + [0.0] + positive_ticks
            cbar.set_ticks(ticks)
            cbar.ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda value, _pos: format_colorbar_tick(value))
            )

        shared_norm, shared_vmin, shared_vmax = symmetric_norm_for_states(states)
        fig, axes = plt.subplots(1, len(plane_specs), figsize=(10.0, 4.8), squeeze=False)
        row_mappable = None
        for col_index, (state, plane_spec) in enumerate(zip(states, plane_specs)):
            ax = axes[0, col_index]
            row_mappable = state['render'](
                ax, shared_norm, shared_vmin, shared_vmax,
                x_limits=state['x_limits'],
                y_limits=state['y_limits'],
                title=plane_spec['title'],
                show_bh_zoom=False,
                show_center_zoom=False,
            )
            if col_index > 0:
                ax.set_ylabel('')
        cbar = fig.colorbar(row_mappable, ax=axes[0, :].tolist(),
                            fraction=0.024, pad=0.018, extend='both')
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label(states[0]['label'])
        linthresh, explicit_vmax = div_triptych_symlog_settings()
        set_explicit_symlog_colorbar_ticks(cbar, linthresh, explicit_vmax)

        fig.subplots_adjust(left=0.06, right=0.91, bottom=0.12, top=0.90,
                            wspace=0.22)
        if kwargs['output_file'] != 'show':
            fig.savefig(kwargs['output_file'], dpi=kwargs['dpi'],
                        bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()
        return

    if kwargs['orthogonal_triptych'] and not panel_state_only:
        triptych_input_data = read_input_metadata(kwargs['data_file'])
        triptych_bh_state = parse_live_bh_state(triptych_input_data)
        triptych_density_unit_cgs = get_density_unit_cgs_from_input(triptych_input_data)
        triptych_variable_name = (
            kwargs['variable'][8:] if kwargs['variable'][:8] == 'derived:' else kwargs['variable']
        )
        secondary_panel_rmax = 20.0
        tertiary_panel_x_min_offset = -20.0
        tertiary_panel_x_max_offset = 100.0

        def orthogonal_bound(axis_name, edge_name, fallback):
            explicit = kwargs.get(f'ortho_{axis_name}_{edge_name}')
            if explicit is not None:
                return explicit
            return fallback

        def orthogonal_panel_bound(panel_name, axis_name, edge_name, fallback):
            explicit = kwargs.get(f'ortho_{panel_name}_{axis_name}_{edge_name}')
            if explicit is not None:
                return explicit
            return fallback

        ortho_x_min = orthogonal_bound('x', 'min', kwargs['x1_min'])
        ortho_x_max = orthogonal_bound('x', 'max', kwargs['x1_max'])
        ortho_y_min = orthogonal_bound('y', 'min', kwargs['x2_min'])
        ortho_y_max = orthogonal_bound('y', 'max', kwargs['x2_max'])
        ortho_z_min = orthogonal_bound('z', 'min', ortho_y_min)
        ortho_z_max = orthogonal_bound('z', 'max', ortho_y_max)

        if triptych_bh_state is None:
            yoz_y_min = ortho_y_min
            yoz_y_max = ortho_y_max
            yoz_z_min = ortho_z_min
            yoz_z_max = ortho_z_max
        else:
            yoz_y_min = triptych_bh_state['y'] + tertiary_panel_x_min_offset
            yoz_y_max = triptych_bh_state['y'] + tertiary_panel_x_max_offset
            yoz_z_min = triptych_bh_state['z'] - secondary_panel_rmax
            yoz_z_max = triptych_bh_state['z'] + secondary_panel_rmax

        yoz_y_min = orthogonal_panel_bound('yoz', 'y', 'min', yoz_y_min)
        yoz_y_max = orthogonal_panel_bound('yoz', 'y', 'max', yoz_y_max)
        yoz_z_min = orthogonal_panel_bound('yoz', 'z', 'min', yoz_z_min)
        yoz_z_max = orthogonal_panel_bound('yoz', 'z', 'max', yoz_z_max)
        if not (yoz_y_max > yoz_y_min and yoz_z_max > yoz_z_min):
            raise RuntimeError('Invalid orthogonal triptych yoz-panel bounds.')

        panel_specs = (
            {
                'dimension': 'z',
                'title': 'xoy',
                'location': kwargs['location'],
                'x1_min': ortho_x_min,
                'x1_max': ortho_x_max,
                'x2_min': ortho_y_min,
                'x2_max': ortho_y_max,
                'show_bh_zoom': True,
                'show_center_zoom': True,
                'share_y_group': None,
                'show_ylabel': True,
                'row': 'top',
            },
            {
                'dimension': 'x',
                'title': 'yoz',
                'location': (kwargs['location'] if triptych_bh_state is None
                             else triptych_bh_state['x']),
                'x1_min': yoz_y_min,
                'x1_max': yoz_y_max,
                'x2_min': yoz_z_min,
                'x2_max': yoz_z_max,
                'show_bh_zoom': False,
                'show_center_zoom': False,
                'share_y_group': 'z',
                'show_ylabel': True,
                'row': 'top',
            },
        )
        panel_states = []
        for panel_spec in panel_specs:
            panel_kwargs = dict(kwargs)
            panel_kwargs['dimension'] = panel_spec['dimension']
            panel_kwargs['location'] = panel_spec['location']
            panel_kwargs['orthogonal_triptych'] = False
            panel_kwargs['_panel_state_only'] = True
            panel_kwargs['x1_min'] = panel_spec['x1_min']
            panel_kwargs['x1_max'] = panel_spec['x1_max']
            panel_kwargs['x2_min'] = panel_spec['x2_min']
            panel_kwargs['x2_max'] = panel_spec['x2_max']
            try:
                panel_state = main(**panel_kwargs)
            except RuntimeError as exc:
                raise RuntimeError(
                    'Orthogonal triptych plotting requires a 3D dataset with '
                    'resolvable xoy and yoz slices.'
                ) from exc
            panel_state['title'] = panel_spec['title']
            panel_state['show_bh_zoom'] = panel_spec['show_bh_zoom']
            panel_state['show_center_zoom'] = panel_spec['show_center_zoom']
            panel_state['share_y_group'] = panel_spec['share_y_group']
            panel_state['show_ylabel'] = panel_spec['show_ylabel']
            panel_state['row'] = panel_spec['row']
            panel_states.append(panel_state)

        def valid_limit_values(values):
            values = np.asarray(values, dtype=np.float64)
            valid = np.isfinite(values)
            if kwargs['norm'] == 'log':
                valid &= values > 0.0
            return values[valid]

        if kwargs['vmin'] is None:
            vmin_candidates = []
            for panel_state in panel_states:
                panel_values = valid_limit_values(panel_state['quantity_masked'])
                if panel_values.size > 0:
                    vmin_candidates.append(float(np.min(panel_values)))
            if len(vmin_candidates) == 0:
                raise RuntimeError('Unable to determine a finite color minimum.')
            shared_vmin = min(vmin_candidates)
        else:
            shared_vmin = kwargs['vmin']
        if kwargs['vmax'] is None:
            vmax_candidates = []
            for panel_state in panel_states:
                panel_values = valid_limit_values(panel_state['quantity_masked'])
                if panel_values.size > 0:
                    vmax_candidates.append(float(np.max(panel_values)))
            if len(vmax_candidates) == 0:
                raise RuntimeError('Unable to determine a finite color maximum.')
            shared_vmax = max(vmax_candidates)
        else:
            shared_vmax = kwargs['vmax']
        if kwargs['norm'] == 'linear':
            shared_norm = colors.Normalize(shared_vmin, shared_vmax)
            render_vmin = None
            render_vmax = None
        elif kwargs['norm'] == 'log':
            shared_norm = colors.LogNorm(shared_vmin, shared_vmax)
            render_vmin = None
            render_vmax = None
        else:
            shared_norm = kwargs['norm']
            render_vmin = shared_vmin
            render_vmax = shared_vmax

        top_row_height_in = 5.5
        panel_top_in = 0.40
        panel_bottom_in = 0.75
        left_margin_in = 0.85
        panel_gap_in = 0.32
        cbar_gap_in = 0.55
        cbar_width_in = 0.42
        right_margin_in = 0.80
        top_row_states = [panel_state for panel_state in panel_states
                          if panel_state['row'] == 'top']

        top_panel_widths_in = []
        for panel_state in top_row_states:
            x_min, x_max = panel_state['x_limits']
            y_min, y_max = panel_state['y_limits']
            top_panel_widths_in.append(top_row_height_in
                                       * max(x_max - x_min, 1.0e-12)
                                       / max(y_max - y_min, 1.0e-12))
        top_row_width_in = sum(top_panel_widths_in) + panel_gap_in * (len(top_row_states) - 1)

        fig_height = panel_bottom_in + top_row_height_in + panel_top_in
        fig_width = (left_margin_in + top_row_width_in
                     + cbar_gap_in + cbar_width_in + right_margin_in)
        fig = plt.figure(figsize=(fig_width, fig_height))

        axes = []
        sharey_axes = {}
        mappable = None

        top_row_bottom_in = panel_bottom_in
        x_cursor = left_margin_in
        for panel_index, (panel_state, panel_width_in) in enumerate(
                zip(top_row_states, top_panel_widths_in)):
            sharey = None
            if panel_state['share_y_group'] is not None:
                sharey = sharey_axes.get(panel_state['share_y_group'])
            ax = fig.add_axes([x_cursor / fig_width,
                               top_row_bottom_in / fig_height,
                               panel_width_in / fig_width,
                               top_row_height_in / fig_height],
                              sharey=sharey)
            if panel_state['share_y_group'] is not None \
                    and panel_state['share_y_group'] not in sharey_axes:
                sharey_axes[panel_state['share_y_group']] = ax
            mappable = panel_state['render'](
                ax, shared_norm, render_vmin, render_vmax,
                x_limits=panel_state['x_limits'],
                y_limits=panel_state['y_limits'],
                title=panel_state['title'],
                show_ylabel=panel_state['show_ylabel'],
                ylabel=panel_state['y_label'] if panel_state['show_ylabel'] else None,
                show_bh_zoom=panel_state['show_bh_zoom'],
                show_center_zoom=panel_state['show_center_zoom'],
            )
            if not panel_state['show_ylabel']:
                ax.tick_params(labelleft=False)
            axes.append(ax)
            x_cursor += panel_width_in
            if panel_index < len(top_row_states) - 1:
                x_cursor += panel_gap_in

        cbar_left_in = left_margin_in + top_row_width_in + cbar_gap_in
        cax = fig.add_axes([cbar_left_in / fig_width,
                            panel_bottom_in / fig_height,
                            cbar_width_in / fig_width,
                            top_row_height_in / fig_height])
        make_colorbar(
            fig,
            mappable,
            cax,
            panel_states[0]['label'],
            plot_variable_name=triptych_variable_name,
            plot_density_unit_cgs=triptych_density_unit_cgs,
        )

        if kwargs['output_file'] != 'show':
            fig.savefig(kwargs['output_file'], dpi=kwargs['dpi'],
                        bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()
        return

    # Set physical units
    c_cgs = 2.99792458e10
    kb_cgs = 1.380649e-16
    amu_cgs = 1.660538921e-24
    gg_msun_cgs = 1.32712440018e26
    kappa_a_coefficient = 7.04536e25 + 1.95705e24

    # Set derived dependencies
    derived_dependencies = set_derived_dependencies()

    # Read data
    with open(kwargs['data_file'], 'rb') as f:

        # Get file size
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0, 0)

        # Read header metadata
        line = f.readline().decode('ascii')
        if line != 'Athena binary output version=1.1\n':
            raise RuntimeError('Unrecognized data file format.')
        next(f)
        next(f)
        next(f)
        line = f.readline().decode('ascii')
        if line[:19] != '  size of location=':
            raise RuntimeError('Could not read location size.')
        location_size = int(line[19:])
        line = f.readline().decode('ascii')
        if line[:19] != '  size of variable=':
            raise RuntimeError('Could not read variable size.')
        variable_size = int(line[19:])
        next(f)
        line = f.readline().decode('ascii')
        if line[:12] != '  variables:':
            raise RuntimeError('Could not read variable names.')
        variable_names_base = line[12:].split()
        line = f.readline().decode('ascii')
        if line[:16] != '  header offset=':
            raise RuntimeError('Could not read header offset.')
        header_offset = int(line[16:])

        # Process header metadata
        if location_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported for '
                               'location data.')
        location_format = 'f' if location_size == 4 else 'd'
        if variable_size not in (4, 8):
            raise RuntimeError('Only 4- and 8-byte integer types supported for cell '
                               'data.')
        variable_format = 'f' if variable_size == 4 else 'd'
        num_variables_base = len(variable_names_base)
        if kwargs['variable'][:8] == 'derived:':
            variable_name = kwargs['variable'][8:]
            if variable_name not in derived_dependencies:
                raise RuntimeError('Derived variable "{0}" not valid; options are " \
                        "{{{1}}}.'.format(variable_name,
                                          ', '.join(derived_dependencies.keys())))
            variable_names = []
            variable_inds = []
            for dependency in derived_dependencies[variable_name]:
                if dependency not in variable_names_base:
                    raise RuntimeError('Requirement "{0}" for "{1}" not found.'
                                       .format(dependency, variable_name))
                variable_names.append(dependency)
                variable_ind = 0
                while variable_names_base[variable_ind] != dependency:
                    variable_ind += 1
                variable_inds.append(variable_ind)
        elif kwargs['variable'] == 'level':
            variable_name = kwargs['variable']
            variable_names = [variable_name]
            variable_inds = [-1]
        else:
            variable_name = kwargs['variable']
            if variable_name not in variable_names_base:
                raise RuntimeError('Variable "{0}" not found; options are {{{1}}}.'
                                   .format(variable_name,
                                           ', '.join(variable_names_base)))
            variable_names = [variable_name]
            variable_ind = 0
            while variable_names_base[variable_ind] != variable_name:
                variable_ind += 1
            variable_inds = [variable_ind]
        if kwargs['streamlines']:
            for dependency in ('velx', 'vely', 'velz'):
                if dependency in variable_names:
                    continue
                if dependency not in variable_names_base:
                    raise RuntimeError('Requirement "{0}" for streamline overlay not '
                                       'found.'.format(dependency))
                variable_names.append(dependency)
                variable_ind = 0
                while variable_names_base[variable_ind] != dependency:
                    variable_ind += 1
                variable_inds.append(variable_ind)
        if kwargs['bound_unbound_contour']:
            for dependency in ('velx', 'vely', 'velz', 'grav_phi'):
                if dependency in variable_names:
                    continue
                if dependency not in variable_names_base:
                    raise RuntimeError('Requirement "{0}" for bound/unbound contour not '
                                       'found.'.format(dependency))
                variable_names.append(dependency)
                variable_ind = 0
                while variable_names_base[variable_ind] != dependency:
                    variable_ind += 1
                variable_inds.append(variable_ind)
        variable_names_sorted = \
            [name for _, name in sorted(zip(variable_inds, variable_names))]
        variable_inds_sorted = \
            [ind for ind, _ in sorted(zip(variable_inds, variable_names))]

        # Read input file metadata
        input_data = {}
        start_of_data = f.tell() + header_offset
        while f.tell() < start_of_data:
            line = f.readline().decode('ascii')
            if line[0] == '#':
                continue
            if line[0] == '<':
                section_name = line[1:-2]
                input_data[section_name] = {}
                continue
            key, val = line.split('=', 1)
            input_data[section_name][key.strip()] = val.split('#', 1)[0].strip()

        # Extract number of ghost cells from input file metadata
        try:
            num_ghost = int(input_data['mesh']['nghost'])
        except:  # noqa: E722
            raise RuntimeError('Unable to find number of ghost cells in input file.')

        active_eos_name = ''
        try:
            active_eos_name = TabulatedLteTable.resolve_active_eos_name(
                input_data, variable_names_base
            )
        except RuntimeError:
            active_eos_name = ''
        tabulated_eos = None
        tabulated_names = ('pgas', 'pgas_rho', 'entropy', 'c_s', 'mach', 'div_v_cs',
                           'div_v_panels', 'T', 'xh2', 'xion', 'xhe1', 'xhe2',
                           'gamma1', 'gamma3m1', 'mu', 'beta_rad', 'prad_pgas')
        if (active_eos_name in TABULATED_LTE_EOS_NAMES
                and kwargs['variable'] in ['derived:' + name for name in tabulated_names]):
            tabulated_eos = TabulatedLteTable.from_input_data(
                input_data, kwargs['data_file'], variable_names_base
            )
        tabulated_only_names = ('xh2', 'xion', 'xhe1', 'xhe2', 'gamma1', 'gamma3m1', 'mu',
                                'beta_rad')
        if (kwargs['variable'] in ['derived:' + name for name in tabulated_only_names]
                and tabulated_eos is None):
            raise RuntimeError(f'"{kwargs["variable"]}" is only defined for tabulated LTE EOS.')

        # Extract adiabatic index from input file metadata
        names = ('pgas', 'pgas_rho', 'entropy', 'c_s', 'mach', 'div_v_cs',
                 'div_v_panels', 'T', 'prad_pgas', 'sigmah_rel', 'va_rel', 'wgas',
                 'wgasrad', 'Begas', 'Begasrad', 'cons_hydro_rel_t',
                 'cons_hydro_rel_x', 'cons_hydro_rel_y', 'cons_hydro_rel_z')
        if (kwargs['variable'] in ['derived:' + name for name in names]
                and tabulated_eos is None):
            try:
                gamma_adi = float(input_data['hydro']['gamma'])
            except:  # noqa: E722
                try:
                    gamma_adi = float(input_data['mhd']['gamma'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find adiabatic index in input file.')
        names = ('beta_inv_nr', 'beta_inv_rel', 'wmhd', 'wmhdrad', 'Bemhd', 'Bemhdrad',
                 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y', 'cons_mhd_rel_z')
        if (kwargs['variable'] in ['derived:' + name for name in names]
                and tabulated_eos is None):
            try:
                gamma_adi = float(input_data['mhd']['gamma'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find adiabatic index in input file.')

        # Extract molecular weight from input file metadata
        if kwargs['variable'] == 'derived:T' and tabulated_eos is None:
            try:
                mu = float(input_data['units']['mu'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find molecular weight in input file.')

        # Extract opacity flags and values from input file metadata
        names = ('kappa_a', 'kappa_t', 'alpha_a', 'alpha_t', 'tau_a', 'tau_t')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            try:
                power_opacity = bool(input_data['radiation']['power_opacity'])
            except:  # noqa: E722
                power_opacity = False
            if not power_opacity:
                try:
                    kappa_r_cgs = float(input_data['radiation']['kappa_a'])
                    kappa_pr_cgs = float(input_data['radiation']['kappa_p'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find absorption opacities in input '
                                       'file.')
        names = ('kappa_s', 'kappa_t', 'alpha_s', 'alpha_t', 'tau_s', 'tau_t')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            try:
                kappa_s_cgs = float(input_data['radiation']['kappa_s'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find scattering opacity in input file.')

        # Extract length unit from input file metadata
        names = ('T', 'tau_a', 'tau_s', 'tau_t')
        names_alt = ('kappa_a', 'kappa_t', 'alpha_a', 'alpha_t')
        if (kwargs['variable'] in ['derived:' + name for name in names]
            or (kwargs['variable'] in ['derived:' + name for name in names_alt]
                and power_opacity)):
            if input_data['coord']['general_rel'] == 'true':
                try:
                    length_cgs = float(input_data['units']['bhmass_msun']) * gg_msun_cgs \
                        / c_cgs ** 2
                except:  # noqa: E722
                    raise RuntimeError('Unable to find black hole mass in input file.')
            else:
                try:
                    length_cgs = float(input_data['units']['length_cgs'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find length unit in input file.')

        # Extract time unit from input file metadata
        names = ('kappa_a', 'kappa_t', 'alpha_a', 'alpha_t', 'tau_a', 'tau_t')
        if ((kwargs['variable'] == 'derived:T' and tabulated_eos is None)
                or (kwargs['variable'] in ['derived:' + name for name in names]
                    and power_opacity)):
            if input_data['coord']['general_rel'] == 'true':
                time_cgs = length_cgs / c_cgs
            else:
                try:
                    time_cgs = float(input_data['units']['time_cgs'])
                except:  # noqa: E722
                    raise RuntimeError('Unable to find time unit in input file.')

        density_unit_cgs = get_density_unit_cgs_from_input(input_data)

        # Extract density unit from input file metadata when required explicitly.
        names = ('alpha_a', 'alpha_s', 'alpha_t', 'tau_a', 'tau_s', 'tau_t')
        names_alt = ('kappa_a', 'kappa_t')
        if ((kwargs['variable'] in ['derived:' + name for name in names]
             or (kwargs['variable'] in ['derived:' + name for name in names_alt]
                 and power_opacity))
                and density_unit_cgs is None):
            raise RuntimeError('Unable to find density unit in input file.')

        # Check input file metadata for relativity
        names = ('vr_nr', 'vth_nr', 'vph_nr', 'Br_nr', 'Bth_nr', 'Bph_nr', 'pmag_nr',
                 'beta_inv_nr', 'sigma_nr', 't', 'x', 'y', 'z', 'cons_em_nr_t',
                 'cons_mhd_nr_t', 'cons_mhd_nr_x', 'cons_mhd_nr_y', 'cons_mhd_nr_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            assert input_data['coord']['general_rel'] == 'false', \
                    '"{0}" is only defined for non-GR data.'.format(variable_name)
        names = ('uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth', 'uph', 'u_t', 'u_x', 'u_y',
                 'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz', 'vr_rel', 'vth_rel',
                 'vph_rel', 'bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x',
                 'b_y', 'b_z', 'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel',
                 'pmag_rel', 'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'va_rel',
                 'pmag_prad', 'wgas', 'wmhd', 'wgasrad', 'wmhdrad', 'Begas', 'Bemhd',
                 'Begasrad', 'Bemhdrad', 'cons_hydro_rel_t', 'cons_hydro_rel_x',
                 'cons_hydro_rel_y', 'cons_hydro_rel_z', 'cons_em_rel_t', 'cons_em_rel_x',
                 'cons_em_rel_y', 'cons_em_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x',
                 'cons_mhd_rel_y', 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            assert input_data['coord']['general_rel'] == 'true', \
                    '"{0}" is only defined for GR data.'.format(variable_name)
        if kwargs['horizon'] or kwargs['horizon_mask'] or kwargs['ergosphere']:
            assert input_data['coord']['general_rel'] == 'true', '"horizon", ' \
                    '"horizon_mask", and "ergosphere" options only pertain to GR data.'
        names = ('velx', 'vely', 'velz')
        if kwargs['variable'] in names or kwargs['variable'] in ('derived:vel_xyz',
                                                                 'derived:vel_norm'):
            general_rel_v = bool(input_data['coord']['general_rel'])
        else:
            general_rel_v = False

        # Extract black hole spin from input file metadata
        names = ('uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth', 'uph', 'u_t', 'u_x', 'u_y',
                 'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz', 'vr_rel', 'vth_rel',
                 'vph_rel', 'bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x',
                 'b_y', 'b_z', 'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel',
                 'pmag_rel', 'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'va_rel',
                 'pmag_prad', 'Rtr', 'Rtth', 'Rtph', 'Rrr', 'Rthth', 'Rphph', 'Rrth',
                 'Rrph', 'Rthph', 'Rtr_Rtt', 'Rtth_Rtt', 'Rtph_Rtt', 'Rrr_Rtt',
                 'Rthth_Rtt', 'Rphph_Rtt', 'Rrth_Rtt', 'Rrph_Rtt', 'Rthph_Rtt', 'wmhd',
                 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad', 'Bemhdrad', 'cons_hydro_rel_t',
                 'cons_hydro_rel_x', 'cons_hydro_rel_y', 'cons_hydro_rel_z',
                 'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y', 'cons_em_rel_z',
                 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y', 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            try:
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole spin in input file.')
        if kwargs['horizon'] or kwargs['horizon_mask'] or kwargs['ergosphere']:
            try:
                bh_a = float(input_data['coord']['a'])
            except:  # noqa: E722
                raise RuntimeError('Unable to find black hole spin in input file.')

        # Prepare lists to hold results
        inferred_root_blocks = infer_effective_root_blocks(kwargs['data_file'], input_data)
        divergence_names = ('div_v', 'div_v_cs', 'div_v_panels')
        needs_divergence = kwargs['variable'] in ['derived:' + name for name in divergence_names]
        max_level_calculated = -1
        block_loc_for_level = []
        block_ind_for_level = []
        num_blocks_used = 0
        extents = []
        quantities = {}
        for name in variable_names_sorted:
            quantities[name] = []
        divergence_blocks = []

        # Go through blocks
        first_time = True
        while f.tell() < file_size:

            # Read grid structure data
            block_indices = np.array(struct.unpack('@6i', f.read(24))) - num_ghost
            block_i, block_j, block_k, block_level = struct.unpack('@4i', f.read(16))

            # Process grid structure data
            if first_time:
                block_nx = block_indices[1] - block_indices[0] + 1
                block_ny = block_indices[3] - block_indices[2] + 1
                block_nz = block_indices[5] - block_indices[4] + 1
                cells_per_block = block_nz * block_ny * block_nx
                block_cell_format = '=' + str(cells_per_block) + variable_format
                variable_data_size = cells_per_block * variable_size
                if kwargs['dimension'] is None:
                    if block_nx > 1 and block_ny > 1 and block_nz > 1:
                        kwargs['dimension'] = 'z'
                    elif block_nx > 1 and block_ny > 1:
                        kwargs['dimension'] = 'z'
                    elif block_nx > 1 and block_nz > 1:
                        kwargs['dimension'] = 'y'
                    elif block_ny > 1 and block_nz > 1:
                        kwargs['dimension'] = 'x'
                    else:
                        raise RuntimeError('Input file only contains 1D data.')
                if kwargs['dimension'] == 'x':
                    if block_ny == 1:
                        raise RuntimeError('Data in file has no extent in y-direction.')
                    if block_nz == 1:
                        raise RuntimeError('Data in file has no extent in z-direction.')
                    block_nx1 = block_ny
                    block_nx2 = block_nz
                    slice_block_n = block_nx
                    slice_location_min = float(input_data['mesh']['x1min'])
                    slice_location_max = float(input_data['mesh']['x1max'])
                    slice_root_blocks = inferred_root_blocks.get(
                        'x1',
                        int(input_data['mesh']['nx1']) // int(input_data['meshblock']['nx1']))
                if kwargs['dimension'] == 'y':
                    if block_nx == 1:
                        raise RuntimeError('Data in file has no extent in x-direction.')
                    if block_nz == 1:
                        raise RuntimeError('Data in file has no extent in z-direction.')
                    block_nx1 = block_nx
                    block_nx2 = block_nz
                    slice_block_n = block_ny
                    slice_location_min = float(input_data['mesh']['x2min'])
                    slice_location_max = float(input_data['mesh']['x2max'])
                    slice_root_blocks = inferred_root_blocks.get(
                        'x2',
                        int(input_data['mesh']['nx2']) // int(input_data['meshblock']['nx2']))
                if kwargs['dimension'] == 'z':
                    if block_nx == 1:
                        raise RuntimeError('Data in file has no extent in x-direction.')
                    if block_ny == 1:
                        raise RuntimeError('Data in file has no extent in y-direction.')
                    block_nx1 = block_nx
                    block_nx2 = block_ny
                    slice_block_n = block_nz
                    slice_location_min = float(input_data['mesh']['x3min'])
                    slice_location_max = float(input_data['mesh']['x3max'])
                    slice_root_blocks = inferred_root_blocks.get(
                        'x3',
                        int(input_data['mesh']['nx3']) // int(input_data['meshblock']['nx3']))
                slice_normalized_coord = (kwargs['location'] - slice_location_min) \
                    / (slice_location_max - slice_location_min)
                first_time = False

            # Determine if block is needed
            if block_level > max_level_calculated:
                for level in range(max_level_calculated + 1, block_level + 1):
                    if kwargs['location'] <= slice_location_min:
                        block_loc_for_level.append(0)
                        block_ind_for_level.append(0)
                    elif kwargs['location'] >= slice_location_max:
                        block_loc_for_level.append(slice_root_blocks - 1)
                        block_ind_for_level.append(slice_block_n - 1)
                    else:
                        slice_mesh_n = slice_block_n * slice_root_blocks * 2 ** level
                        mesh_ind = int(slice_normalized_coord * slice_mesh_n)
                        block_loc_for_level.append(mesh_ind // slice_block_n)
                        block_ind_for_level.append(mesh_ind - slice_block_n
                                                   * block_loc_for_level[-1])
                max_level_calculated = block_level
            if kwargs['dimension'] == 'x' and block_i != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'y' and block_j != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            if kwargs['dimension'] == 'z' and block_k != block_loc_for_level[block_level]:
                f.seek(6 * location_size + num_variables_base * variable_data_size, 1)
                continue
            num_blocks_used += 1

            # Read coordinate data
            block_lims = struct.unpack('=6' + location_format, f.read(6 * location_size))
            if kwargs['dimension'] == 'x':
                extents.append((block_lims[2], block_lims[3], block_lims[4],
                                block_lims[5]))
            if kwargs['dimension'] == 'y':
                extents.append((block_lims[0], block_lims[1], block_lims[4],
                                block_lims[5]))
            if kwargs['dimension'] == 'z':
                extents.append((block_lims[0], block_lims[1], block_lims[2],
                                block_lims[3]))

            # Read cell data
            cell_data_start = f.tell()
            if len(variable_inds_sorted) > 0:
                full_velocity_data = {}
                for ind, name in zip(variable_inds_sorted, variable_names_sorted):
                    if ind == -1:
                        if kwargs['dimension'] == 'x':
                            quantities[name].append(np.full((block_nz, block_ny),
                                                            block_level))
                        if kwargs['dimension'] == 'y':
                            quantities[name].append(np.full((block_nz, block_nx),
                                                            block_level))
                        if kwargs['dimension'] == 'z':
                            quantities[name].append(np.full((block_ny, block_nx),
                                                            block_level))
                    else:
                        f.seek(cell_data_start + ind * variable_data_size, 0)
                        cell_data = (np.array(struct.unpack(block_cell_format,
                                                            f.read(variable_data_size)))
                                     .reshape(block_nz, block_ny, block_nx))
                        block_ind = block_ind_for_level[block_level]
                        if needs_divergence and name in ('velx', 'vely', 'velz'):
                            full_velocity_data[name] = cell_data
                        quantities[name].append(slice_block_quantity(
                            cell_data, kwargs['dimension'], block_ind))
                f.seek((num_variables_base - ind - 1) * variable_data_size, 1)
                if needs_divergence:
                    missing_velocity = [
                        name for name in ('velx', 'vely', 'velz') if name not in full_velocity_data
                    ]
                    if missing_velocity:
                        raise RuntimeError(
                            'Unable to compute div_v; missing full block data for '
                            + ', '.join(missing_velocity))
                    div_v_full = block_velocity_divergence(
                        full_velocity_data['velx'],
                        full_velocity_data['vely'],
                        full_velocity_data['velz'],
                        block_lims)
                    div_v_slice = slice_block_quantity(
                        div_v_full, kwargs['dimension'], block_ind)
                    divergence_blocks.append(div_v_slice)
            else:
                f.seek(num_variables_base * variable_data_size, 1)

    # Prepare to calculate derived quantity
    for name in variable_names_sorted:
        quantities[name] = np.array(quantities[name])
    if needs_divergence:
        div_v = np.array(divergence_blocks)

    # Extract optional BH metadata recorded in the file header.
    bh_plot_state = parse_live_bh_state(input_data)

    bound_unbound_quantity = None
    if kwargs['bound_unbound_contour']:
        problem_data = input_data.get('problem', {})
        try:
            bh_vx = float(problem_data.get('bh_live_vx', '0.0')) \
                if problem_data.get('bh_live_state_valid', '0').lower() in ('1', 'true') else 0.0
            bh_vy = float(problem_data.get('bh_live_vy', '0.0')) \
                if problem_data.get('bh_live_state_valid', '0').lower() in ('1', 'true') else 0.0
            bh_vz = float(problem_data.get('bh_live_vz', '0.0')) \
                if problem_data.get('bh_live_state_valid', '0').lower() in ('1', 'true') else 0.0
        except ValueError:
            bh_vx = 0.0
            bh_vy = 0.0
            bh_vz = 0.0
        bound_unbound_quantity = (
            0.5 * ((quantities['velx'] - bh_vx) ** 2
                   + (quantities['vely'] - bh_vy) ** 2
                   + (quantities['velz'] - bh_vz) ** 2)
            + quantities['grav_phi']
        )

    # Calculate gas pressure or related quantity
    names = ('pgas', 'pgas_rho', 'entropy', 'c_s', 'mach', 'T', 'xh2', 'xion', 'xhe1', 'xhe2',
             'gamma1', 'gamma3m1', 'mu', 'beta_rad', 'prad_pgas')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        dens_safe = np.maximum(quantities['dens'], np.finfo(float).tiny)
        if tabulated_eos is not None:
            pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, quantities['eint'])
        else:
            pgas = (gamma_adi - 1.0) * quantities['eint']

        if kwargs['variable'] == 'derived:pgas':
            quantity = pgas
        elif kwargs['variable'] in ('derived:pgas_rho', 'derived:entropy',
                                    'derived:c_s', 'derived:mach', 'derived:T', 'derived:xh2', 'derived:xion',
                                    'derived:xhe1', 'derived:xhe2', 'derived:gamma1',
                                    'derived:gamma3m1', 'derived:mu',
                                    'derived:beta_rad'):
            if kwargs['variable'] == 'derived:pgas_rho':
                quantity = pgas / dens_safe
            elif kwargs['variable'] == 'derived:entropy':
                if tabulated_eos is not None:
                    raise RuntimeError('"derived:entropy" is only defined for gamma-law '
                                       'EOS in this plotting script.')
                quantity = pgas / np.power(dens_safe, gamma_adi)
            elif kwargs['variable'] in ('derived:c_s', 'derived:mach'):
                if tabulated_eos is not None:
                    sound_speed = np.sqrt(np.maximum(
                        tabulated_eos.sound_speed2_from_rho_eint(dens_safe, quantities['eint']),
                        0.0))
                else:
                    sound_speed = np.sqrt(np.maximum(gamma_adi * pgas / dens_safe, 0.0))
                if kwargs['variable'] == 'derived:c_s':
                    quantity = sound_speed
                else:
                    speed = np.sqrt(np.maximum(
                        quantities['velx'] ** 2 + quantities['vely'] ** 2 + quantities['velz'] ** 2,
                        0.0))
                    quantity = speed / np.maximum(sound_speed, SAFE_POSITIVE)
            else:
                if tabulated_eos is not None:
                    override_plot_chemistry = kwargs['variable'] in (
                        'derived:xion',
                        'derived:xhe1',
                        'derived:xhe2',
                        'derived:mu',
                    )
                    temp_code, xh2, xion, xhe1, xhe2, gamma1, gamma3m1, mu_tab, beta_rad = \
                        tabulated_eos.thermo_fields_from_rho_eint(
                        dens_safe, quantities['eint'],
                        override_plot_chemistry=override_plot_chemistry)
                    if kwargs['variable'] == 'derived:T':
                        quantity = temp_code * tabulated_eos.temp_unit_cgs
                    elif kwargs['variable'] == 'derived:xh2':
                        quantity = xh2
                    elif kwargs['variable'] == 'derived:xion':
                        quantity = xion
                    elif kwargs['variable'] == 'derived:xhe1':
                        quantity = xhe1
                    elif kwargs['variable'] == 'derived:xhe2':
                        quantity = xhe2
                    elif kwargs['variable'] == 'derived:gamma1':
                        quantity = gamma1
                    elif kwargs['variable'] == 'derived:mu':
                        quantity = mu_tab
                    elif kwargs['variable'] == 'derived:beta_rad':
                        quantity = beta_rad
                    else:
                        quantity = gamma3m1
                else:
                    quantity = (mu * amu_cgs / kb_cgs * (length_cgs / time_cgs) ** 2
                                * pgas / dens_safe)
        else:
            if tabulated_eos is not None:
                quantity = tabulated_eos.prad_pgas_from_rho_eint(dens_safe, quantities['eint'])
            else:
                prad = quantities['r00_ff'] / 3.0
                quantity = prad / pgas

    # Prepare Cartesian velocity panels
    if kwargs['variable'] == 'derived:vel_xyz':
        velocity_panels = (quantities['velx'], quantities['vely'], quantities['velz'])

    # Calculate velocity divergence and sound-speed-normalized divergence
    if kwargs['variable'] in ('derived:div_v', 'derived:div_v_cs', 'derived:div_v_panels'):
        if kwargs['variable'] == 'derived:div_v_cs':
            dens_safe = np.maximum(quantities['dens'], np.finfo(float).tiny)
            if tabulated_eos is not None:
                sound_speed = np.sqrt(np.maximum(
                    tabulated_eos.sound_speed2_from_rho_eint(dens_safe, quantities['eint']),
                    0.0))
            else:
                pgas = (gamma_adi - 1.0) * quantities['eint']
                sound_speed = np.sqrt(np.maximum(gamma_adi * pgas / dens_safe, 0.0))
            problem_data = input_data.get('problem', {})
            try:
                star_radius = float(problem_data.get('star_radius'))
                mass_ratio = float(problem_data.get('mass_ratio'))
            except (TypeError, ValueError):
                raise RuntimeError(
                    'Cannot compute derived:div_v_cs without problem/star_radius and problem/mass_ratio.'
                )
            tidal_radius = star_radius * mass_ratio ** (1.0 / 3.0)
            div_v_cs = div_v * max(tidal_radius, SAFE_POSITIVE) / np.maximum(sound_speed, SAFE_POSITIVE)
        if kwargs['variable'] == 'derived:div_v':
            quantity = div_v
        elif kwargs['variable'] == 'derived:div_v_cs':
            quantity = div_v_cs
        else:
            divergence_panels = (div_v,)

    # Calculate transformed gravity potential
    if kwargs['variable'] == 'derived:grav_phi_abs':
        quantity = np.maximum(np.abs(quantities['grav_phi']), SAFE_POSITIVE)

    # Calculate velocity norm
    if kwargs['variable'] == 'derived:vel_norm':
        quantity = np.sqrt(np.maximum(
            quantities['velx'] ** 2 + quantities['vely'] ** 2 + quantities['velz'] ** 2,
            0.0))

    # Calculate non-relativistic velocity
    names = ('vr_nr', 'vth_nr', 'vph_nr')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        vx = quantities['velx']
        vy = quantities['vely']
        vz = quantities['velz']
        vr, vth, vph = cart_to_sph(vx, vy, vz, x, y, z)
        if kwargs['variable'] == 'derived:vr':
            quantity = vr
        elif kwargs['variable'] == 'derived:vth':
            quantity = vth
        else:
            quantity = vph

    # Calculate relativistic velocity
    names = ('uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth', 'uph', 'u_t', 'u_x', 'u_y',
             'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz', 'vr_rel', 'vth_rel',
             'vph_rel')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        if kwargs['variable'] == 'derived:uut':
            quantity = uut
        names = ('ut', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            if kwargs['variable'] == 'derived:ut':
                quantity = ut
            elif kwargs['variable'] == 'derived:ux':
                quantity = ux
            elif kwargs['variable'] == 'derived:uy':
                quantity = uy
            elif kwargs['variable'] == 'derived:uz':
                quantity = uz
            elif kwargs['variable'] == 'derived:vx':
                quantity = ux / ut
            elif kwargs['variable'] == 'derived:vy':
                quantity = uy / ut
            else:
                quantity = uz / ut
        names = ('ur', 'uth', 'uph', 'vr_rel', 'vth_rel', 'vph_rel')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            ur, uth, uph = cks_to_sks_vec_con(ux, uy, uz, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:ur':
                quantity = ur
            elif kwargs['variable'] == 'derived:uth':
                quantity = uth
            elif kwargs['variable'] == 'derived:uph':
                quantity = uph
            elif kwargs['variable'] == 'derived:vr_rel':
                quantity = ur / ut
            elif kwargs['variable'] == 'derived:vth_rel':
                quantity = uth / ut
            else:
                quantity = uph / ut
        names = ('u_t', 'u_x', 'u_y', 'u_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            if kwargs['variable'] == 'derived:u_t':
                quantity = u_t
            elif kwargs['variable'] == 'derived:u_x':
                quantity = u_x
            elif kwargs['variable'] == 'derived:u_y':
                quantity = u_y
            else:
                quantity = u_z
        names = ('u_r', 'u_th', 'u_ph')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            u_r, u_th, u_ph = cks_to_sks_vec_cov(u_x, u_y, u_z, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:u_r':
                quantity = u_r
            elif kwargs['variable'] == 'derived:u_th':
                quantity = u_th
            else:
                quantity = u_ph

    # Calculate non-relativistic magnetic field or related quantity
    names = ('Br_nr', 'Bth_nr', 'Bph_nr', 'pmag_nr', 'beta_inv_nr', 'sigma_nr')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        bbx = quantities['bcc1']
        bby = quantities['bcc2']
        bbz = quantities['bcc3']
        names = ('Br_nr', 'Bth_nr', 'Bph_nr')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                          kwargs['dimension'], kwargs['location'])
            bbr, bbth, bbph = cart_to_sph(bbx, bby, bbz, x, y, z)
            if kwargs['variable'] == 'derived:Br_nr':
                quantity = bbr
            elif kwargs['variable'] == 'derived:Bth_nr':
                quantity = bbth
            else:
                quantity = bbph
        names = ('pmag_nr', 'beta_inv_nr', 'sigma_nr')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            pmag = 0.5 * (bbx ** 2 + bby ** 2 + bbz ** 2)
            if kwargs['variable'] == 'derived:pmag_nr':
                quantity = pmag
            elif kwargs['variable'] == 'derived:beta_inv_nr':
                dens_safe = np.maximum(quantities['dens'], np.finfo(float).tiny)
                if tabulated_eos is not None:
                    pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, quantities['eint'])
                else:
                    pgas = (gamma_adi - 1.0) * quantities['eint']
                quantity = pmag / pgas
            else:
                quantity = 2.0 * pmag / quantities['dens']

    # Calculate relativistic magnetic field or related quantity
    names = ('bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x', 'b_y', 'b_z',
             'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel', 'pmag_rel',
             'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'va_rel', 'pmag_prad')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        bbx = quantities['bcc1']
        bby = quantities['bcc2']
        bbz = quantities['bcc3']
        bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x,
                                                   u_y, u_z)
        names = ('bt', 'bx', 'by', 'bz')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            if kwargs['variable'] == 'derived:bt':
                quantity = bt
            elif kwargs['variable'] == 'derived:bx':
                quantity = bx
            elif kwargs['variable'] == 'derived:by':
                quantity = by
            else:
                quantity = bz
        names = ('br', 'bth', 'bph')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            br, bth, bph = cks_to_sks_vec_con(bx, by, bz, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:br':
                quantity = br
            elif kwargs['variable'] == 'derived:bth':
                quantity = bth
            else:
                quantity = bph
        names = ('b_t', 'b_x', 'b_y', 'b_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            if kwargs['variable'] == 'derived:b_t':
                quantity = b_t
            elif kwargs['variable'] == 'derived:b_x':
                quantity = b_x
            elif kwargs['variable'] == 'derived:b_y':
                quantity = b_y
            else:
                quantity = b_z
        names = ('b_r', 'b_th', 'b_ph')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            b_r, b_th, b_ph = cks_to_sks_vec_cov(b_x, b_y, b_z, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:b_r':
                quantity = b_r
            elif kwargs['variable'] == 'derived:b_th':
                quantity = b_th
            else:
                quantity = b_ph
        names = ('Br_rel', 'Bth_rel', 'Bph_rel')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            ur, uth, uph = cks_to_sks_vec_con(ux, uy, uz, bh_a, x, y, z)
            br, bth, bph = cks_to_sks_vec_con(bx, by, bz, bh_a, x, y, z)
            if kwargs['variable'] == 'derived:Br_rel':
                quantity = br * ut - bt * ur
            elif kwargs['variable'] == 'derived:Bth_rel':
                quantity = bth * ut - bt * uth
            else:
                quantity = bph * ut - bt * uph
        names = ('pmag_rel', 'beta_inv_rel', 'sigma_rel', 'sigmah_rel', 'va_rel',
                 'pmag_prad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            pmag = 0.5 * (b_t * bt + b_x * bx + b_y * by + b_z * bz)
            if kwargs['variable'] == 'derived:pmag_rel':
                quantity = pmag
            elif kwargs['variable'] == 'derived:beta_inv_rel':
                dens_safe = np.maximum(quantities['dens'], np.finfo(float).tiny)
                if tabulated_eos is not None:
                    pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, quantities['eint'])
                else:
                    pgas = (gamma_adi - 1.0) * quantities['eint']
                quantity = pmag / pgas
            elif kwargs['variable'] == 'derived:sigma_rel':
                quantity = 2.0 * pmag / quantities['dens']
            elif kwargs['variable'] == 'derived:sigmah_rel':
                dens_safe = np.maximum(quantities['dens'], np.finfo(float).tiny)
                if tabulated_eos is not None:
                    pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, quantities['eint'])
                    whydro = quantities['dens'] + quantities['eint'] + pgas
                else:
                    whydro = quantities['dens'] + gamma_adi * quantities['eint']
                quantity = 2.0 * pmag / whydro
            elif kwargs['variable'] == 'derived:va_rel':
                dens_safe = np.maximum(quantities['dens'], np.finfo(float).tiny)
                if tabulated_eos is not None:
                    pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, quantities['eint'])
                    wmhd = quantities['dens'] + quantities['eint'] + pgas + 2.0 * pmag
                else:
                    wmhd = quantities['dens'] + gamma_adi * quantities['eint'] + 2.0 * pmag
                quantity = np.sqrt(2.0 * pmag / wmhd)
            else:
                prad = quantities['r00_ff'] / 3.0
                with warnings.catch_warnings():
                    message = 'divide by zero encountered in divide'
                    warnings.filterwarnings('ignore', message=message,
                                            category=RuntimeWarning)
                    message = 'divide by zero encountered in true_divide'
                    warnings.filterwarnings('ignore', message=message,
                                            category=RuntimeWarning)
                    quantity = pmag / prad

    # Calculate relativistic radiation quantity
    names = ('prad', 'Rtr', 'Rtth', 'Rtph', 'Rrr', 'Rthth', 'Rphph', 'Rrth', 'Rrph',
             'Rthph', 'Rtx_Rtt', 'Rty_Rtt', 'Rtz_Rtt', 'Rxx_Rtt', 'Ryy_Rtt', 'Rzz_Rtt',
             'Rxy_Rtt', 'Rxz_Rtt', 'Ryz_Rtt', 'Rtr_Rtt', 'Rtth_Rtt', 'Rtph_Rtt',
             'Rrr_Rtt', 'Rthth_Rtt', 'Rphph_Rtt', 'Rrth_Rtt', 'Rrph_Rtt', 'Rthph_Rtt',
             'R01_R00_ff', 'R02_R00_ff', 'R03_R00_ff', 'R11_R00_ff', 'R22_R00_ff',
             'R33_R00_ff', 'R12_R00_ff', 'R13_R00_ff', 'R23_R00_ff')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        if kwargs['variable'] == 'derived:prad':
            quantity = quantities['r00_ff'] / 3.0
        with warnings.catch_warnings():
            message = 'invalid value encountered in divide'
            warnings.filterwarnings('ignore', message=message, category=RuntimeWarning)
            message = 'invalid value encountered in true_divide'
            warnings.filterwarnings('ignore', message=message, category=RuntimeWarning)
            names = ('Rtr', 'Rtth', 'Rtph', 'Rtr_Rtt', 'Rtth_Rtt', 'Rtph_Rtt')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                              kwargs['dimension'], kwargs['location'])
                rrtr, rrtth, rrtph = cks_to_sks_vec_con(quantities['r01'],
                                                        quantities['r02'],
                                                        quantities['r03'], bh_a, x, y, z)
                if kwargs['variable'] == 'derived:Rtr':
                    quantity = rrtr
                if kwargs['variable'] == 'derived:Rtth':
                    quantity = rrtth
                if kwargs['variable'] == 'derived:Rtph':
                    quantity = rrtph
                if kwargs['variable'] == 'derived:Rtr_Rtt':
                    quantity = rrtr / quantities['r00']
                if kwargs['variable'] == 'derived:Rtth_Rtt':
                    quantity = rrtth / quantities['r00']
                if kwargs['variable'] == 'derived:Rtph_Rtt':
                    quantity = rrtph / quantities['r00']
            names = ('Rrr', 'Rthth', 'Rphph', 'Rrth', 'Rrph', 'Rthph', 'Rrr_Rtt',
                     'Rthth_Rtt', 'Rphph_Rtt', 'Rrth_Rtt', 'Rrph_Rtt', 'Rthph_Rtt')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                              kwargs['dimension'], kwargs['location'])
                rrrr, rrrth, rrrph, _, rrthth, rrthph, _, _, rrphph = cks_to_sks_tens_con(
                    quantities['r11'], quantities['r12'], quantities['r13'],
                    quantities['r12'], quantities['r22'], quantities['r23'],
                    quantities['r13'], quantities['r23'], quantities['r33'], bh_a, x, y,
                    z)
                if kwargs['variable'] == 'derived:Rrr':
                    quantity = rrrr
                if kwargs['variable'] == 'derived:Rthth':
                    quantity = rrthth
                if kwargs['variable'] == 'derived:Rphph':
                    quantity = rrphph
                if kwargs['variable'] == 'derived:Rrth':
                    quantity = rrrth
                if kwargs['variable'] == 'derived:Rrph':
                    quantity = rrrph
                if kwargs['variable'] == 'derived:Rthph':
                    quantity = rrthph
                if kwargs['variable'] == 'derived:Rrr_Rtt':
                    quantity = rrrr / quantities['r00']
                if kwargs['variable'] == 'derived:Rthth_Rtt':
                    quantity = rrthth / quantities['r00']
                if kwargs['variable'] == 'derived:Rphph_Rtt':
                    quantity = rrphph / quantities['r00']
                if kwargs['variable'] == 'derived:Rrth_Rtt':
                    quantity = rrrth / quantities['r00']
                if kwargs['variable'] == 'derived:Rrph_Rtt':
                    quantity = rrrph / quantities['r00']
                if kwargs['variable'] == 'derived:Rthph_Rtt':
                    quantity = rrthph / quantities['r00']
            if kwargs['variable'] == 'derived:Rtx_Rtt':
                quantity = quantities['r01'] / quantities['r00']
            if kwargs['variable'] == 'derived:Rty_Rtt':
                quantity = quantities['r02'] / quantities['r00']
            if kwargs['variable'] == 'derived:Rtz_Rtt':
                quantity = quantities['r03'] / quantities['r00']
            if kwargs['variable'] == 'derived:Rxx_Rtt':
                quantity = quantities['r11'] / quantities['r00']
            if kwargs['variable'] == 'derived:Ryy_Rtt':
                quantity = quantities['r22'] / quantities['r00']
            if kwargs['variable'] == 'derived:Rzz_Rtt':
                quantity = quantities['r33'] / quantities['r00']
            if kwargs['variable'] == 'derived:Rxy_Rtt':
                quantity = quantities['r12'] / quantities['r00']
            if kwargs['variable'] == 'derived:Rxz_Rtt':
                quantity = quantities['r13'] / quantities['r00']
            if kwargs['variable'] == 'derived:Ryz_Rtt':
                quantity = quantities['r23'] / quantities['r00']
            if kwargs['variable'] == 'derived:R01_R00_ff':
                quantity = quantities['r01_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R02_R00_ff':
                quantity = quantities['r02_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R03_R00_ff':
                quantity = quantities['r03_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R11_R00_ff':
                quantity = quantities['r11_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R22_R00_ff':
                quantity = quantities['r22_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R33_R00_ff':
                quantity = quantities['r33_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R12_R00_ff':
                quantity = quantities['r12_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R13_R00_ff':
                quantity = quantities['r13_ff'] / quantities['r00_ff']
            if kwargs['variable'] == 'derived:R23_R00_ff':
                quantity = quantities['r23_ff'] / quantities['r00_ff']

    # Calculate relativistic opacity quantity:
    names = ('kappa_a', 'kappa_s', 'kappa_t', 'alpha_a', 'alpha_s', 'alpha_t', 'tau_a',
             'tau_s', 'tau_t')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        print('\nWarning: Opacity inferred based on particular version of AthenaK.\n')
        names = ('kappa_a', 'kappa_t', 'alpha_a', 'alpha_t', 'tau_a', 'tau_t')
        if kwargs['variable'] in ['derived:' + name for name in names] and power_opacity:
            rho_cgs = quantities['dens'] * density_unit_cgs
            tt_scaled = quantities['eint'] * length_cgs ** 2 * amu_cgs \
                / (quantities['dens'] * time_cgs ** 2 * kb_cgs)
            kappa_a_cgs = kappa_a_coefficient * rho_cgs / tt_scaled ** 3.5
        if kwargs['variable'] in ['derived:' + name for name in names]:
            if not power_opacity:
                kappa_a_cgs = kappa_pr_cgs + kappa_r_cgs
        if kwargs['variable'] == 'derived:kappa_a':
            quantity = kappa_a_cgs
        if kwargs['variable'] == 'derived:kappa_s':
            quantity = kappa_s_cgs * np.ones((num_blocks_used, 1, 1))
        if kwargs['variable'] == 'derived:kappa_t':
            quantity = kappa_a_cgs + kappa_s_cgs
        if kwargs['variable'] == 'derived:alpha_a':
            quantity = kappa_a_cgs * quantities['dens'] * density_unit_cgs
        if kwargs['variable'] == 'derived:alpha_s':
            quantity = kappa_s_cgs * quantities['dens'] * density_unit_cgs
        if kwargs['variable'] == 'derived:alpha_t':
            quantity = (kappa_a_cgs + kappa_s_cgs) * quantities['dens'] * density_unit_cgs
        if kwargs['variable'] == 'derived:tau_a':
            quantity = kappa_a_cgs * quantities['dens'] * density_unit_cgs * length_cgs
        if kwargs['variable'] == 'derived:tau_s':
            quantity = kappa_s_cgs * quantities['dens'] * density_unit_cgs * length_cgs
        if kwargs['variable'] == 'derived:tau_t':
            quantity = (kappa_a_cgs + kappa_s_cgs) \
                * quantities['dens'] * density_unit_cgs * length_cgs

    # Calculate relativistic enthalpy density or Bernoulli parameter
    names = ('wgas', 'wmhd', 'wgasrad', 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad',
             'Bemhdrad')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        rho = quantities['dens']
        ugas = quantities['eint']
        if tabulated_eos is not None:
            dens_safe = np.maximum(rho, np.finfo(float).tiny)
            pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, ugas)
            w = rho + ugas + pgas
        else:
            w = rho + gamma_adi * ugas
        names = ('wgasrad', 'wmhdrad', 'Begasrad', 'Bemhdrad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            urad = quantities['r00_ff']
            w += 4.0 / 3.0 * urad
        names = ('wgas', 'wgasrad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            quantity = w
        names = ('wmhd', 'wmhdrad', 'Begas', 'Bemhd', 'Begasrad', 'Bemhdrad')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                          kwargs['dimension'], kwargs['location'])
            alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
                g_yz, g_zz = cks_geometry(bh_a, x, y, z)
            uux = quantities['velx']
            uuy = quantities['vely']
            uuz = quantities['velz']
            uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
            u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            names = ('wmhd', 'wmhdrad', 'Bemhd', 'Bemhdrad')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                bbx = quantities['bcc1']
                bby = quantities['bcc2']
                bbz = quantities['bcc3']
                bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz,
                                                           u_x, u_y, u_z)
                b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                                  g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
                w += b_t * bt + b_x * bx + b_y * by + b_z * bz
            names = ('wmhd', 'wmhdrad')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = w
            names = ('Begas', 'Bemhd', 'Begasrad', 'Bemhdrad')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = -u_t * w / rho - 1.0

    # Calculate non-relativistic conserved quantity
    names = ('cons_hydro_nr_t', 'cons_hydro_nr_x', 'cons_hydro_nr_y', 'cons_hydro_nr_z',
             'cons_em_nr_t', 'cons_mhd_nr_t', 'cons_mhd_nr_x', 'cons_mhd_nr_y',
             'cons_mhd_nr_z')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        quantity = 0.0
        names = ('cons_hydro_nr_t', 'cons_mhd_nr_t')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            ugas = quantities['eint']
            vx = quantities['velx']
            vy = quantities['vely']
            vz = quantities['velz']
            quantity = 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2) + ugas
        names = ('cons_hydro_nr_x', 'cons_mhd_nr_x')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            vx = quantities['velx']
            quantity = rho * vx
        names = ('cons_hydro_nr_y', 'cons_mhd_nr_y')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            vy = quantities['vely']
            quantity = rho * vy
        names = ('cons_hydro_nr_z', 'cons_mhd_nr_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            vz = quantities['velz']
            quantity = rho * vz
        names = ('cons_em_nr_t', 'cons_mhd_nr_t')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            bbx = quantities['bcc1']
            bby = quantities['bcc2']
            bbz = quantities['bcc3']
            quantity += 0.5 * (bbx ** 2 + bby ** 2 + bbz ** 2)

    # Calculate relativistic conserved quantity
    names = ('cons_hydro_rel_t', 'cons_hydro_rel_x', 'cons_hydro_rel_y',
             'cons_hydro_rel_z', 'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y',
             'cons_em_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y',
             'cons_mhd_rel_z')
    if kwargs['variable'] in ['derived:' + name for name in names]:
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
            g_yz, g_zz = cks_geometry(bh_a, x, y, z)
        uux = quantities['velx']
        uuy = quantities['vely']
        uuz = quantities['velz']
        uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
        ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz)
        u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz, g_tt, g_tx, g_ty, g_tz, g_xx,
                                          g_xy, g_xz, g_yy, g_yz, g_zz)
        quantity = 0.0
        names = ('cons_hydro_rel_t', 'cons_hydro_rel_x', 'cons_hydro_rel_y',
                 'cons_hydro_rel_z', 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y',
                 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            rho = quantities['dens']
            ugas = quantities['eint']
            dens_safe = np.maximum(rho, np.finfo(float).tiny)
            if tabulated_eos is not None:
                pgas = tabulated_eos.pressure_from_rho_eint(dens_safe, ugas)
            else:
                pgas = (gamma_adi - 1.0) * ugas
            wgas = rho + ugas + pgas
            names = ('cons_hydro_rel_t', 'cons_mhd_rel_t')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_t + pgas
            names = ('cons_hydro_rel_x', 'cons_mhd_rel_x')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_x
            names = ('cons_hydro_rel_y', 'cons_mhd_rel_y')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_y
            names = ('cons_hydro_rel_z', 'cons_mhd_rel_z')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity = wgas * ut * u_z
        names = ('cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y', 'cons_em_rel_z',
                 'cons_mhd_rel_t', 'cons_mhd_rel_x', 'cons_mhd_rel_y', 'cons_mhd_rel_z')
        if kwargs['variable'] in ['derived:' + name for name in names]:
            bbx = quantities['bcc1']
            bby = quantities['bcc2']
            bbz = quantities['bcc3']
            bt, bx, by, bz = three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x,
                                                       u_y, u_z)
            b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz, g_tt, g_tx, g_ty, g_tz,
                                              g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
            umag = 0.5 * (b_t * bt + b_x * bx + b_y * by + b_z * bz)
            pmag = umag
            names = ('cons_em_rel_t', 'cons_mhd_rel_t')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_t + pmag - bt * b_t
            names = ('cons_em_rel_x', 'cons_mhd_rel_x')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_x - bt * b_x
            names = ('cons_em_rel_y', 'cons_mhd_rel_y')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_y - bt * b_y
            names = ('cons_em_rel_z', 'cons_mhd_rel_z')
            if kwargs['variable'] in ['derived:' + name for name in names]:
                quantity += (umag + pmag) * ut * u_z - bt * b_z

    # Extract quantity without derivation
    if kwargs['variable'][:8] != 'derived:':
        quantity = quantities[variable_name]
        if variable_name == 'temperature':
            if tabulated_eos is not None:
                quantity = quantity * tabulated_eos.temp_unit_cgs
            elif ('units' in input_data and ('hydro' in input_data or 'mhd' in input_data)
                  and 'mu' in input_data['units']
                  and 'length_cgs' in input_data['units']
                  and 'time_cgs' in input_data['units']):
                mu_raw = float(input_data['units']['mu'])
                length_cgs_raw = float(input_data['units']['length_cgs'])
                time_cgs_raw = float(input_data['units']['time_cgs'])
                quantity = quantity * (
                    mu_raw * amu_cgs / kb_cgs * (length_cgs_raw / time_cgs_raw) ** 2)

    def get_bh_plot_coordinates():
        if bh_plot_state is None:
            return None
        if kwargs['dimension'] == 'x':
            plot_x = bh_plot_state['y']
            plot_y = bh_plot_state['z']
            normal_offset = abs(bh_plot_state['x'] - kwargs['location'])
        elif kwargs['dimension'] == 'y':
            plot_x = bh_plot_state['x']
            plot_y = bh_plot_state['z']
            normal_offset = abs(bh_plot_state['y'] - kwargs['location'])
        else:
            plot_x = bh_plot_state['x']
            plot_y = bh_plot_state['y']
            normal_offset = abs(bh_plot_state['z'] - kwargs['location'])
        return plot_x, plot_y, normal_offset

    def add_bh_overlay(ax):
        if not kwargs['bh_mask']:
            return
        bh_coords = get_bh_plot_coordinates()
        if bh_coords is None:
            return
        plot_x, plot_y, normal_offset = bh_coords

        radius = bh_plot_state['radius']
        if radius > 0.0 and normal_offset <= radius:
            circle_radius = np.sqrt(max(radius ** 2 - normal_offset ** 2, 0.0))
            if circle_radius > 0.0:
                ax.add_artist(patches.Circle((plot_x, plot_y), radius=circle_radius,
                                             linewidth=kwargs['bh_mask_linewidth'],
                                             facecolor=kwargs['bh_mask_color'],
                                             edgecolor=kwargs['bh_mask_edgecolor'],
                                             alpha=kwargs['bh_mask_alpha'],
                                             zorder=6))
                min_radius_pixels = kwargs['bh_mask_min_pixels']
                if min_radius_pixels > 0.0:
                    center_px = ax.transData.transform((plot_x, plot_y))
                    edge_x_px = ax.transData.transform((plot_x + circle_radius, plot_y))
                    edge_y_px = ax.transData.transform((plot_x, plot_y + circle_radius))
                    radius_x_px = abs(edge_x_px[0] - center_px[0])
                    radius_y_px = abs(edge_y_px[1] - center_px[1])
                    displayed_radius_px = max(min(radius_x_px, radius_y_px), 0.0)
                    if displayed_radius_px < min_radius_pixels:
                        radius_points = min_radius_pixels * 72.0 / ax.figure.dpi
                        ax.scatter([plot_x], [plot_y], s=(2.0 * radius_points) ** 2,
                                   c=kwargs['bh_mask_color'],
                                   edgecolors=kwargs['bh_mask_edgecolor'],
                                   linewidths=kwargs['bh_mask_linewidth'],
                                   alpha=kwargs['bh_mask_alpha'], zorder=7)
        if kwargs['bh_marker']:
            ax.scatter([plot_x], [plot_y], s=kwargs['bh_marker_size'],
                       c=kwargs['bh_marker_color'], edgecolors='none', zorder=7)

    def get_streamline_components():
        if not kwargs['streamlines']:
            return None, None
        if kwargs['dimension'] == 'x':
            return quantities['vely'], quantities['velz']
        if kwargs['dimension'] == 'y':
            return quantities['velx'], quantities['velz']
        return quantities['velx'], quantities['vely']

    streamline_grid_cache = {}

    def build_streamline_grid(stream_u, stream_v, x_min, x_max, y_min, y_max):
        if stream_u is None or stream_v is None:
            return None
        x_span = x_max - x_min
        y_span = y_max - y_min
        if x_span <= 0.0 or y_span <= 0.0:
            return None

        cache_key = (float(x_min), float(x_max), float(y_min), float(y_max),
                     int(kwargs['stream_resolution']))
        cached = streamline_grid_cache.get(cache_key)
        if cached is not None:
            return cached

        max_res = max(int(kwargs['stream_resolution']), 16)
        if x_span >= y_span:
            nx = max_res
            ny = max(16, int(round(max_res * y_span / x_span)))
        else:
            ny = max_res
            nx = max(16, int(round(max_res * x_span / y_span)))

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        grid_u = np.full((ny, nx), np.nan)
        grid_v = np.full((ny, nx), np.nan)
        grid_cell_area = np.full((ny, nx), np.inf)

        for block_num in range(num_blocks_used):
            x0, x1, y0, y1 = extents[block_num]
            if x1 <= x_min or x0 >= x_max or y1 <= y_min or y0 >= y_max:
                continue

            block_u = np.asarray(stream_u[block_num], dtype=np.float64)
            block_v = np.asarray(stream_v[block_num], dtype=np.float64)
            valid = np.isfinite(block_u) & np.isfinite(block_v)
            if not np.any(valid):
                continue

            block_nx = block_u.shape[1]
            block_ny = block_u.shape[0]
            dx = (x1 - x0) / block_nx
            dy = (y1 - y0) / block_ny
            cell_area = abs(dx * dy)

            ix0 = max(0, int(np.floor((max(x0, x_min) - x_min) / x_span * (nx - 1))))
            ix1 = min(nx - 1, int(np.ceil((min(x1, x_max) - x_min) / x_span * (nx - 1))))
            iy0 = max(0, int(np.floor((max(y0, y_min) - y_min) / y_span * (ny - 1))))
            iy1 = min(ny - 1, int(np.ceil((min(y1, y_max) - y_min) / y_span * (ny - 1))))
            if ix1 < ix0 or iy1 < iy0:
                continue

            x_sub = x[ix0:ix1 + 1]
            y_sub = y[iy0:iy1 + 1]
            src_ix = np.clip(np.rint((x_sub - (x0 + 0.5 * dx)) / dx).astype(np.int64),
                             0, block_nx - 1)
            src_iy = np.clip(np.rint((y_sub - (y0 + 0.5 * dy)) / dy).astype(np.int64),
                             0, block_ny - 1)

            sampled_u = block_u[np.ix_(src_iy, src_ix)]
            sampled_v = block_v[np.ix_(src_iy, src_ix)]
            sampled_valid = np.isfinite(sampled_u) & np.isfinite(sampled_v)
            current_area = grid_cell_area[iy0:iy1 + 1, ix0:ix1 + 1]
            overwrite = sampled_valid & (cell_area <= current_area)
            if not np.any(overwrite):
                continue

            grid_u_slice = grid_u[iy0:iy1 + 1, ix0:ix1 + 1]
            grid_v_slice = grid_v[iy0:iy1 + 1, ix0:ix1 + 1]
            grid_u_slice[overwrite] = sampled_u[overwrite]
            grid_v_slice[overwrite] = sampled_v[overwrite]
            current_area[overwrite] = cell_area

        speed = np.hypot(grid_u, grid_v)
        valid_grid = np.isfinite(speed) & (speed > 0.0)
        if not np.any(valid_grid):
            streamline_grid_cache[cache_key] = None
            return None

        grid = (x, y, np.ma.masked_invalid(grid_u), np.ma.masked_invalid(grid_v))
        streamline_grid_cache[cache_key] = grid
        return grid

    def add_streamlines(ax, stream_u, stream_v, x_min, x_max, y_min, y_max,
                        arrowsize=None):
        stream_grid = build_streamline_grid(stream_u, stream_v, x_min, x_max, y_min, y_max)
        if stream_grid is None:
            return
        x, y, grid_u, grid_v = stream_grid
        effective_arrowsize = kwargs['stream_arrowsize'] if arrowsize is None else arrowsize
        stream = ax.streamplot(x, y, grid_u, grid_v, color=kwargs['stream_color'],
                               linewidth=0.6, density=kwargs['stream_density'],
                               arrowsize=effective_arrowsize, zorder=4)
        stream.lines.set_alpha(kwargs['stream_alpha'])
        stream.arrows.set_alpha(kwargs['stream_alpha'])

    def add_bound_unbound_contour(ax, contour_quantity, panel_extents, x_min, x_max, y_min, y_max):
        if contour_quantity is None:
            return
        for block_num in range(num_blocks_used):
            block_extent = panel_extents[block_num]
            if (block_extent[1] < x_min or block_extent[0] > x_max
                    or block_extent[3] < y_min or block_extent[2] > y_max):
                continue
            block_quantity = np.asarray(contour_quantity[block_num], dtype=np.float64)
            if np.all(~np.isfinite(block_quantity)):
                continue
            qmin = np.nanmin(block_quantity)
            qmax = np.nanmax(block_quantity)
            if not np.isfinite(qmin) or not np.isfinite(qmax) or qmin > 0.0 or qmax < 0.0:
                continue
            nx = block_quantity.shape[1]
            ny = block_quantity.shape[0]
            x = np.linspace(block_extent[0], block_extent[1], nx)
            y = np.linspace(block_extent[2], block_extent[3], ny)
            ax.contour(
                x, y, block_quantity, levels=[0.0],
                colors=[kwargs['bound_unbound_color']],
                linewidths=kwargs['bound_unbound_linewidth'],
                alpha=kwargs['bound_unbound_alpha'],
                zorder=5,
            )

    def add_zoom_inset(ax, panel_quantity, panel_extents, norm, vmin, vmax, center_x,
                       center_y, zoom_r, inset_rect):
        if zoom_r is None:
            return
        x_min = center_x - zoom_r
        x_max = center_x + zoom_r
        y_min = center_y - zoom_r
        y_max = center_y + zoom_r
        inset_blocks = []
        for block_num in range(num_blocks_used):
            x0, x1, y0, y1 = panel_extents[block_num]
            if x1 < x_min or x0 > x_max or y1 < y_min or y0 > y_max:
                continue
            inset_blocks.append(block_num)
        inset = ax.inset_axes(inset_rect)
        inset.set_xlim((x_min, x_max))
        inset.set_ylim((y_min, y_max))
        for block_num in inset_blocks:
            imshow_kwargs = {
                'cmap': kwargs['cmap'],
                'interpolation': 'none',
                'origin': 'lower',
                'extent': panel_extents[block_num],
            }
            if norm is not None:
                imshow_kwargs['norm'] = norm
            else:
                imshow_kwargs['vmin'] = vmin
                imshow_kwargs['vmax'] = vmax
            inset.imshow(panel_quantity[block_num], **imshow_kwargs)
        add_streamlines(
            inset, stream_u, stream_v, x_min, x_max, y_min, y_max,
            arrowsize=0.6 * kwargs['stream_arrowsize'])
        add_bound_unbound_contour(inset, bound_unbound_quantity, panel_extents,
                                  x_min, x_max, y_min, y_max)
        if kwargs['grid']:
            for block_num in inset_blocks:
                x0 = panel_extents[block_num][0]
                y0 = panel_extents[block_num][2]
                width = panel_extents[block_num][1] - panel_extents[block_num][0]
                height = panel_extents[block_num][3] - panel_extents[block_num][2]
                box = patches.Rectangle((x0, y0), width, height,
                                        linestyle=grid_line_style,
                                        linewidth=grid_line_width,
                                        facecolor='none',
                                        edgecolor=kwargs['grid_color'],
                                        alpha=kwargs['grid_alpha'])
                inset.add_artist(box)
        add_bh_overlay(inset)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        locator_patch, connector1, connector2 = mark_inset(
            ax, inset, loc1=2, loc2=4, fc='none', ec='black', lw=0.9, alpha=0.9)
        locator_patch.set_zorder(6)
        connector1.set_zorder(6)
        connector2.set_zorder(6)

    def add_bh_zoom_inset(ax, panel_quantity, panel_extents, norm, vmin, vmax):
        if kwargs['bh_zoom_rmax'] is None:
            return
        bh_coords = get_bh_plot_coordinates()
        if bh_coords is None:
            return
        plot_x, plot_y, _ = bh_coords
        add_zoom_inset(ax, panel_quantity, panel_extents, norm, vmin, vmax, plot_x,
                       plot_y, kwargs['bh_zoom_rmax'], [0.46, -0.09, 0.51, 0.51])

    def add_center_zoom_inset(ax, panel_quantity, panel_extents, norm, vmin, vmax):
        if kwargs['center_zoom_rmax'] is None:
            return
        add_zoom_inset(ax, panel_quantity, panel_extents, norm, vmin, vmax, 0.0, 0.0,
                       kwargs['center_zoom_rmax'], [0.07, 0.69, 0.22, 0.22])

    def resolve_plot_bounds():
        if kwargs['dimension'] == 'x':
            x1_min = float(input_data['mesh']['x2min'])
            x1_max = float(input_data['mesh']['x2max'])
            x2_min = float(input_data['mesh']['x3min'])
            x2_max = float(input_data['mesh']['x3max'])
            x_label = '$y$'
            y_label = '$z$'
        if kwargs['dimension'] == 'y':
            x1_min = float(input_data['mesh']['x1min'])
            x1_max = float(input_data['mesh']['x1max'])
            x2_min = float(input_data['mesh']['x3min'])
            x2_max = float(input_data['mesh']['x3max'])
            x_label = '$x$'
            y_label = '$z$'
        if kwargs['dimension'] == 'z':
            x1_min = float(input_data['mesh']['x1min'])
            x1_max = float(input_data['mesh']['x1max'])
            x2_min = float(input_data['mesh']['x2min'])
            x2_max = float(input_data['mesh']['x2max'])
            x_label = '$x$'
            y_label = '$y$'
        if kwargs['x1_min'] is not None:
            x1_min = kwargs['x1_min']
        if kwargs['x1_max'] is not None:
            x1_max = kwargs['x1_max']
        if kwargs['x2_min'] is not None:
            x2_min = kwargs['x2_min']
        if kwargs['x2_max'] is not None:
            x2_max = kwargs['x2_max']
        if kwargs['r_max'] is not None:
            x1_min = -kwargs['r_max']
            x1_max = kwargs['r_max']
            x2_min = -kwargs['r_max']
            x2_max = kwargs['r_max']
        if kwargs['bh_center_rmax'] is not None:
            bh_coords = get_bh_plot_coordinates()
            if bh_coords is None:
                raise RuntimeError('Cannot use --bh_center_rmax without live BH metadata.')
            plot_x, plot_y, _ = bh_coords
            x1_min = plot_x - kwargs['bh_center_rmax']
            x1_max = plot_x + kwargs['bh_center_rmax']
            x2_min = plot_y - kwargs['bh_center_rmax']
            x2_max = plot_y + kwargs['bh_center_rmax']
        return x1_min, x1_max, x2_min, x2_max, x_label, y_label

    stream_u, stream_v = get_streamline_components()

    # Plot velocity divergence in one figure
    if kwargs['variable'] == 'derived:div_v_panels':
        if kwargs['norm'] not in (None, 'linear', 'symlog'):
            raise RuntimeError('"derived:div_v_panels" only supports linear or symlog normalization.')
        if kwargs['norm'] is None:
            kwargs['norm'] = 'symlog'

        labels = set_labels(general_rel_v)
        panel_quantities = divergence_panels
        panel_labels = (labels.get('div_v', 'div_v'),)
        x1_min, x1_max, x2_min, x2_max, x_label, y_label = resolve_plot_bounds()
        coord_x, coord_y, coord_z = xyz(num_blocks_used, block_nx1, block_nx2, extents,
                                        kwargs['dimension'], kwargs['location'])
        if kwargs['dimension'] == 'x':
            plot_coord_x, plot_coord_y = coord_y, coord_z
        elif kwargs['dimension'] == 'y':
            plot_coord_x, plot_coord_y = coord_x, coord_z
        else:
            plot_coord_x, plot_coord_y = coord_x, coord_y
        visible_mask = ((plot_coord_x >= x1_min) & (plot_coord_x <= x1_max)
                        & (plot_coord_y >= x2_min) & (plot_coord_y <= x2_max))

        def masked_panel_for_limits(panel_quantity):
            panel_masked = np.where(visible_mask, panel_quantity, np.nan)
            if kwargs['horizon_mask']:
                a2 = bh_a ** 2
                r_hor = 1.0 + (1.0 - a2) ** 0.5
                rr2 = coord_x ** 2 + coord_y ** 2 + coord_z ** 2
                r = np.sqrt(0.5 * (rr2 - a2
                                    + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * coord_z ** 2)))
                panel_masked = np.where(r > r_hor, panel_masked, np.nan)
            return panel_masked

        def div_panel_symlog_settings():
            linthresh = kwargs['left_linthresh']
            vmax = kwargs['left_vmax']
            if linthresh is None:
                linthresh = kwargs['linthresh'] if kwargs['linthresh'] is not None else 5.0
            if vmax is None:
                vmax = kwargs['vmax'] if kwargs['vmax'] is not None else 30.0
            return linthresh, vmax

        def symmetric_norm_for_panel(panel_quantity):
            panel_masked = masked_panel_for_limits(panel_quantity)
            if kwargs['norm'] == 'symlog':
                finite = panel_masked[np.isfinite(panel_masked)]
                linthresh, explicit_vmax = div_panel_symlog_settings()
                if finite.size == 0:
                    vabs = 1.0
                elif explicit_vmax is not None:
                    vabs = explicit_vmax
                else:
                    vabs = np.nanmax(np.abs(finite))
                    if not np.isfinite(vabs) or vabs == 0.0:
                        vabs = 1.0
                return colors.SymLogNorm(linthresh=linthresh, vmin=-vabs, vmax=vabs)
            if kwargs['vmin'] is None or kwargs['vmax'] is None:
                finite = panel_masked[np.isfinite(panel_masked)]
                if finite.size == 0:
                    vabs = 1.0
                else:
                    vabs = np.nanmax(np.abs(finite))
                    if not np.isfinite(vabs) or vabs == 0.0:
                        vabs = 1.0
                panel_vmin = -vabs if kwargs['vmin'] is None else kwargs['vmin']
                panel_vmax = vabs if kwargs['vmax'] is None else kwargs['vmax']
            else:
                panel_vmin = kwargs['vmin']
                panel_vmax = kwargs['vmax']
            if panel_vmin < 0.0 < panel_vmax:
                return colors.TwoSlopeNorm(vcenter=0.0, vmin=panel_vmin, vmax=panel_vmax)
            return colors.Normalize(panel_vmin, panel_vmax)

        def set_explicit_symlog_colorbar_ticks(cbar, linthresh, vmax):
            if kwargs['norm'] != 'symlog' or vmax is None:
                return
            vabs = abs(float(vmax))
            linthresh = abs(float(linthresh))
            if not np.isfinite(vabs) or not np.isfinite(linthresh) or vabs <= 0.0:
                return
            positive_ticks = []
            if 0.0 < linthresh < vabs:
                positive_ticks.append(linthresh)
                min_power = int(np.ceil(np.log10(linthresh)))
                max_power = int(np.floor(np.log10(vabs)))
                for power in range(min_power, max_power + 1):
                    tick = 10.0 ** power
                    if linthresh < tick < vabs:
                        positive_ticks.append(tick)
            positive_ticks.append(vabs)
            positive_ticks = sorted(set(positive_ticks))
            ticks = [-tick for tick in reversed(positive_ticks)] + [0.0] + positive_ticks
            cbar.set_ticks(ticks)
            cbar.ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(lambda value, _pos: format_colorbar_tick(value))
            )

        panel_norms = [symmetric_norm_for_panel(panel_quantity)
                       for panel_quantity in panel_quantities]

        fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.0))
        axes = [ax]
        for panel_index, (ax, panel_quantity, panel_label, panel_norm) in enumerate(
                zip(axes, panel_quantities, panel_labels, panel_norms)):
            mappable = None
            for block_num in range(num_blocks_used):
                mappable = ax.imshow(panel_quantity[block_num], cmap=kwargs['cmap'],
                                     norm=panel_norm, interpolation='none', origin='lower',
                                     extent=extents[block_num])
            add_streamlines(ax, stream_u, stream_v, x1_min, x1_max, x2_min, x2_max)
            add_bound_unbound_contour(ax, bound_unbound_quantity, extents,
                                      x1_min, x1_max, x2_min, x2_max)
            if kwargs['grid']:
                for block_num in range(num_blocks_used):
                    x0 = extents[block_num][0]
                    y0 = extents[block_num][2]
                    width = extents[block_num][1] - extents[block_num][0]
                    height = extents[block_num][3] - extents[block_num][2]
                    box = patches.Rectangle((x0, y0), width, height,
                                            linestyle=grid_line_style,
                                            linewidth=grid_line_width,
                                            facecolor='none',
                                            edgecolor=kwargs['grid_color'],
                                            alpha=kwargs['grid_alpha'])
                    ax.add_artist(box)
            add_bh_overlay(ax)
            add_bh_zoom_inset(ax, panel_quantity, extents, panel_norm, None, None)
            add_center_zoom_inset(ax, panel_quantity, extents, panel_norm, None, None)
            ax.set_title(panel_label)
            ax.set_xlim((x1_min, x1_max))
            ax.set_ylim((x2_min, x2_max))
            ax.set_xlabel(x_label, labelpad=x1_labelpad)
            ax.set_ylabel(y_label, labelpad=x2_labelpad)
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.02, extend='both')
            cbar.ax.tick_params(labelsize=13)
            cbar.set_label(panel_label)
            linthresh, explicit_vmax = div_panel_symlog_settings()
            set_explicit_symlog_colorbar_ticks(cbar, linthresh, explicit_vmax)

        fig.subplots_adjust(left=0.12, right=0.90, bottom=0.14, top=0.90)
        if kwargs['output_file'] != 'show':
            fig.savefig(kwargs['output_file'], dpi=kwargs['dpi'],
                        bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()
        return

    # Plot Cartesian velocity components in one shared figure
    if kwargs['variable'] == 'derived:vel_xyz':
        if kwargs['norm'] not in (None, 'linear'):
            raise RuntimeError('"derived:vel_xyz" only supports linear normalization.')

        labels = set_labels(general_rel_v)
        panel_names = ('velx', 'vely', 'velz')
        panel_labels = [labels.get(name, name) for name in panel_names]

        if kwargs['horizon_mask']:
            a2 = bh_a ** 2
            r_hor = 1.0 + (1.0 - a2) ** 0.5
            x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                          kwargs['location'])
            rr2 = x ** 2 + y ** 2 + z ** 2
            r = np.sqrt(0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z ** 2)))
            velocity_panels_masked = [np.where(r > r_hor, panel, np.nan)
                                      for panel in velocity_panels]
        else:
            velocity_panels_masked = velocity_panels

        if kwargs['vmin'] is None or kwargs['vmax'] is None:
            panel_stack = np.stack([np.where(np.isfinite(panel), panel, np.nan)
                                    for panel in velocity_panels_masked])
            vabs = np.nanmax(np.abs(panel_stack))
            if not np.isfinite(vabs) or vabs == 0.0:
                vabs = 1.0
            vmin = -vabs if kwargs['vmin'] is None else kwargs['vmin']
            vmax = vabs if kwargs['vmax'] is None else kwargs['vmax']
        else:
            vmin = kwargs['vmin']
            vmax = kwargs['vmax']
        norm = colors.Normalize(vmin, vmax)

        x1_min, x1_max, x2_min, x2_max, x_label, y_label = resolve_plot_bounds()

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharex=True, sharey=True)
        mappable = None
        for panel_index, (ax, panel_quantity, panel_label) in enumerate(
                zip(axes, velocity_panels, panel_labels)):
            for block_num in range(num_blocks_used):
                mappable = ax.imshow(panel_quantity[block_num], cmap=kwargs['cmap'], norm=norm,
                                     interpolation='none', origin='lower',
                                     extent=extents[block_num])

            if kwargs['grid']:
                for block_num in range(num_blocks_used):
                    x0 = extents[block_num][0]
                    y0 = extents[block_num][2]
                    width = extents[block_num][1] - extents[block_num][0]
                    height = extents[block_num][3] - extents[block_num][2]
                    box = patches.Rectangle((x0, y0), width, height,
                                            linestyle=grid_line_style,
                                            linewidth=grid_line_width,
                                            facecolor='none',
                                            edgecolor=kwargs['grid_color'],
                                            alpha=kwargs['grid_alpha'])
                    ax.add_artist(box)

            if kwargs['horizon'] or kwargs['horizon_mask']:
                r_hor = 1.0 + (1.0 - bh_a ** 2) ** 0.5
                if kwargs['dimension'] in ('x', 'y') \
                        and kwargs['location'] ** 2 < r_hor ** 2 + bh_a ** 2:
                    full_width = 2.0 * (r_hor ** 2 + bh_a ** 2
                                        - kwargs['location'] ** 2) ** 0.5
                    full_height = 2.0 * ((r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2)
                                         / (1.0 + bh_a ** 2 / r_hor ** 2)) ** 0.5
                    if kwargs['horizon_mask']:
                        horizon_mask = patches.Ellipse(
                            (0.0, 0.0), full_width, full_height,
                            facecolor=kwargs['horizon_mask_color'], edgecolor='none')
                        ax.add_artist(horizon_mask)
                    if kwargs['horizon']:
                        horizon = patches.Ellipse(
                            (0.0, 0.0), full_width, full_height,
                            linestyle=horizon_line_style, linewidth=horizon_line_width,
                            facecolor='none', edgecolor=kwargs['horizon_color'])
                        ax.add_artist(horizon)
                if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
                    radius = ((r_hor ** 2 + bh_a ** 2)
                              * (1.0 - kwargs['location'] ** 2 / r_hor ** 2)) ** 0.5
                    if kwargs['horizon_mask']:
                        horizon_mask = patches.Circle(
                            (0.0, 0.0), radius=radius,
                            facecolor=kwargs['horizon_mask_color'], edgecolor='none')
                        ax.add_artist(horizon_mask)
                    if kwargs['horizon']:
                        horizon = patches.Circle(
                            (0.0, 0.0), radius=radius,
                            linestyle=horizon_line_style, linewidth=horizon_line_width,
                            facecolor='none', edgecolor=kwargs['horizon_color'])
                        ax.add_artist(horizon)

            add_bh_overlay(ax)
            add_bh_zoom_inset(ax, panel_quantity, extents, norm, vmin, vmax)
            add_center_zoom_inset(ax, panel_quantity, extents, norm, vmin, vmax)

            if kwargs['ergosphere']:
                r_hor = 1.0 + (1.0 - bh_a ** 2) ** 0.5
                if kwargs['dimension'] in ('x', 'y') \
                        and kwargs['location'] ** 2 < 4.0 + bh_a ** 2:
                    w = np.linspace(abs(kwargs['location']), (4.0 + bh_a ** 2) ** 0.5,
                                    ergosphere_num_points)
                    z = np.empty_like(w)
                    for ind, w_val in enumerate(w):
                        def residual_hor(z_val):
                            rr2 = w_val ** 2 + z_val ** 2
                            r2 = 0.5 * (rr2 - bh_a ** 2 + ((rr2 - bh_a ** 2) ** 2
                                        + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                            return r2 - r_hor ** 2
                        if residual_hor(0.0) < 0.0:
                            z_min = brentq(residual_hor, 0.0, 2.0)
                        else:
                            z_min = 0.0

                        def residual_ergo(z_val):
                            rr2 = w_val ** 2 + z_val ** 2
                            r2 = 0.5 * (rr2 - bh_a ** 2 + ((rr2 - bh_a ** 2) ** 2
                                        + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                            return r2 ** 2 - 2.0 * r2 ** 1.5 + bh_a ** 2 * z_val ** 2
                        if residual_ergo(z_min) <= 0.0:
                            z[ind] = brentq(residual_ergo, z_min, 2.0)
                        else:
                            z[ind] = 0.0
                    xy_plot = np.sqrt(w ** 2 - kwargs['location'] ** 2)
                    xy_plot = np.concatenate((-xy_plot[::-1], xy_plot))
                    xy_plot = np.concatenate((xy_plot, xy_plot[::-1]))
                    z_plot = np.concatenate((z[::-1], z))
                    z_plot = np.concatenate((z_plot, -z_plot[::-1]))
                    ax.plot(xy_plot, z_plot, linestyle=ergosphere_line_style,
                            linewidth=ergosphere_line_width,
                            color=kwargs['ergosphere_color'], zorder=0)
                if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
                    def residual_ergo(r):
                        return r ** 4 - 2.0 * r ** 3 + bh_a ** 2 * kwargs['location'] ** 2
                    r_ergo = brentq(residual_ergo, r_hor, 2.0)
                    radius = ((r_ergo ** 2 + bh_a ** 2)
                              * (1.0 - kwargs['location'] ** 2 / r_ergo ** 2)) ** 0.5
                    ergosphere = patches.Circle(
                        (0.0, 0.0), radius=radius,
                        linestyle=ergosphere_line_style, linewidth=ergosphere_line_width,
                        facecolor='none', edgecolor=kwargs['ergosphere_color'])
                    ax.add_artist(ergosphere)

            add_bh_overlay(ax)

            ax.set_title(panel_label)
            ax.set_xlim((x1_min, x1_max))
            ax.set_ylim((x2_min, x2_max))
            ax.set_xlabel(x_label, labelpad=x1_labelpad)
            if panel_index == 0:
                ax.set_ylabel(y_label, labelpad=x2_labelpad)

        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.92, wspace=0.08)
        cax = fig.add_axes([0.895, 0.15, 0.028, 0.70])
        make_colorbar(fig, mappable, cax, labels.get('vel_xyz', 'Velocity'))

        if kwargs['output_file'] != 'show':
            fig.savefig(kwargs['output_file'], dpi=kwargs['dpi'],
                        bbox_inches='tight', pad_inches=0.02)
        else:
            plt.show()
        return

    # Mask horizon for purposes of calculating colorbar limits
    if kwargs['horizon_mask']:
        a2 = bh_a ** 2
        r_hor = 1.0 + (1.0 - a2) ** 0.5
        x, y, z = xyz(num_blocks_used, block_nx1, block_nx2, extents, kwargs['dimension'],
                      kwargs['location'])
        rr2 = x ** 2 + y ** 2 + z ** 2
        r = np.sqrt(0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z ** 2)))
        quantity_masked = np.where(r > r_hor, quantity, np.nan)
    else:
        quantity_masked = quantity

    # Calculate color scaling
    if kwargs['vmin'] is None:
        if kwargs['norm'] == 'linear':
            vmin = np.nanmin(np.where(quantity_masked > -np.inf, quantity_masked, np.nan))
        elif kwargs['norm'] == 'log':
            vmin = np.nanmin(np.where(quantity_masked > 0.0, quantity_masked, np.nan))
        else:
            vmin = np.nanmin(quantity_masked)
    else:
        vmin = kwargs['vmin']
    if kwargs['vmax'] is None:
        if kwargs['norm'] in ('linear', 'log'):
            vmax = np.nanmax(np.where(quantity_masked < np.inf, quantity_masked, np.nan))
        else:
            vmax = np.nanmax(quantity_masked)
    else:
        vmax = kwargs['vmax']
    if kwargs['norm'] == 'linear':
        norm = colors.Normalize(vmin, vmax)
        vmin = None
        vmax = None
    elif kwargs['norm'] == 'log':
        norm = colors.LogNorm(vmin, vmax)
        vmin = None
        vmax = None
    else:
        norm = kwargs['norm']

    # Set colorbar label
    labels = set_labels(general_rel_v)
    if variable_name in labels:
        label = labels[variable_name]
    else:
        label = variable_name

    x1_min, x1_max, x2_min, x2_max, x_label, y_label = resolve_plot_bounds()

    def render_panel(ax, panel_norm, panel_vmin, panel_vmax, x_limits=None, y_limits=None,
                     title=None, show_xlabel=True, show_ylabel=True, ylabel=None,
                     show_bh_zoom=True, show_center_zoom=True):
        current_x_limits = (x1_min, x1_max) if x_limits is None else x_limits
        current_y_limits = (x2_min, x2_max) if y_limits is None else y_limits
        mappable = None
        for block_num in range(num_blocks_used):
            mappable = ax.imshow(quantity[block_num], cmap=kwargs['cmap'], norm=panel_norm,
                                 vmin=panel_vmin, vmax=panel_vmax,
                                 interpolation='none', origin='lower',
                                 extent=extents[block_num])
        add_streamlines(ax, stream_u, stream_v, current_x_limits[0], current_x_limits[1],
                        current_y_limits[0], current_y_limits[1])
        add_bound_unbound_contour(ax, bound_unbound_quantity, extents, current_x_limits[0],
                                  current_x_limits[1], current_y_limits[0],
                                  current_y_limits[1])

        if kwargs['grid']:
            for block_num in range(num_blocks_used):
                x0 = extents[block_num][0]
                y0 = extents[block_num][2]
                width = extents[block_num][1] - extents[block_num][0]
                height = extents[block_num][3] - extents[block_num][2]
                box = patches.Rectangle((x0, y0), width, height,
                                        linestyle=grid_line_style,
                                        linewidth=grid_line_width,
                                        facecolor='none',
                                        edgecolor=kwargs['grid_color'],
                                        alpha=kwargs['grid_alpha'])
                ax.add_artist(box)

        if kwargs['horizon'] or kwargs['horizon_mask']:
            r_hor = 1.0 + (1.0 - bh_a ** 2) ** 0.5
            if kwargs['dimension'] in ('x', 'y') \
                    and kwargs['location'] ** 2 < r_hor ** 2 + bh_a ** 2:
                full_width = 2.0 * (r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2) ** 0.5
                full_height = 2.0 * ((r_hor ** 2 + bh_a ** 2 - kwargs['location'] ** 2)
                                     / (1.0 + bh_a ** 2 / r_hor ** 2)) ** 0.5
                if kwargs['horizon_mask']:
                    horizon_mask = patches.Ellipse((0.0, 0.0), full_width, full_height,
                                                   facecolor=kwargs['horizon_mask_color'],
                                                   edgecolor='none')
                    ax.add_artist(horizon_mask)
                if kwargs['horizon']:
                    horizon = patches.Ellipse((0.0, 0.0), full_width, full_height,
                                              linestyle=horizon_line_style,
                                              linewidth=horizon_line_width,
                                              facecolor='none',
                                              edgecolor=kwargs['horizon_color'])
                    ax.add_artist(horizon)
            if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
                radius = ((r_hor ** 2 + bh_a ** 2)
                          * (1.0 - kwargs['location'] ** 2 / r_hor ** 2)) ** 0.5
                if kwargs['horizon_mask']:
                    horizon_mask = patches.Circle((0.0, 0.0), radius=radius,
                                                  facecolor=kwargs['horizon_mask_color'],
                                                  edgecolor='none')
                    ax.add_artist(horizon_mask)
                if kwargs['horizon']:
                    horizon = patches.Circle((0.0, 0.0), radius=radius,
                                             linestyle=horizon_line_style,
                                             linewidth=horizon_line_width,
                                             facecolor='none',
                                             edgecolor=kwargs['horizon_color'])
                    ax.add_artist(horizon)

        if kwargs['ergosphere']:
            r_hor = 1.0 + (1.0 - bh_a ** 2) ** 0.5
            if kwargs['dimension'] in ('x', 'y') \
                    and kwargs['location'] ** 2 < 4.0 + bh_a ** 2:
                w = np.linspace(abs(kwargs['location']), (4.0 + bh_a ** 2) ** 0.5,
                                ergosphere_num_points)
                z = np.empty_like(w)
                for ind, w_val in enumerate(w):
                    def residual_hor(z_val):
                        rr2 = w_val ** 2 + z_val ** 2
                        r2 = 0.5 * (rr2 - bh_a ** 2 + ((rr2 - bh_a ** 2) ** 2
                                    + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                        return r2 - r_hor ** 2
                    if residual_hor(0.0) < 0.0:
                        z_min = brentq(residual_hor, 0.0, 2.0)
                    else:
                        z_min = 0.0

                    def residual_ergo(z_val):
                        rr2 = w_val ** 2 + z_val ** 2
                        r2 = 0.5 * (rr2 - bh_a ** 2 + ((rr2 - bh_a ** 2) ** 2
                                    + 4.0 * bh_a ** 2 * z_val ** 2) ** 0.5)
                        return r2 ** 2 - 2.0 * r2 ** 1.5 + bh_a ** 2 * z_val ** 2
                    if residual_ergo(z_min) <= 0.0:
                        z[ind] = brentq(residual_ergo, z_min, 2.0)
                    else:
                        z[ind] = 0.0
                xy_plot = np.sqrt(w ** 2 - kwargs['location'] ** 2)
                xy_plot = np.concatenate((-xy_plot[::-1], xy_plot))
                xy_plot = np.concatenate((xy_plot, xy_plot[::-1]))
                z_plot = np.concatenate((z[::-1], z))
                z_plot = np.concatenate((z_plot, -z_plot[::-1]))
                ax.plot(xy_plot, z_plot, linestyle=ergosphere_line_style,
                        linewidth=ergosphere_line_width, color=kwargs['ergosphere_color'],
                        zorder=0)
            if kwargs['dimension'] == 'z' and abs(kwargs['location']) < r_hor:
                def residual_ergo(r):
                    return r ** 4 - 2.0 * r ** 3 + bh_a ** 2 * kwargs['location'] ** 2
                r_ergo = brentq(residual_ergo, r_hor, 2.0)
                radius = ((r_ergo ** 2 + bh_a ** 2)
                          * (1.0 - kwargs['location'] ** 2 / r_ergo ** 2)) ** 0.5
                ergosphere = patches.Circle((0.0, 0.0), radius=radius,
                                            linestyle=ergosphere_line_style,
                                            linewidth=ergosphere_line_width,
                                            facecolor='none',
                                            edgecolor=kwargs['ergosphere_color'])
                ax.add_artist(ergosphere)

        add_bh_overlay(ax)
        if show_bh_zoom:
            add_bh_zoom_inset(ax, quantity, extents, panel_norm, panel_vmin, panel_vmax)
        if show_center_zoom:
            add_center_zoom_inset(ax, quantity, extents, panel_norm, panel_vmin, panel_vmax)
        ax.set_xlim(current_x_limits)
        ax.set_ylim(current_y_limits)
        if title is not None:
            ax.set_title(title)
        if show_xlabel:
            ax.set_xlabel(x_label, labelpad=x1_labelpad)
        else:
            ax.set_xlabel('')
        if show_ylabel:
            ax.set_ylabel(y_label if ylabel is None else ylabel, labelpad=x2_labelpad)
        else:
            ax.set_ylabel('')
        return mappable

    if panel_state_only:
        return {
            'render': render_panel,
            'quantity_masked': quantity_masked,
            'label': label,
            'x_label': x_label,
            'y_label': y_label,
            'x_limits': (x1_min, x1_max),
            'y_limits': (x2_min, x2_max),
        }

    # Size the axes to the plotted data aspect so the colorbar can sit tight to the panel.
    data_width = max(x1_max - x1_min, 1.0e-12)
    data_height = max(x2_max - x2_min, 1.0e-12)
    fig_height = 12.0
    panel_bottom_in = 0.75
    panel_top_in = 0.40
    panel_height_in = fig_height - panel_bottom_in - panel_top_in
    panel_width_in = panel_height_in * data_width / data_height
    left_margin_in = 0.85
    cbar_gap_in = 0.55
    cbar_width_in = 0.42
    right_margin_in = 0.80
    fig_width = left_margin_in + panel_width_in + cbar_gap_in + cbar_width_in + right_margin_in

    fig = plt.figure(figsize=(fig_width, fig_height))
    ax = fig.add_axes([left_margin_in / fig_width,
                       panel_bottom_in / fig_height,
                       panel_width_in / fig_width,
                       panel_height_in / fig_height])
    mappable = render_panel(ax, norm, vmin, vmax)

    # Make colorbar
    cax = fig.add_axes([(left_margin_in + panel_width_in + cbar_gap_in) / fig_width,
                        panel_bottom_in / fig_height,
                        cbar_width_in / fig_width,
                        panel_height_in / fig_height])
    make_colorbar(
        fig,
        mappable,
        cax,
        label,
        plot_variable_name=variable_name,
        plot_density_unit_cgs=density_unit_cgs,
    )

    # Save or display figure
    if kwargs['output_file'] != 'show':
        fig.savefig(kwargs['output_file'], dpi=kwargs['dpi'],
                    bbox_inches='tight', pad_inches=0.02)
    else:
        plt.show()


# Function that defines dependencies for derived quantities
def set_derived_dependencies():
    derived_dependencies = {}
    derived_dependencies['pgas'] = ('dens', 'eint')
    names = ('pgas_rho', 'entropy', 'c_s', 'T', 'xh2', 'xion', 'xhe1', 'xhe2', 'gamma1',
             'gamma3m1', 'mu', 'beta_rad')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint')
    derived_dependencies['grav_phi_abs'] = ('grav_phi',)
    derived_dependencies['vel_xyz'] = ('velx', 'vely', 'velz')
    derived_dependencies['vel_norm'] = ('velx', 'vely', 'velz')
    derived_dependencies['mach'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['div_v'] = ('velx', 'vely', 'velz')
    derived_dependencies['div_v_cs'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['div_v_panels'] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['prad_pgas'] = ('dens', 'eint', 'r00_ff')
    names = ('vr_nr', 'vth_nr', 'vph_nr', 'uut', 'ut', 'ux', 'uy', 'uz', 'ur', 'uth',
             'uph', 'u_t', 'u_x', 'u_y', 'u_z', 'u_r', 'u_th', 'u_ph', 'vx', 'vy', 'vz',
             'vr_rel', 'vth_rel', 'vph_rel')
    for name in names:
        derived_dependencies[name] = ('velx', 'vely', 'velz')
    names = ('Br_nr', 'Bth_nr', 'Bph_nr', 'pmag_nr', 'cons_em_nr_t')
    for name in names:
        derived_dependencies[name] = ('bcc1', 'bcc2', 'bcc3')
    derived_dependencies['beta_inv_nr'] = ('eint', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['sigma_nr'] = ('dens', 'bcc1', 'bcc2', 'bcc3')
    names = ('bt', 'bx', 'by', 'bz', 'br', 'bth', 'bph', 'b_t', 'b_x', 'b_y', 'b_z',
             'b_r', 'b_th', 'b_ph', 'Br_rel', 'Bth_rel', 'Bph_rel', 'pmag_rel',
             'cons_em_rel_t', 'cons_em_rel_x', 'cons_em_rel_y', 'cons_em_rel_z')
    for name in names:
        derived_dependencies[name] = ('velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3')
    derived_dependencies['beta_inv_rel'] = ('eint', 'velx', 'vely', 'velz', 'bcc1',
                                            'bcc2', 'bcc3')
    derived_dependencies['sigma_rel'] = ('dens', 'velx', 'vely', 'velz', 'bcc1', 'bcc2',
                                         'bcc3')
    names = ('sigmah_rel', 'va_rel')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz', 'bcc1',
                                      'bcc2', 'bcc3')
    derived_dependencies['pmag_prad'] = ('velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3',
                                         'r00_ff')
    derived_dependencies['prad'] = ('r00_ff',)
    names = ('Rtr', 'Rtth', 'Rtph')
    for name in names:
        derived_dependencies[name] = ('r01', 'r02', 'r03')
    names = ('Rrr', 'Rthth', 'Rphph', 'Rrth', 'Rrph', 'Rthph')
    for name in names:
        derived_dependencies[name] = ('r11', 'r12', 'r13', 'r22', 'r23', 'r33')
    derived_dependencies['Rtx_Rtt'] = ('r00', 'r01')
    derived_dependencies['Rty_Rtt'] = ('r00', 'r02')
    derived_dependencies['Rtz_Rtt'] = ('r00', 'r03')
    derived_dependencies['Rxx_Rtt'] = ('r00', 'r11')
    derived_dependencies['Ryy_Rtt'] = ('r00', 'r22')
    derived_dependencies['Rzz_Rtt'] = ('r00', 'r33')
    derived_dependencies['Rxy_Rtt'] = ('r00', 'r12')
    derived_dependencies['Rxz_Rtt'] = ('r00', 'r13')
    derived_dependencies['Ryz_Rtt'] = ('r00', 'r23')
    names = ('Rtr_Rtt', 'Rtth_Rtt', 'Rtph_Rtt')
    for name in names:
        derived_dependencies[name] = ('r00', 'r01', 'r02', 'r03')
    names = ('Rrr_Rtt', 'Rthth_Rtt', 'Rphph_Rtt', 'Rrth_Rtt', 'Rrph_Rtt', 'Rthph_Rtt')
    for name in names:
        derived_dependencies[name] = ('r00', 'r11', 'r12', 'r13', 'r22', 'r23', 'r33')
    derived_dependencies['R01_R00_ff'] = ('r00_ff', 'r01_ff')
    derived_dependencies['R02_R00_ff'] = ('r00_ff', 'r02_ff')
    derived_dependencies['R03_R00_ff'] = ('r00_ff', 'r03_ff')
    derived_dependencies['R11_R00_ff'] = ('r00_ff', 'r11_ff')
    derived_dependencies['R22_R00_ff'] = ('r00_ff', 'r22_ff')
    derived_dependencies['R33_R00_ff'] = ('r00_ff', 'r33_ff')
    derived_dependencies['R12_R00_ff'] = ('r00_ff', 'r12_ff')
    derived_dependencies['R13_R00_ff'] = ('r00_ff', 'r13_ff')
    derived_dependencies['R23_R00_ff'] = ('r00_ff', 'r23_ff')
    names = ('kappa_a', 'kappa_t', 'alpha_a', 'alpha_t', 'tau_a', 'tau_t')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint')
    derived_dependencies['kappa_s'] = ()
    names = ('alpha_s', 'tau_s')
    for name in names:
        derived_dependencies[name] = ('dens',)
    derived_dependencies['wgas'] = ('dens', 'eint')
    derived_dependencies['wgasrad'] = ('dens', 'eint', 'r00_ff')
    names = ('wmhd', 'Bemhd', 'cons_mhd_nr_t', 'cons_mhd_rel_t', 'cons_mhd_rel_x',
             'cons_mhd_rel_y', 'cons_mhd_rel_z')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz', 'bcc1',
                                      'bcc2', 'bcc3')
    names = ('wmhdrad', 'Bemhdrad')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz', 'bcc1',
                                      'bcc2', 'bcc3', 'r00_ff')
    names = ('Begas', 'cons_hydro_nr_t', 'cons_hydro_rel_t', 'cons_hydro_rel_x',
             'cons_hydro_rel_y', 'cons_hydro_rel_z')
    for name in names:
        derived_dependencies[name] = ('dens', 'eint', 'velx', 'vely', 'velz')
    derived_dependencies['Begasrad'] = ('dens', 'eint', 'velx', 'vely', 'velz', 'r00_ff')
    names = ('cons_hydro_nr_x', 'cons_mhd_nr_x')
    for name in names:
        derived_dependencies[name] = ('dens', 'velx')
    names = ('cons_hydro_nr_y', 'cons_mhd_nr_y')
    for name in names:
        derived_dependencies[name] = ('dens', 'vely')
    names = ('cons_hydro_nr_z', 'cons_mhd_nr_z')
    for name in names:
        derived_dependencies[name] = ('dens', 'velz')
    return derived_dependencies


# Function that defines colorbar labels
def set_labels(general_rel_v):
    labels = {}
    labels['dens'] = r'$\rho$'
    labels['eint'] = r'$u_\mathrm{gas}$'
    if general_rel_v:
        labels['velx'] = r'$u^{x^\prime}$'
        labels['vely'] = r'$u^{y^\prime}$'
        labels['velz'] = r'$u^{z^\prime}$'
    else:
        labels['velx'] = '$v^x$'
        labels['vely'] = '$v^y$'
        labels['velz'] = '$v^z$'
    labels['bcc1'] = '$B^x$'
    labels['bcc2'] = '$B^y$'
    labels['bcc3'] = '$B^z$'
    labels['r00'] = '$R^{tt}$'
    labels['r01'] = '$R^{tx}$'
    labels['r02'] = '$R^{ty}$'
    labels['r03'] = '$R^{tz}$'
    labels['r11'] = '$R^{xx}$'
    labels['r12'] = '$R^{xy}$'
    labels['r13'] = '$R^{xz}$'
    labels['r22'] = '$R^{yy}$'
    labels['r23'] = '$R^{yz}$'
    labels['r33'] = '$R^{zz}$'
    labels['r00_ff'] = r'$R^{\bar{t}\bar{t}}$'
    labels['r01_ff'] = r'$R^{\bar{t}\bar{x}}$'
    labels['r02_ff'] = r'$R^{\bar{t}\bar{y}}$'
    labels['r03_ff'] = r'$R^{\bar{t}\bar{z}}$'
    labels['r11_ff'] = r'$R^{\bar{x}\bar{x}}$'
    labels['r12_ff'] = r'$R^{\bar{x}\bar{y}}$'
    labels['r13_ff'] = r'$R^{\bar{x}\bar{z}}$'
    labels['r22_ff'] = r'$R^{\bar{y}\bar{y}}$'
    labels['r23_ff'] = r'$R^{\bar{y}\bar{z}}$'
    labels['r33_ff'] = r'$R^{\bar{z}\bar{z}}$'
    labels['pgas'] = r'$p_\mathrm{gas}$'
    labels['pgas_rho'] = r'$p_\mathrm{gas} / \rho$'
    labels['entropy'] = r'$p_\mathrm{gas} / \rho^\gamma$'
    labels['c_s'] = r'$c_\mathrm{s}$'
    labels['T'] = r'$T$ ($\mathrm{K}$)'
    labels['temperature'] = r'$T$ ($\mathrm{K}$)'
    labels['xh2'] = r'$x_{\mathrm{H}_2}$'
    labels['xion'] = r'$x_\mathrm{ion}$'
    labels['xhe1'] = r'$x_{\mathrm{He}^+}$'
    labels['xhe2'] = r'$x_{\mathrm{He}^{2+}}$'
    labels['gamma1'] = r'$\Gamma_1$'
    labels['gamma3m1'] = r'$\Gamma_3 - 1$'
    labels['mu'] = r'$\mu$'
    labels['beta_rad'] = r'$\beta_\mathrm{rad}$'
    labels['prad_pgas'] = r'$p_\mathrm{rad} / p_\mathrm{gas}$'
    labels['grav_phi_abs'] = r'$|\Phi_\mathrm{grav}|$'
    labels['vel_xyz'] = 'Velocity'
    labels['vel_norm'] = 'Velocity magnitude'
    labels['mach'] = 'Mach number'
    labels['div_v'] = r'$\nabla \cdot \mathbf{v}$'
    labels['div_v_cs'] = r'$(\nabla \cdot \mathbf{v})\,r_\mathrm{t}/c_\mathrm{s}$'
    labels['div_v_panels'] = r'$\nabla \cdot \mathbf{v}$'
    labels['vr_nr'] = r'$v^{\hat{r}}$'
    labels['vth_nr'] = r'$v^{\hat{\theta}}$'
    labels['vph_nr'] = r'$v^{\hat{\phi}}$'
    labels['uut'] = r'$u^{t^\prime}$'
    labels['ut'] = '$u^t$'
    labels['ux'] = '$u^x$'
    labels['uy'] = '$u^y$'
    labels['uz'] = '$u^z$'
    labels['ur'] = '$u^r$'
    labels['uth'] = r'$u^\theta$'
    labels['uph'] = r'$u^\phi$'
    labels['u_t'] = '$u_t$'
    labels['u_x'] = '$u_x$'
    labels['u_y'] = '$u_y$'
    labels['u_z'] = '$u_z$'
    labels['u_r'] = '$u_r$'
    labels['u_th'] = r'$u_\theta$'
    labels['u_ph'] = r'$u_\phi$'
    labels['vx'] = '$v^x$'
    labels['vy'] = '$v^y$'
    labels['vz'] = '$v^z$'
    labels['vr_rel'] = '$v^r$'
    labels['vth_rel'] = r'$v^\theta$'
    labels['vph_rel'] = r'$v^\phi$'
    labels['Br_nr'] = r'$B^{\hat{r}}$'
    labels['Bth_nr'] = r'$B^{\hat{\theta}}$'
    labels['Bph_nr'] = r'$B^{\hat{\phi}}$'
    labels['pmag_nr'] = r'$p_\mathrm{mag}$'
    labels['beta_inv_nr'] = r'$\beta^{-1}$'
    labels['sigma_nr'] = r'$\sigma$'
    labels['bt'] = '$b^t$'
    labels['bx'] = '$b^x$'
    labels['by'] = '$b^y$'
    labels['bz'] = '$b^z$'
    labels['br'] = '$b^r$'
    labels['bth'] = r'$b^\theta$'
    labels['bph'] = r'$b^\phi$'
    labels['b_t'] = '$b_t$'
    labels['b_x'] = '$b_x$'
    labels['b_y'] = '$b_y$'
    labels['b_z'] = '$b_z$'
    labels['b_r'] = '$b_r$'
    labels['b_th'] = r'$b_\theta$'
    labels['b_ph'] = r'$b_\phi$'
    labels['Br_rel'] = '$B^r$'
    labels['Bth_rel'] = r'$B^\theta$'
    labels['Bph_rel'] = r'$B^\phi$'
    labels['pmag_rel'] = r'$p_\mathrm{mag}$'
    labels['beta_inv_rel'] = r'$\beta^{-1}$'
    labels['sigma_rel'] = r'$\sigma$'
    labels['sigmah_rel'] = r'$\sigma_\mathrm{hot}$'
    labels['va_rel'] = r'$v_\mathrm{A}$'
    labels['pmag_prad'] = r'$p_\mathrm{mag} / p_\mathrm{rad}$'
    labels['prad'] = r'$p_\mathrm{rad}$'
    labels['Rtr'] = '$R^{tr}$'
    labels['Rtth'] = r'$R^{t\theta}$'
    labels['Rtph'] = r'$R^{t\phi}$'
    labels['Rrr'] = '$R^{rr}$'
    labels['Rthth'] = r'$R^{\theta\theta}$'
    labels['Rphph'] = r'$R^{\phi\phi}$'
    labels['Rrth'] = r'$R^{r\theta}$'
    labels['Rrph'] = r'$R^{r\phi}$'
    labels['Rthph'] = r'$R^{\theta\phi}$'
    labels['Rtx_Rtt'] = '$R^{tx} / R^{tt}$'
    labels['Rty_Rtt'] = '$R^{ty} / R^{tt}$'
    labels['Rtz_Rtt'] = '$R^{tz} / R^{tt}$'
    labels['Rxx_Rtt'] = '$R^{xx} / R^{tt}$'
    labels['Ryy_Rtt'] = '$R^{yy} / R^{tt}$'
    labels['Rzz_Rtt'] = '$R^{zz} / R^{tt}$'
    labels['Rxy_Rtt'] = '$R^{xy} / R^{tt}$'
    labels['Rxz_Rtt'] = '$R^{xz} / R^{tt}$'
    labels['Ryz_Rtt'] = '$R^{yz} / R^{tt}$'
    labels['Rtr_Rtt'] = '$R^{tr} / R^{tt}$'
    labels['Rtth_Rtt'] = r'$R^{t\theta} / R^{tt}$'
    labels['Rtph_Rtt'] = r'$R^{t\phi} / R^{tt}$'
    labels['Rrr_Rtt'] = '$R^{rr} / R^{tt}$'
    labels['Rthth_Rtt'] = r'$R^{\theta\theta} / R^{tt}$'
    labels['Rphph_Rtt'] = r'$R^{\phi\phi} / R^{tt}$'
    labels['Rrth_Rtt'] = r'$R^{r\theta} / R^{tt}$'
    labels['Rrph_Rtt'] = r'$R^{r\phi} / R^{tt}$'
    labels['Rthph_Rtt'] = r'$R^{\theta\phi} / R^{tt}$'
    labels['R01_R00_ff'] = r'$R^{\bar{0}\bar{1}} / R^{\bar{0}\bar{0}}$'
    labels['R02_R00_ff'] = r'$R^{\bar{0}\bar{2}} / R^{\bar{0}\bar{0}}$'
    labels['R03_R00_ff'] = r'$R^{\bar{0}\bar{3}} / R^{\bar{0}\bar{0}}$'
    labels['R11_R00_ff'] = r'$R^{\bar{1}\bar{1}} / R^{\bar{0}\bar{0}}$'
    labels['R22_R00_ff'] = r'$R^{\bar{2}\bar{2}} / R^{\bar{0}\bar{0}}$'
    labels['R33_R00_ff'] = r'$R^{\bar{3}\bar{3}} / R^{\bar{0}\bar{0}}$'
    labels['R12_R00_ff'] = r'$R^{\bar{1}\bar{2}} / R^{\bar{0}\bar{0}}$'
    labels['R13_R00_ff'] = r'$R^{\bar{1}\bar{3}} / R^{\bar{0}\bar{0}}$'
    labels['R23_R00_ff'] = r'$R^{\bar{2}\bar{3}} / R^{\bar{0}\bar{0}}$'
    labels['kappa_a'] = r'$\kappa^\mathrm{a}$ ($\mathrm{cm}^2\ \mathrm{g}^{-1}$)'
    labels['kappa_s'] = r'$\kappa^\mathrm{s}$ ($\mathrm{cm}^2\ \mathrm{g}^{-1}$)'
    labels['kappa_t'] = r'$\kappa^\mathrm{t}$ ($\mathrm{cm}^2\ \mathrm{g}^{-1}$)'
    labels['alpha_a'] = r'$\alpha^\mathrm{a}$ ($\mathrm{cm}^{-1}$)'
    labels['alpha_s'] = r'$\alpha^\mathrm{s}$ ($\mathrm{cm}^{-1}$)'
    labels['alpha_t'] = r'$\alpha^\mathrm{t}$ ($\mathrm{cm}^{-1}$)'
    labels['tau_a'] = r'$\alpha^\mathrm{a} r_\mathrm{g}$'
    labels['tau_s'] = r'$\alpha^\mathrm{s} r_\mathrm{g}$'
    labels['tau_t'] = r'$\alpha^\mathrm{t} r_\mathrm{g}$'
    labels['wgas'] = r'$w_\mathrm{gas}$'
    labels['wmhd'] = r'$w_\mathrm{MHD}$'
    labels['wgasrad'] = r'$w_\mathrm{gas+rad}$'
    labels['wmhdrad'] = r'$w_\mathrm{MHD+rad}$'
    labels['Begas'] = r'$\mathrm{Be}_\mathrm{gas}$'
    labels['Bemhd'] = r'$\mathrm{Be}_\mathrm{MHD}$'
    labels['Begasrad'] = r'$\mathrm{Be}_\mathrm{gas+rad}$'
    labels['Bemhdrad'] = r'$\mathrm{Be}_\mathrm{MHD+rad}$'
    labels['cons_hydro_nr_t'] = r'$e_\mathrm{hydro}$'
    labels['cons_hydro_nr_x'] = r'$m_\mathrm{hydro}^x$'
    labels['cons_hydro_nr_y'] = r'$m_\mathrm{hydro}^y$'
    labels['cons_hydro_nr_z'] = r'$m_\mathrm{hydro}^z$'
    labels['cons_em_nr_t'] = r'$e_\mathrm{EM}$'
    labels['cons_mhd_nr_t'] = r'$e_\mathrm{MHD}$'
    labels['cons_mhd_nr_x'] = r'$m_\mathrm{MHD}^x$'
    labels['cons_mhd_nr_y'] = r'$m_\mathrm{MHD}^y$'
    labels['cons_mhd_nr_z'] = r'$m_\mathrm{MHD}^z$'
    labels['cons_hydro_rel_t'] = r'$(T_\mathrm{hydro})^t{}_t$'
    labels['cons_hydro_rel_x'] = r'$(T_\mathrm{hydro})^t{}_x$'
    labels['cons_hydro_rel_y'] = r'$(T_\mathrm{hydro})^t{}_y$'
    labels['cons_hydro_rel_z'] = r'$(T_\mathrm{hydro})^t{}_z$'
    labels['cons_em_rel_t'] = r'$(T_\mathrm{EM})^t{}_t$'
    labels['cons_em_rel_x'] = r'$(T_\mathrm{EM})^t{}_x$'
    labels['cons_em_rel_y'] = r'$(T_\mathrm{EM})^t{}_y$'
    labels['cons_em_rel_z'] = r'$(T_\mathrm{EM})^t{}_z$'
    labels['cons_mhd_rel_t'] = r'$(T_\mathrm{MHD})^t{}_t$'
    labels['cons_mhd_rel_x'] = r'$(T_\mathrm{MHD})^t{}_x$'
    labels['cons_mhd_rel_y'] = r'$(T_\mathrm{MHD})^t{}_y$'
    labels['cons_mhd_rel_z'] = r'$(T_\mathrm{MHD})^t{}_z$'
    return labels


# Function for calculating cell coordinates
def xyz(num_blocks_used, block_nx1, block_nx2, extents, dimension, location):
    x1 = np.empty((num_blocks_used, block_nx2, block_nx1))
    x2 = np.empty((num_blocks_used, block_nx2, block_nx1))
    for block_ind in range(len(extents)):
        x1f = np.linspace(extents[block_ind][0], extents[block_ind][1], block_nx1 + 1)
        x1v = 0.5 * (x1f[:-1] + x1f[1:])
        x1[block_ind, :, :] = x1v[None, :]
        x2f = np.linspace(extents[block_ind][2], extents[block_ind][3], block_nx2 + 1)
        x2v = 0.5 * (x2f[:-1] + x2f[1:])
        x2[block_ind, :, :] = x2v[:, None]
    if dimension == 'x':
        x = np.full((num_blocks_used, block_nx2, block_nx1), location)
        y = x1
        z = x2
    if dimension == 'y':
        y = np.full((num_blocks_used, block_nx2, block_nx1), location)
        x = x1
        z = x2
    if dimension == 'z':
        z = np.full((num_blocks_used, block_nx2, block_nx1), location)
        x = x1
        y = x2
    return x, y, z


# Function for converting Cartesian coordinates to spherical
def cart_to_sph(ax, ay, az, x, y, z):
    rr = np.sqrt(x ** 2 + y ** 2)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dr_dx = x / r
    dr_dy = y / r
    dr_dz = z / r
    dth_dx = x * z / (rr * r)
    dth_dy = y * z / (rr * r)
    dth_dz = -rr / r
    dph_dx = -y / rr
    dph_dy = x / rr
    dph_dz = 0.0
    ar = dr_dx * ax + dr_dy * ay + dr_dz * az
    ath = dth_dx * ax + dth_dy * ay + dth_dz * az
    aph = dph_dx * ax + dph_dy * ay + dph_dz * az
    return ar, ath, aph


# Function for calculating quantities related to CKS metric
def cks_geometry(a, x, y, z):
    a2 = a ** 2
    z2 = z ** 2
    rr2 = x ** 2 + y ** 2 + z2
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = np.sqrt(r2)
    with warnings.catch_warnings():
        message = 'invalid value encountered in divide'
        warnings.filterwarnings('ignore', message=message, category=RuntimeWarning)
        message = 'invalid value encountered in true_divide'
        warnings.filterwarnings('ignore', message=message, category=RuntimeWarning)
        f = 2.0 * r2 * r / (r2 ** 2 + a2 * z2)
        lx = (r * x + a * y) / (r2 + a2)
        ly = (r * y - a * x) / (r2 + a2)
        lz = z / r
    gtt = -1.0 - f
    alpha2 = -1.0 / gtt
    alpha = np.sqrt(alpha2)
    betax = alpha2 * f * lx
    betay = alpha2 * f * ly
    betaz = alpha2 * f * lz
    g_tt = -1.0 + f
    g_tx = f * lx
    g_ty = f * ly
    g_tz = f * lz
    g_xx = 1.0 + f * lx ** 2
    g_xy = f * lx * ly
    g_xz = f * lx * lz
    g_yy = 1.0 + f * ly ** 2
    g_yz = f * ly * lz
    g_zz = 1.0 + f * lz ** 2
    return alpha, betax, betay, betaz, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, \
        g_yz, g_zz


# Function for converting contravariant vector CKS components to SKS
def cks_to_sks_vec_con(ax, ay, az, a, x, y, z):
    a2 = a ** 2
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    rr2 = x2 + y2 + z2
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = np.sqrt(r2)
    dr_dx = r * x / (2.0 * r2 - rr2 + a2)
    dr_dy = r * y / (2.0 * r2 - rr2 + a2)
    dr_dz = r * z * (1.0 + a2 / r2) / (2.0 * r2 - rr2 + a2)
    dth_dx = z / r * dr_dx / np.sqrt(r2 - z2)
    dth_dy = z / r * dr_dy / np.sqrt(r2 - z2)
    dth_dz = (z / r * dr_dz - 1.0) / np.sqrt(r2 - z2)
    dph_dx = -y / (x2 + y2) + a / (r2 + a2) * dr_dx
    dph_dy = x / (x2 + y2) + a / (r2 + a2) * dr_dy
    dph_dz = a / (r2 + a2) * dr_dz
    ar = dr_dx * ax + dr_dy * ay + dr_dz * az
    ath = dth_dx * ax + dth_dy * ay + dth_dz * az
    aph = dph_dx * ax + dph_dy * ay + dph_dz * az
    return ar, ath, aph


# Function for calculating normal-frame Lorentz factor
def normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
    uut = np.sqrt(1.0 + g_xx * uux ** 2 + 2.0 * g_xy * uux * uuy
                  + 2.0 * g_xz * uux * uuz + g_yy * uuy ** 2 + 2.0 * g_yz * uuy * uuz
                  + g_zz * uuz ** 2)
    return uut


# Function for transforming velocity from normal frame to coordinate frame
def norm_to_coord(uut, uux, uuy, uuz, alpha, betax, betay, betaz):
    ut = uut / alpha
    ux = uux - betax * ut
    uy = uuy - betay * ut
    uz = uuz - betaz * ut
    return ut, ux, uy, uz


# Function for transforming vector from contravariant to covariant components
def lower_vector(at, ax, ay, az,
                 g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
    a_t = g_tt * at + g_tx * ax + g_ty * ay + g_tz * az
    a_x = g_tx * at + g_xx * ax + g_xy * ay + g_xz * az
    a_y = g_ty * at + g_xy * ax + g_yy * ay + g_yz * az
    a_z = g_tz * at + g_xz * ax + g_yz * ay + g_zz * az
    return a_t, a_x, a_y, a_z


# Function for converting covariant covector CKS components to SKS
def cks_to_sks_vec_cov(a_x, a_y, a_z, a, x, y, z):
    a2 = a ** 2
    z2 = z ** 2
    rr2 = x ** 2 + y ** 2 + z2
    r2 = 0.5 * (rr2 - a2 + np.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = np.sqrt(r2)
    th = np.arccos(z / r)
    sth = np.sin(th)
    cth = np.cos(th)
    ph = np.arctan2(y, x) - np.arctan2(a, r)
    sph = np.sin(ph)
    cph = np.cos(ph)
    dx_dr = sth * cph
    dy_dr = sth * sph
    dz_dr = cth
    dx_dth = cth * (r * cph - a * sph)
    dy_dth = cth * (r * sph + a * cph)
    dz_dth = -r * sth
    dx_dph = sth * (-r * sph - a * cph)
    dy_dph = sth * (r * cph - a * sph)
    dz_dph = 0.0
    a_r = dx_dr * a_x + dy_dr * a_y + dz_dr * a_z
    a_th = dx_dth * a_x + dy_dth * a_y + dz_dth * a_z
    a_ph = dx_dph * a_x + dy_dph * a_y + dz_dph * a_z
    return a_r, a_th, a_ph


# Function for converting 3-magnetic field to 4-magnetic field
def three_field_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x, u_y, u_z):
    bt = u_x * bbx + u_y * bby + u_z * bbz
    bx = (bbx + bt * ux) / ut
    by = (bby + bt * uy) / ut
    bz = (bbz + bt * uz) / ut
    return bt, bx, by, bz


# Function for converting contravariant rank-2 tensor CKS components to SKS
def cks_to_sks_tens_con(axx, axy, axz, ayx, ayy, ayz, azx, azy, azz, a, x, y, z):
    axr, axth, axph = cks_to_sks_vec_con(axx, axy, axz, a, x, y, z)
    ayr, ayth, ayph = cks_to_sks_vec_con(ayx, ayy, ayz, a, x, y, z)
    azr, azth, azph = cks_to_sks_vec_con(azx, azy, azz, a, x, y, z)
    arr, athr, aphr = cks_to_sks_vec_con(axr, ayr, azr, a, x, y, z)
    arth, athth, aphth = cks_to_sks_vec_con(axth, ayth, azth, a, x, y, z)
    arph, athph, aphph = cks_to_sks_vec_con(axph, ayph, azph, a, x, y, z)
    return arr, arth, arph, athr, athth, athph, aphr, aphth, aphph


# Parse inputs and execute main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', help='name of input file, possibly including path')
    parser.add_argument('variable', help='name of variable to be plotted, any valid'
                                         'derived quantity prefaced by "derived:"')
    parser.add_argument('output_file', help='name of output to be (over)written; use '
                        '"show" to show interactive plot instead')
    parser.add_argument('-d', '--dimension', choices=('x', 'y', 'z', '1', '2', '3'),
                        help='dimension orthogonal to slice for 3D data')
    parser.add_argument('-l', '--location', type=float, default=0.0,
                        help='coordinate value along which slice is to be taken '
                             '(default: 0)')
    parser.add_argument('--r_max', type=float,
                        help='half-width of plot in both coordinates, centered at the '
                             'origin')
    parser.add_argument('--x1_min', type=float,
                        help='horizontal coordinate of left edge of plot')
    parser.add_argument('--x1_max', type=float,
                        help='horizontal coordinate of right edge of plot')
    parser.add_argument('--x2_min', type=float,
                        help='vertical coordinate of bottom edge of plot')
    parser.add_argument('--x2_max', type=float,
                        help='vertical coordinate of top edge of plot')
    parser.add_argument('--bh_zoom_rmax', type=float,
                        help='half-width of BH-centered zoom inset')
    parser.add_argument('--bh_center_rmax', type=float,
                        help='half-width of main plot bounds centered on live BH metadata')
    parser.add_argument('--center_zoom_rmax', type=float,
                        help='half-width of origin-centered zoom inset')
    parser.add_argument('--orthogonal_triptych', action='store_true',
                        help='plot xoy and yoz slices together in one figure')
    parser.add_argument('--ortho_x_min', type=float,
                        help='left bound for orthogonal triptych x coordinates')
    parser.add_argument('--ortho_x_max', type=float,
                        help='right bound for orthogonal triptych x coordinates')
    parser.add_argument('--ortho_y_min', type=float,
                        help='lower bound for orthogonal triptych y coordinates')
    parser.add_argument('--ortho_y_max', type=float,
                        help='upper bound for orthogonal triptych y coordinates')
    parser.add_argument('--ortho_z_min', type=float,
                        help='lower bound for orthogonal triptych z coordinates')
    parser.add_argument('--ortho_z_max', type=float,
                        help='upper bound for orthogonal triptych z coordinates')
    parser.add_argument('--ortho_yoz_y_min', type=float,
                        help='left bound for orthogonal triptych yoz-panel y coordinates')
    parser.add_argument('--ortho_yoz_y_max', type=float,
                        help='right bound for orthogonal triptych yoz-panel y coordinates')
    parser.add_argument('--ortho_yoz_z_min', type=float,
                        help='lower bound for orthogonal triptych yoz-panel z coordinates')
    parser.add_argument('--ortho_yoz_z_max', type=float,
                        help='upper bound for orthogonal triptych yoz-panel z coordinates')
    parser.add_argument('-c', '--cmap', help='name of Matplotlib colormap to use')
    parser.add_argument('-n', '--norm', help='name of Matplotlib norm to use')
    parser.add_argument('--vmin', type=optional_float, help='colormap minimum')
    parser.add_argument('--vmax', type=optional_float, help='colormap maximum')
    parser.add_argument('--linthresh', type=optional_float,
                        help='linear threshold for symlog normalization')
    parser.add_argument('--left_linthresh', type=optional_float,
                        help='linear threshold for derived:div_v_panels symlog')
    parser.add_argument('--left_vmax', type=optional_float,
                        help='absolute colorbar maximum for derived:div_v_panels')
    parser.add_argument('--right_linthresh', type=optional_float,
                        help=argparse.SUPPRESS)
    parser.add_argument('--right_vmax', type=optional_float,
                        help=argparse.SUPPRESS)
    parser.add_argument('--grid', action='store_true',
                        help='flag indicating domain decomposition should be overlaid')
    parser.add_argument('--grid_color', default='gray',
                        help='color string for grid overlay')
    parser.add_argument('--grid_alpha', type=float, default=0.5,
                        help='opacity of grid overlay')
    parser.add_argument('--streamlines', action='store_true',
                        help='flag indicating in-plane velocity streamlines should be '
                             'overlaid on slice plots')
    parser.add_argument('--stream_color', default='k',
                        help='color string for streamline overlay')
    parser.add_argument('--stream_alpha', type=float, default=1.0,
                        help='opacity of streamline overlay')
    parser.add_argument('--stream_density', type=float, default=3.375,
                        help='relative density of streamline seeds')
    parser.add_argument('--stream_arrowsize', type=float, default=0.35,
                        help='size of streamline arrows')
    parser.add_argument('--stream_resolution', type=int, default=128,
                        help='maximum resolution of the AMR-resampled streamline field')
    parser.add_argument('--bound_unbound_contour', action='store_true',
                        help='flag indicating the E=0 contour separating bound '
                             'and unbound gas should be overlaid')
    parser.add_argument('--bound_unbound_color', default='cyan',
                        help='color string for bound/unbound contour')
    parser.add_argument('--bound_unbound_alpha', type=float, default=0.9,
                        help='opacity of bound/unbound contour')
    parser.add_argument('--bound_unbound_linewidth', type=float, default=1.0,
                        help='line width of bound/unbound contour')
    parser.add_argument('--bh_mask', dest='bh_mask', action='store_true',
                        help='enable BH excision-mask overlay from dump metadata')
    parser.add_argument('--no_bh_mask', dest='bh_mask', action='store_false',
                        help='disable BH excision-mask overlay from dump metadata')
    parser.set_defaults(bh_mask=False)
    parser.add_argument('--bh_mask_color', default='k',
                        help='face color for BH excision mask')
    parser.add_argument('--bh_mask_edgecolor', default='none',
                        help='edge color for BH excision mask')
    parser.add_argument('--bh_mask_alpha', type=float, default=1.0,
                        help='opacity of BH excision mask')
    parser.add_argument('--bh_mask_linewidth', type=float, default=0.0,
                        help='line width for BH excision mask edge')
    parser.add_argument('--bh_mask_min_pixels', type=float, default=5.0,
                        help='minimum displayed radius of BH mask in pixels '
                             '(0 uses exact physical radius only)')
    parser.add_argument('--bh_marker', action='store_true',
                        help='also draw a fixed-size marker at the live BH position')
    parser.add_argument('--bh_marker_color', default='k',
                        help='color for optional live BH position marker')
    parser.add_argument('--bh_marker_size', type=float, default=18.0,
                        help='size of optional live BH position marker')
    parser.add_argument('--horizon', action='store_true',
                        help='flag indicating black hole event horizon should be marked')
    parser.add_argument('--horizon_color', default='k',
                        help='color string for event horizon marker')
    parser.add_argument('--horizon_mask', action='store_true',
                        help='flag indicating black hole event horizon should be masked')
    parser.add_argument('--horizon_mask_color', default='k',
                        help='color string for event horizon mask')
    parser.add_argument('--ergosphere', action='store_true',
                        help='flag indicating black hole ergosphere should be marked')
    parser.add_argument('--ergosphere_color', default='gray',
                        help='color string for ergosphere marker')
    parser.add_argument('--notex', action='store_true',
                        help='flag indicating Latex integration is not to be used')
    parser.add_argument('--dpi', type=float, default=300,
                        help='resolution of output figure (default: 300)')
    args = parser.parse_args()
    main(**vars(args))
