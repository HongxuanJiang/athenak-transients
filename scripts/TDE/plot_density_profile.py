#!/usr/bin/env python3

"""
Plot solid-angle-averaged density profiles from AthenaK .bin snapshots.

The script automatically finds snapshots in the current directory, picks the
first and last files, and plots radial density profiles for both:
  - x-axis: radius (linear)
  - y-axis: density (log scale)

Usage:
  python3 plot_density_profile.py
"""

import glob
import os
import re
import struct

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def read_header_metadata(file_obj):
    """Read Athena binary header and input metadata."""
    line = file_obj.readline().decode("ascii")
    if line != "Athena binary output version=1.1\n":
        raise RuntimeError("Unrecognized data file format.")

    # Preheader lines
    next(file_obj)  # size of preheader
    time_line = file_obj.readline().decode("ascii").strip()
    next(file_obj)  # cycle

    line = file_obj.readline().decode("ascii")
    if not line.startswith("  size of location="):
        raise RuntimeError("Could not read location size.")
    location_size = int(line[19:])

    line = file_obj.readline().decode("ascii")
    if not line.startswith("  size of variable="):
        raise RuntimeError("Could not read variable size.")
    variable_size = int(line[19:])

    next(file_obj)  # blank/comment line before variables

    line = file_obj.readline().decode("ascii")
    if not line.startswith("  variables:"):
        raise RuntimeError("Could not read variable names.")
    variable_names = line[12:].split()

    line = file_obj.readline().decode("ascii")
    if not line.startswith("  header offset="):
        raise RuntimeError("Could not read header offset.")
    header_offset = int(line[16:])

    if location_size not in (4, 8):
        raise RuntimeError("Only 4- and 8-byte location types are supported.")
    if variable_size not in (4, 8):
        raise RuntimeError("Only 4- and 8-byte variable types are supported.")

    start_of_data = file_obj.tell() + header_offset
    input_data = {}
    section_name = None
    while file_obj.tell() < start_of_data:
        line = file_obj.readline().decode("ascii")
        if not line or line.startswith("#"):
            continue
        if line.startswith("<") and line.rstrip().endswith(">"):
            section_name = line[1:-2]
            input_data[section_name] = {}
            continue
        if section_name is None:
            continue
        key, val = line.split("=", 1)
        input_data[section_name][key.strip()] = val.split("#", 1)[0].strip()

    time_value = None
    if time_line.startswith("time="):
        # Defensive parse in case spacing differs from expected format.
        time_value = float(time_line.split("=", 1)[1].strip())
    elif time_line.startswith("time"):
        time_value = float(time_line.split("=", 1)[1].strip())
    elif "time=" in time_line:
        time_value = float(time_line.split("time=", 1)[1].strip())
    else:
        try:
            time_value = float(time_line.split("=", 1)[1].strip())
        except (IndexError, ValueError):
            time_value = None

    return {
        "location_size": location_size,
        "variable_size": variable_size,
        "location_format": "f" if location_size == 4 else "d",
        "variable_format": "f" if variable_size == 4 else "d",
        "variable_names": variable_names,
        "num_variables": len(variable_names),
        "input_data": input_data,
        "time": time_value,
    }


def default_rmax_from_mesh(input_data):
    """Compute a radius upper bound from mesh extents."""
    mesh = input_data.get("mesh", {})
    required = ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max")
    if not all(k in mesh for k in required):
        return None
    xmax = max(abs(float(mesh["x1min"])), abs(float(mesh["x1max"])))
    ymax = max(abs(float(mesh["x2min"])), abs(float(mesh["x2max"])))
    zmax = max(abs(float(mesh["x3min"])), abs(float(mesh["x3max"])))
    return np.sqrt(xmax * xmax + ymax * ymax + zmax * zmax)


def infer_num_bins(input_data, r_max):
    """Infer linear radial bin count from mesh resolution."""
    mesh = input_data.get("mesh", {})
    required = ("nx1", "nx2", "nx3", "x1min", "x1max", "x2min", "x2max", "x3min", "x3max")
    if not all(k in mesh for k in required):
        return 256

    nx1 = int(mesh["nx1"])
    nx2 = int(mesh["nx2"])
    nx3 = int(mesh["nx3"])
    if nx1 <= 0 or nx2 <= 0 or nx3 <= 0:
        return 256

    dx = (float(mesh["x1max"]) - float(mesh["x1min"])) / nx1
    dy = (float(mesh["x2max"]) - float(mesh["x2min"])) / nx2
    dz = (float(mesh["x3max"]) - float(mesh["x3min"])) / nx3
    dr = min(abs(dx), abs(dy), abs(dz))
    if dr <= 0.0:
        return 256

    bins = int(np.ceil(r_max / dr))
    return max(64, bins)


def snapshot_index(filename):
    """Extract trailing numeric snapshot index from ...<digits>.bin."""
    match = re.search(r"(\d+)\.bin$", os.path.basename(filename))
    return int(match.group(1)) if match else None


def sorted_snapshot_files():
    """Auto-detect and sort snapshot files in current directory."""
    candidate_patterns = ("TDEExternal.hydro_w.*.bin", "*.bin")
    files = []
    for pattern in candidate_patterns:
        found = glob.glob(pattern)
        if len(found) >= 2:
            files = found
            break
    if len(files) < 2:
        raise RuntimeError("Need at least 2 .bin snapshot files in current directory.")

    if all(snapshot_index(f) is not None for f in files):
        files = sorted(files, key=lambda f: snapshot_index(f))
    else:
        files = sorted(files)
    return files


def infer_output_name(first_file, last_file):
    """Infer output figure name from snapshot basename prefix."""
    b1 = os.path.basename(first_file)
    b2 = os.path.basename(last_file)
    m1 = re.match(r"^(.*?)(\d+)\.bin$", b1)
    m2 = re.match(r"^(.*?)(\d+)\.bin$", b2)
    if m1 is not None and m2 is not None and m1.group(1) == m2.group(1):
        prefix = m1.group(1).rstrip(".")
        if prefix:
            return prefix + ".density_profile_first_last.png"
    return "density_profile_first_last.png"


def compute_density_profile(filename, bin_edges):
    """Compute shell-averaged density profile from one AthenaK .bin file."""
    rho_sum = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    vol_sum = np.zeros(len(bin_edges) - 1, dtype=np.float64)

    with open(filename, "rb") as file_obj:
        file_obj.seek(0, 2)
        file_size = file_obj.tell()
        file_obj.seek(0, 0)

        meta = read_header_metadata(file_obj)
        if "dens" not in meta["variable_names"]:
            raise RuntimeError(f'"dens" not found in {filename}')
        dens_index = meta["variable_names"].index("dens")

        # Number of ghost cells is stored in the embedded input file metadata.
        try:
            num_ghost = int(meta["input_data"]["mesh"]["nghost"])
        except KeyError as exc:
            raise RuntimeError("Unable to find mesh/nghost in metadata.") from exc

        location_format = meta["location_format"]
        variable_format = meta["variable_format"]
        variable_size = meta["variable_size"]
        num_variables = meta["num_variables"]

        while file_obj.tell() < file_size:
            block_indices = np.array(struct.unpack("@6i", file_obj.read(24))) - num_ghost
            block_nx = block_indices[1] - block_indices[0] + 1
            block_ny = block_indices[3] - block_indices[2] + 1
            block_nz = block_indices[5] - block_indices[4] + 1

            # Skip block logical indices (i, j, k, level).
            file_obj.seek(16, 1)

            block_lims = struct.unpack(
                "=6" + location_format, file_obj.read(6 * meta["location_size"])
            )

            xf, dx = np.linspace(block_lims[0], block_lims[1], block_nx + 1, retstep=True)
            yf, dy = np.linspace(block_lims[2], block_lims[3], block_ny + 1, retstep=True)
            zf, dz = np.linspace(block_lims[4], block_lims[5], block_nz + 1, retstep=True)

            x = 0.5 * (xf[:-1] + xf[1:])
            y = 0.5 * (yf[:-1] + yf[1:])
            z = 0.5 * (zf[:-1] + zf[1:])

            cells_per_block = block_nx * block_ny * block_nz
            variable_data_size = cells_per_block * variable_size
            block_cell_format = "=" + str(cells_per_block) + variable_format

            cell_data_start = file_obj.tell()
            file_obj.seek(cell_data_start + dens_index * variable_data_size, 0)
            rho = np.array(
                struct.unpack(block_cell_format, file_obj.read(variable_data_size))
            ).reshape(block_nz, block_ny, block_nx)

            # Move to the next block: skip all variables for this block.
            file_obj.seek(cell_data_start + num_variables * variable_data_size, 0)

            r = np.sqrt(
                x[None, None, :] * x[None, None, :]
                + y[None, :, None] * y[None, :, None]
                + z[:, None, None] * z[:, None, None]
            )

            r_flat = r.ravel()
            rho_flat = rho.ravel()
            cell_volume = float(dx * dy * dz)

            rho_sum += np.histogram(r_flat, bins=bin_edges, weights=rho_flat * cell_volume)[0]
            vol_sum += np.histogram(
                r_flat, bins=bin_edges, weights=np.full(r_flat.shape, cell_volume)
            )[0]

    profile = np.full_like(rho_sum, np.nan, dtype=np.float64)
    valid = vol_sum > 0.0
    profile[valid] = rho_sum[valid] / vol_sum[valid]
    return profile, meta["time"]


def main():
    files = sorted_snapshot_files()
    first_file = files[0]
    last_file = files[-1]

    with open(first_file, "rb") as first_obj:
        first_meta = read_header_metadata(first_obj)
    r_max = default_rmax_from_mesh(first_meta["input_data"])
    if r_max is None:
        raise RuntimeError("Unable to infer radial range from mesh metadata.")
    if r_max <= 0.0:
        raise RuntimeError("Inferred radial range is not positive.")

    bins = infer_num_bins(first_meta["input_data"], r_max)
    bin_edges = np.linspace(0.0, r_max, bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    profile_first, time_first = compute_density_profile(first_file, bin_edges)
    profile_last, time_last = compute_density_profile(last_file, bin_edges)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    mask_first = np.isfinite(profile_first) & (profile_first > 0.0)
    mask_last = np.isfinite(profile_last) & (profile_last > 0.0)
    if not np.any(mask_first):
        raise RuntimeError(f"No positive density values found in {first_file}.")
    if not np.any(mask_last):
        raise RuntimeError(f"No positive density values found in {last_file}.")

    first_label = os.path.basename(first_file)
    last_label = os.path.basename(last_file)
    if time_first is not None:
        first_label += f" (t={time_first:g})"
    if time_last is not None:
        last_label += f" (t={time_last:g})"

    ax.plot(r_centers[mask_first], profile_first[mask_first], lw=2.0, label=first_label)
    ax.plot(r_centers[mask_last], profile_last[mask_last], lw=2.0, label=last_label)

    ax.set_xlabel("Radius")
    ax.set_ylabel("Solid-angle-averaged density")
    ax.set_xlim(0.0, r_max)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_file = infer_output_name(first_file, last_file)
    fig.savefig(output_file, dpi=200)

    print(f"First snapshot: {first_file}")
    print(f"Last snapshot:  {last_file}")
    print(f"Radial bins:    {bins}")
    print(f"r_max:          {r_max:g}")
    print(f"Saved figure:   {output_file}")


if __name__ == "__main__":
    main()
