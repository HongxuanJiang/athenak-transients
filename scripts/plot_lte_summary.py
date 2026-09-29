#!/usr/bin/env python3

"""Plot a Zach-style summary panel from an AthenaK native LTE table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def read_header_block(fp, name: str) -> list[str]:
    begin = f"<{name}begin>\n".encode("ascii")
    end = f"<{name}end>\n".encode("ascii")
    line = fp.readline()
    if line != begin:
        raise RuntimeError(f"Expected {begin!r}, got {line!r}")
    out: list[str] = []
    while True:
        line = fp.readline()
        if not line:
            raise RuntimeError(f"Unterminated {name} header block")
        if line == end:
            return out
        out.append(line.decode("ascii").strip())


def parse_key_values(lines: list[str], cast=str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in lines:
        if "=" not in line:
            raise RuntimeError(f"Invalid header line: {line}")
        key, value = line.split("=", 1)
        parsed[key.strip()] = cast(value.strip())
    return parsed


def read_native_table(path: Path) -> tuple[dict[str, str], dict[str, float], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with path.open("rb") as fp:
        metadata = parse_key_values(read_header_block(fp, "metadata"))
        scalars = parse_key_values(read_header_block(fp, "scalars"), float)
        point_counts = parse_key_values(read_header_block(fp, "points"), int)
        field_names = [line.strip() for line in read_header_block(fp, "fields") if line.strip()]
        dtype = np.dtype("<f8")
        raw = np.frombuffer(fp.read(), dtype=dtype)

    if metadata.get("log_axis_base", "e") != "e":
        raise RuntimeError(f'Expected native AthenaK table with log_axis_base = e, got "{metadata.get("log_axis_base")}".')

    offset = 0
    logrho = np.array(raw[offset:offset + point_counts["logrho"]], copy=True)
    offset += point_counts["logrho"]
    logtemp = np.array(raw[offset:offset + point_counts["logtemp"]], copy=True)
    offset += point_counts["logtemp"]

    shape = (point_counts["logrho"], point_counts["logtemp"])
    npoints = shape[0] * shape[1]
    fields: dict[str, np.ndarray] = {}
    for name in field_names:
        arr = raw[offset:offset + npoints]
        if arr.size != npoints:
            raise RuntimeError(f"Truncated field {name} in {path}")
        fields[name] = np.array(arr, copy=True).reshape(shape)
        offset += npoints

    if offset != raw.size:
        raise RuntimeError(f"Trailing binary data in {path}")
    return metadata, scalars, logrho, logtemp, fields


def build_gamma_cmap() -> LinearSegmentedColormap:
    points = (np.array([1.0, 1.2, 4.0 / 3.0, 7.0 / 5.0, 1.6, 5.0 / 3.0]) - 1.0) / (5.0 / 3.0 - 1.0)
    colors = ["white", "thistle", "powderblue", "limegreen", "gold", "tomato"]
    return LinearSegmentedColormap.from_list("gam", list(zip(points, colors)), N=1024)


def build_mu_cmap(y_he: float) -> LinearSegmentedColormap:
    mmw1 = 1.0 / ((1.0 - y_he) * 2.0 + y_he * 3.0 / 4.0)
    mmw2 = 1.0 / ((1.0 - y_he) * 2.0 + y_he * 2.0 / 4.0)
    mmw3 = 1.0 / ((1.0 - y_he) * 2.0 + y_he * 1.0 / 4.0)
    mmw4 = 1.0 / ((1.0 - y_he) + y_he * 1.0 / 4.0)
    mmw5 = 1.0 / ((1.0 - y_he) * 1.0 / 2.0 + y_he * 1.0 / 4.0)
    points = (np.array([0.6, mmw1, mmw2, mmw3, mmw4, mmw5, 2.4]) - 0.6) / (2.4 - 0.6)
    turbo = mpl.colormaps["turbo"]
    colors = turbo(np.concatenate(([0.0], np.linspace(0.05, 0.95, 5), [1.0])))
    return LinearSegmentedColormap.from_list("mmw", list(zip(points, colors)), N=1024)


def add_pair_contours(
    ax: plt.Axes,
    temp: np.ndarray,
    rho: np.ndarray,
    fields: dict[str, np.ndarray],
) -> None:
    if "pair_pressure_fraction" not in fields:
        return
    log_pair_fraction = np.log10(np.maximum(fields["pair_pressure_fraction"], 1.0e-300))
    levels = [-10.0, -6.0, -3.0]
    finite_max = float(np.nanmax(log_pair_fraction))
    if finite_max < levels[0]:
        return
    active_levels = [level for level in levels if level <= finite_max]
    contours = ax.contour(
        temp,
        rho,
        log_pair_fraction,
        levels=active_levels,
        colors=("white", "black", "black")[: len(active_levels)],
        linewidths=(1.0, 1.0, 1.4)[: len(active_levels)],
        linestyles=(":", "--", "-")[: len(active_levels)],
        alpha=0.9,
    )
    labels = {level: rf"$10^{{{int(level)}}}$" for level in active_levels}
    ax.clabel(contours, fmt=labels, inline=True, fontsize=7)


def annotate_pair_region(ax_gam: plt.Axes, fields: dict[str, np.ndarray]) -> None:
    if "pair_pressure_fraction" not in fields:
        return
    if float(np.nanmax(fields["pair_pressure_fraction"])) <= 1.0e-3:
        return
    ax_gam.annotate(
        r"$e^-+e^+\ {\rm pairs}$",
        (0.87, 0.78),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=11,
        rotation=38,
    )


def is_chabrier_summary_table(metadata: dict[str, str]) -> bool:
    table_type = str(metadata.get("table_type", ""))
    return table_type in (
        "lte_chabrier2021_t13_helm_union",
        "lte_chabrier2021_t13_helm_union_prad",
    )


def is_saha_hydrogen_table(metadata: dict[str, str]) -> bool:
    return str(metadata.get("table_type", "")) == "saha_hydrogen_lte"


def build_source_state(metadata: dict[str, str], fields: dict[str, np.ndarray]) -> tuple[np.ndarray, list[int], list[str], ListedColormap]:
    if "source_t13" not in fields:
        raise RuntimeError("Union table is missing the required source_t13 mask")

    state = np.zeros_like(fields["source_t13"], dtype=np.int32)
    next_state = 1
    ticks = [0]
    labels = ["invalid"]
    colors = ["#ffffff"]

    def add(mask_name: str, label: str, color: str) -> None:
        nonlocal next_state
        if mask_name not in fields:
            return
        mask = fields[mask_name] > 0.5
        if not np.any(mask):
            return
        state[mask] = next_state
        ticks.append(next_state)
        labels.append(label)
        colors.append(color)
        next_state += 1

    add("source_t13", "T13", "#0b84a5")
    if "source_blend_t13_scvh" in fields:
        add("source_blend_t13_scvh", "T13/SCvH blend", "#4daf4a")
    else:
        add("source_blend_chabrier", "T13/Chabrier", "#4daf4a")

    if "source_scvh" in fields:
        add("source_scvh", "SCvH", "#f6c85f")
    else:
        add("source_chabrier", "Chabrier", "#f6c85f")

    if "source_corner_cp" in fields:
        add("source_corner_cp", "CP corner", "#6f4e7c")
    if "source_corner_helm" in fields:
        add("source_corner_helm", "HELM fallback", "#9dd9d2")
    elif "source_corner" in fields:
        corner_name = str(metadata.get("union_corner_backstop", "corner")).lower()
        if "helm" in corner_name:
            corner_label = (
                "HELM"
                if is_chabrier_summary_table(metadata)
                and "union_helm_preferred_log10_temp_min" in metadata
                else "HELM fallback"
            )
        elif "cp" in corner_name:
            corner_label = "CP corner"
        else:
            corner_label = "corner"
        add("source_corner", corner_label, "#9dd9d2")

    return state, ticks, labels, ListedColormap(colors[: len(labels)])


def smooth_source_state_for_display(
    state: np.ndarray,
    iterations: int = 2,
    majority_threshold: int = 6,
) -> np.ndarray:
    """Suppress 1-2 cell spikes in the diagnostic source map.

    This is display-only smoothing for the ownership panel. It leaves invalid
    cells untouched and only relabels a cell when a strong 3x3 neighborhood
    majority disagrees with the current state.
    """
    smoothed = np.array(state, copy=True)
    valid = smoothed > 0
    labels = np.unique(smoothed[valid])
    if labels.size == 0:
        return smoothed

    for _ in range(iterations):
        padded = np.pad(smoothed, 1, mode="edge")
        count_stack = []
        for label in labels:
            counts = np.zeros_like(smoothed, dtype=np.int32)
            for di in range(3):
                for dj in range(3):
                    counts += (padded[di: di + smoothed.shape[0], dj: dj + smoothed.shape[1]] == label)
            count_stack.append(counts)
        stack = np.stack(count_stack, axis=0)
        best_idx = np.argmax(stack, axis=0)
        best_count = np.max(stack, axis=0)
        best_state = labels[best_idx]
        replace = valid & (best_state != smoothed) & (best_count >= majority_threshold)
        if not np.any(replace):
            break
        smoothed[replace] = best_state[replace]
    return smoothed


def smooth_boundary_indices(indices: np.ndarray, window: int = 9) -> np.ndarray:
    out = np.array(indices, copy=True)
    if out.size == 0:
        return out
    half = window // 2
    for i in range(out.size):
        if not np.isfinite(out[i]):
            continue
        lo = max(0, i - half)
        hi = min(out.size, i + half + 1)
        neighborhood = out[lo:hi]
        neighborhood = neighborhood[np.isfinite(neighborhood)]
        if neighborhood.size > 0:
            out[i] = np.nanmedian(neighborhood)
    return out


def regularize_chabrier_source_state_for_display(
    state: np.ndarray,
    labels: list[str],
) -> np.ndarray:
    """Enforce a smooth row-wise T13->blend->Chabrier->HELM display order."""
    if not {"T13", "T13/Chabrier", "Chabrier", "HELM fallback"}.issubset(set(labels)):
        return smooth_source_state_for_display(state)

    t_state = labels.index("T13")
    b_state = labels.index("T13/Chabrier")
    c_state = labels.index("Chabrier")
    h_state = labels.index("HELM fallback")

    valid = state > 0
    nr, nt = state.shape
    b0 = np.full(nr, np.nan, dtype=np.float64)
    c0 = np.full(nr, np.nan, dtype=np.float64)
    h0 = np.full(nr, np.nan, dtype=np.float64)
    for i in range(nr):
        for st, arr in ((b_state, b0), (c_state, c0), (h_state, h0)):
            idx = np.where(state[i] == st)[0]
            if idx.size > 0:
                arr[i] = float(idx[0])

    b0 = smooth_boundary_indices(b0)
    c0 = smooth_boundary_indices(c0)
    h0 = smooth_boundary_indices(h0)

    display = np.zeros_like(state)
    for i in range(nr):
        row_valid = valid[i]
        if not np.any(row_valid):
            continue
        display[i, row_valid] = t_state
        end = nt
        if np.isfinite(h0[i]):
            hh = int(np.clip(round(h0[i]), 0, nt))
            if hh < end:
                mask = np.zeros(nt, dtype=bool)
                mask[hh:end] = True
                mask &= row_valid
                display[i, mask] = h_state
                end = hh
        if np.isfinite(c0[i]):
            cc = int(np.clip(round(c0[i]), 0, end))
            if cc < end:
                mask = np.zeros(nt, dtype=bool)
                mask[cc:end] = True
                mask &= row_valid
                display[i, mask] = c_state
                end = cc
        if np.isfinite(b0[i]):
            bb = int(np.clip(round(b0[i]), 0, end))
            if bb < end:
                mask = np.zeros(nt, dtype=bool)
                mask[bb:end] = True
                mask &= row_valid
                display[i, mask] = b_state
    return smooth_source_state_for_display(display, iterations=1, majority_threshold=7)


def annotate_mu_axis(ax_mu) -> None:
    ax_mu.annotate(r"${\rm H_2}, {\rm He}$", (0.20, 0.10), xycoords="axes fraction", ha="center", va="bottom", fontsize=12)
    ax_mu.annotate(r"${\rm H}, {\rm He}$", (0.48, 0.10), xycoords="axes fraction", ha="center", va="bottom", fontsize=10, rotation=80)
    ax_mu.annotate(r"${\rm H^+}, {\rm He}$", (0.59, 0.10), xycoords="axes fraction", ha="center", va="bottom", fontsize=10, rotation=80)
    ax_mu.annotate(r"${\rm H^+}, {\rm He^+}$", (0.67, 0.10), xycoords="axes fraction", ha="center", va="bottom", fontsize=10, rotation=80)
    ax_mu.annotate(r"${\rm H^+}, {\rm He^{2+}}$", (0.85, 0.10), xycoords="axes fraction", ha="center", va="bottom", fontsize=12)


def annotate_gamma_axis(ax_gam) -> None:
    ax_gam.annotate(r"${\rm H} + {\rm H} \leftrightarrow {\rm H_2}$", (0.47, 0.35), xycoords="axes fraction", ha="center", va="bottom", fontsize=12, rotation=75)
    ax_gam.annotate(r"${\rm H} \leftrightarrow {\rm H^+}$", (0.62, 0.42), xycoords="axes fraction", ha="center", va="bottom", fontsize=12, rotation=75)
    ax_gam.annotate(r"${\rm He} \leftrightarrow {\rm He^+}$", (0.65, 0.25), xycoords="axes fraction", ha="center", va="bottom", fontsize=10, rotation=75)
    ax_gam.annotate(r"${\rm He^+} \leftrightarrow {\rm He^{2+}}$", (0.75, 0.25), xycoords="axes fraction", ha="center", va="bottom", fontsize=10, rotation=75)
    ax_gam.annotate("radiation dominated", (0.65, 0.10), xycoords="axes fraction", ha="center", va="bottom", fontsize=12)
    ax_gam.annotate("diatomic\nideal gas", (0.30, 0.65), xycoords="axes fraction", ha="center", va="bottom", fontsize=12)
    ax_gam.annotate("rotational degrees not excited", (0.05, 0.28), xycoords="axes fraction", ha="center", va="bottom", fontsize=10, rotation=90)


def annotate_axes(ax_mu, ax_gam) -> None:
    annotate_mu_axis(ax_mu)
    annotate_gamma_axis(ax_gam)


def write_saha_hydrogen_summary(
    output: Path,
    logrho: np.ndarray,
    logtemp: np.ndarray,
    fields: dict[str, np.ndarray],
    dpi: int,
    temp_min: float | None = None,
    temp_max: float | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
    annotate: bool = True,
) -> None:
    rho = np.exp(logrho)
    temp = np.exp(logtemp)
    temp_min = float(temp[0]) if temp_min is None else temp_min
    temp_max = float(temp[-1]) if temp_max is None else temp_max
    rho_min = float(rho[0]) if rho_min is None else rho_min
    rho_max = float(rho[-1]) if rho_max is None else rho_max

    fig, axs = plt.subplots(figsize=(8.8, 4.0), ncols=2, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=1.0e-3, wspace=0.1)

    im1 = axs[0].pcolormesh(
        temp,
        rho,
        fields["xion"],
        shading="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=mpl.colormaps["viridis"],
    )
    im2 = axs[1].pcolormesh(
        temp,
        rho,
        fields["gamma1"],
        shading="auto",
        vmin=1.0,
        vmax=5.0 / 3.0,
        cmap=build_gamma_cmap(),
    )

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(temp_min, temp_max)
        ax.set_ylim(rho_min, rho_max)
        ax.set_xlabel(r"$T$ [${\rm K}$]")

    axs[0].set_ylabel(r"$\rho$ [${\rm g/cm^3}$]")
    axs[0].set_title("H ionization")
    axs[1].set_title(r"$\Gamma_1$")

    cax1 = axs[0].inset_axes([0.0, 1.0, 1.0, 0.1])
    cbar1 = plt.colorbar(im1, cax=cax1, orientation="horizontal")
    cbar1.set_label(r"$x_{\rm ion}$")
    cbar1.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cax1.xaxis.set_label_position("top")
    cax1.xaxis.set_ticks_position("top")

    cax2 = axs[1].inset_axes([0.0, 1.0, 1.0, 0.1])
    cbar2 = plt.colorbar(im2, cax=cax2, orientation="horizontal")
    cbar2.set_label(r"$\Gamma_1$")
    cbar2.set_ticks([1.0, 1.2, 4.0 / 3.0, 1.4, 1.5, 5.0 / 3.0])
    cbar2.set_ticklabels(["1", "1.2", "4/3", "7/5", "1.5", "5/3"])
    cax2.xaxis.set_label_position("top")
    cax2.xaxis.set_ticks_position("top")

    if annotate:
        axs[0].annotate(r"${\rm H}\leftrightarrow{\rm H^+}$", (0.55, 0.38), xycoords="axes fraction", ha="center", va="bottom", fontsize=12, rotation=75)
        axs[1].annotate(r"${\rm H}\leftrightarrow{\rm H^+}$", (0.55, 0.38), xycoords="axes fraction", ha="center", va="bottom", fontsize=12, rotation=75)
        axs[1].annotate("monatomic\nideal gas", (0.35, 0.74), xycoords="axes fraction", ha="center", va="bottom", fontsize=12)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def write_chabrier_combined_figure(
    output: Path,
    metadata: dict[str, str],
    logrho: np.ndarray,
    logtemp: np.ndarray,
    fields: dict[str, np.ndarray],
    dpi: int,
    temp_min: float | None = None,
    temp_max: float | None = None,
    rho_min: float | None = None,
    rho_max: float | None = None,
    annotate: bool = True,
) -> None:
    rho = np.exp(logrho)
    temp = np.exp(logtemp)
    gamma1 = fields["gamma1"]
    state, ticks, labels, cmap = build_source_state(metadata, fields)
    state = regularize_chabrier_source_state_for_display(state, labels)
    temp_min = float(temp[0]) if temp_min is None else temp_min
    temp_max = float(temp[-1]) if temp_max is None else temp_max
    rho_min = float(rho[0]) if rho_min is None else rho_min
    rho_max = float(rho[-1]) if rho_max is None else rho_max

    fig, axs = plt.subplots(figsize=(10.6, 4.2), ncols=2, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=1.0e-3, wspace=0.30)

    mesh = axs[0].pcolormesh(
        temp, rho, state, shading="auto", cmap=cmap, vmin=-0.5, vmax=len(labels) - 0.5
    )
    add_pair_contours(axs[0], temp, rho, fields)
    gamma_mesh = axs[1].pcolormesh(
        temp, rho, gamma1, shading="auto", vmin=1.0, vmax=5.0 / 3.0, cmap=build_gamma_cmap()
    )

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(temp_min, temp_max)
        ax.set_ylim(rho_min, rho_max)
        ax.set_xlabel(r"$T$ [${\rm K}$]")
    axs[0].set_ylabel(r"$\rho$ [${\rm g/cm^3}$]")
    axs[0].set_title("Source Map")
    axs[1].set_title(r"$\Gamma_1$")

    cbar1 = plt.colorbar(mesh, ax=axs[0], ticks=ticks, fraction=0.06, pad=0.02)
    cbar1.ax.set_yticklabels(labels)
    cbar1.ax.tick_params(labelsize=10)

    cax2 = axs[1].inset_axes([0.0, 1.0, 1.0, 0.1])
    cbar2 = plt.colorbar(gamma_mesh, cax=cax2, orientation="horizontal")
    cbar2.set_label(r"$\Gamma_1$")
    cbar2.set_ticks([1.0, 1.2, 4.0 / 3.0, 1.4, 1.5, 5.0 / 3.0])
    cbar2.set_ticklabels(["1", "1.2", "4/3", "7/5", "1.5", "5/3"])
    cax2.xaxis.set_label_position("top")
    cax2.xaxis.set_ticks_position("top")

    if annotate:
        annotate_gamma_axis(axs[1])
        annotate_pair_region(axs[1], fields)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path, help="AthenaK native LTE table")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output figure path; default is <table-stem>_summary.png next to the table",
    )
    parser.add_argument("--temp-min", type=float, default=None, help="Minimum plotted temperature [K]; default uses the full table")
    parser.add_argument("--temp-max", type=float, default=None, help="Maximum plotted temperature [K]; default uses the full table")
    parser.add_argument("--rho-min", type=float, default=None, help="Minimum plotted density [g cm^-3]; default uses the full table")
    parser.add_argument("--rho-max", type=float, default=None, help="Maximum plotted density [g cm^-3]; default uses the full table")
    parser.add_argument("--dpi", type=int, default=256, help="PNG/PDF dpi")
    parser.add_argument("--no-annotations", action="store_true", help="Disable the Zach-style text annotations")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is None:
        args.output = args.table.with_name(f"{args.table.stem}_summary.png")
    metadata, scalars, logrho, logtemp, fields = read_native_table(args.table)

    if is_saha_hydrogen_table(metadata):
        for name in ("xion", "gamma1"):
            if name not in fields:
                raise RuntimeError(f"{args.table} does not contain the required {name} field")
        write_saha_hydrogen_summary(
            args.output,
            logrho,
            logtemp,
            fields,
            dpi=args.dpi,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            annotate=not args.no_annotations,
        )
        print(f"Wrote H-only Saha summary plot: {args.output}")
        return 0

    if is_chabrier_summary_table(metadata):
        if "gamma1" not in fields:
            raise RuntimeError(f"{args.table} does not contain the required gamma1 field")
        write_chabrier_combined_figure(
            args.output,
            metadata,
            logrho,
            logtemp,
            fields,
            dpi=args.dpi,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
            rho_min=args.rho_min,
            rho_max=args.rho_max,
            annotate=not args.no_annotations,
        )
        print(f"Wrote combined Chabrier summary plot: {args.output}")
        if float(logrho[-1]) / np.log(10.0) > -1.0:
            print(
                "WARNING: plotted table extends above log10(rho_cgs) = -1; "
                "the current T13 generator treats that as outside the ideal-equilibrium "
                "accuracy regime even though it can still generate the table."
            )
        return 0

    if "mu" not in fields or "gamma1" not in fields:
        raise RuntimeError(f"{args.table} does not contain the required mu/gamma1 fields")

    rho = np.exp(logrho)
    temp = np.exp(logtemp)
    mu = fields["mu"]
    gamma1 = fields["gamma1"]
    temp_min = float(temp[0]) if args.temp_min is None else args.temp_min
    temp_max = float(temp[-1]) if args.temp_max is None else args.temp_max
    rho_min = float(rho[0]) if args.rho_min is None else args.rho_min
    rho_max = float(rho[-1]) if args.rho_max is None else args.rho_max

    cmap_mu = build_mu_cmap(float(scalars["y_he"]))
    cmap_gam = build_gamma_cmap()

    fig, axs = plt.subplots(figsize=(8.8, 4.0), ncols=2, sharex=True, sharey=True)
    plt.subplots_adjust(hspace=1.0e-3, wspace=0.1)

    im1 = axs[0].pcolormesh(temp, rho, mu, shading="auto", vmin=0.6, vmax=2.4, cmap=cmap_mu)
    add_pair_contours(axs[0], temp, rho, fields)
    im2 = axs[1].pcolormesh(temp, rho, gamma1, shading="auto", vmin=1.0, vmax=5.0 / 3.0, cmap=cmap_gam)

    for ax in axs:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(temp_min, temp_max)
        ax.set_ylim(rho_min, rho_max)

    axs[0].set_ylabel(r"$\rho$ [${\rm g/cm^3}$]")
    axs[0].set_xlabel(r"$T$ [${\rm K}$]")
    axs[1].set_xlabel(r"$T$ [${\rm K}$]")

    cax1 = axs[0].inset_axes([0.0, 1.0, 1.0, 0.1])
    cbar1 = plt.colorbar(im1, cax=cax1, orientation="horizontal")
    cbar1.set_label(r"$\mu$ [${\rm mol/g}$]")
    cbar1.set_ticks([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    cax1.xaxis.set_label_position("top")
    cax1.xaxis.set_ticks_position("top")

    cax2 = axs[1].inset_axes([0.0, 1.0, 1.0, 0.1])
    cbar2 = plt.colorbar(im2, cax=cax2, orientation="horizontal")
    cbar2.set_label(r"$\Gamma_1$")
    cbar2.set_ticks([1.0, 1.2, 4.0 / 3.0, 1.4, 1.5, 5.0 / 3.0])
    cbar2.set_ticklabels(["1", "1.2", "4/3", "7/5", "1.5", "5/3"])
    cax2.xaxis.set_label_position("top")
    cax2.xaxis.set_ticks_position("top")

    if not args.no_annotations:
        annotate_axes(axs[0], axs[1])
        annotate_pair_region(axs[1], fields)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", dpi=args.dpi)
    print(f"Wrote summary plot: {args.output}")
    if float(logrho[-1]) / np.log(10.0) > -1.0:
        print(
            "WARNING: plotted table extends above log10(rho_cgs) = -1; "
            "the current T13 generator treats that as outside the ideal-equilibrium "
            "accuracy regime even though it can still generate the table."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
