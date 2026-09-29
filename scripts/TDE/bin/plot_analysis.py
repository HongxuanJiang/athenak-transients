#!/usr/bin/env python3
"""Make paper-style figures from compact TDE analysis .dat outputs.

Usage:
  python plot_analysis.py
  python plot_analysis.py /path/to/analysis_dat
  python plot_analysis.py /path/to/analysis_dat /path/to/figure_dir
"""

from __future__ import annotations

import math
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


MSUN_G = 1.98847e33
DAY_S = 86400.0
SNAPSHOT_ANALYSIS_RE = re.compile(r"\.analysis\.dat$")
COLOR = {
    "domain": "#0072B2",
    "bound": "#009E73",
    "unbound": "#D55E00",
    "sink": "#CC79A7",
    "outer": "#E69F00",
    "total": "#000000",
    "control": "#56B4E9",
}
SNAPSHOT_STYLES = [
    {"color": "#0072B2", "ls": "-", "lw": 1.6},
    {"color": "#D55E00", "ls": "--", "lw": 1.6},
    {"color": "#009E73", "ls": "-.", "lw": 1.6},
    {"color": "#CC79A7", "ls": ":", "lw": 1.8},
    {"color": "#E69F00", "ls": (0, (5, 1, 1, 1)), "lw": 1.6},
    {"color": "#56B4E9", "ls": (0, (3, 1, 1, 1, 1, 1)), "lw": 1.6},
]
BUDGET_STYLES = {
    "in_domain_latest": {"color": COLOR["domain"], "ls": "-", "lw": 1.5},
    "sink_cumulative": {"color": COLOR["sink"], "ls": "--", "lw": 1.5},
    "outer_cumulative": {"color": COLOR["outer"], "ls": ":", "lw": 1.8},
    "approximate_total_latest": {"color": COLOR["total"], "ls": "-.", "lw": 1.9},
}
QUANTITY_LABELS = {
    "energy": r"$\epsilon_{\rm orb}$ [erg g$^{-1}$]",
    "j": r"$|j|$ [cm$^2$ s$^{-1}$]",
    "Be": r"$Be$ [erg g$^{-1}$]",
    "e": r"$e_{\rm est}$",
}
CHANNEL_LABELS = {
    "in_domain_latest": "In-domain latest",
    "sink_cumulative": "Sink cumulative",
    "outer_cumulative": "Outer cumulative",
    "approximate_total_latest": "Approx. total latest",
    "in_domain": "In-domain",
    "sink_interval": "Sink interval",
    "outer_interval": "Outer interval",
}
ENERGY_TAIL_FIT_MAX = -5.0e16
ENERGY_TAIL_FIT_MIN_POINTS = 4


def set_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "font.size": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.5,
    })


def parse_columns_header(path: Path) -> List[str]:
    with path.open("r", encoding="ascii") as fp:
        for line in fp:
            if line.startswith("# columns ="):
                return line.split("=", 1)[1].strip().split()
    raise RuntimeError(f"Could not find columns header in {path}.")


def read_numeric_table(path: Path) -> Dict[str, np.ndarray]:
    columns = parse_columns_header(path)
    data = np.genfromtxt(path, comments="#", ndmin=2)
    if data.size == 0:
        return {name: np.zeros(0, dtype=np.float64) for name in columns}
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(columns):
        raise RuntimeError(f"Column mismatch in {path}: expected {len(columns)}, got {data.shape[1]}.")
    return {name: np.array(data[:, i], dtype=np.float64, copy=True) for i, name in enumerate(columns)}


def parse_combined_histogram_file(path: Path) -> Tuple[Dict[str, str], Dict[str, Tuple[float, str]], Dict[Tuple[str, str], np.ndarray]]:
    metadata: Dict[str, str] = {}
    summary: Dict[str, Tuple[float, str]] = {}
    hist_lists: Dict[Tuple[str, str], List[List[float]]] = {}
    with path.open("r", encoding="ascii") as fp:
        for raw in fp:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                body = line[1:].strip()
                if "=" in body:
                    key, value = body.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue
            parts = line.split()
            if parts[0] == "SUMMARY":
                name = parts[1]
                value = float(parts[2]) if parts[2].lower() != "nan" else math.nan
                unit = parts[3] if len(parts) >= 4 else ""
                summary[name] = (value, unit)
                continue
            if parts[0] == "HIST":
                channel = parts[1]
                quantity = parts[2]
                values = [float(x) for x in parts[3:]]
                hist_lists.setdefault((channel, quantity), []).append(values)
                continue
    histograms = {
        key: np.array(rows, dtype=np.float64) if rows else np.zeros((0, 6), dtype=np.float64)
        for key, rows in hist_lists.items()
    }
    return metadata, summary, histograms


def collect_snapshot_analysis_files(analysis_dir: Path) -> List[Path]:
    files = [p for p in analysis_dir.glob("*.analysis.dat") if SNAPSHOT_ANALYSIS_RE.search(p.name)]
    files.sort()
    return files


def choose_time_unit(times_s: np.ndarray) -> Tuple[float, str]:
    max_time = float(np.nanmax(times_s)) if times_s.size else 0.0
    if max_time >= 2.0 * DAY_S:
        return DAY_S, "days"
    if max_time >= 2.0 * 3600.0:
        return 3600.0, "hours"
    return 1.0, "s"


def build_edges_from_centers(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.array([0.0, 1.0], dtype=np.float64)
    if values.size == 1:
        width = max(abs(values[0]) * 0.05, 1.0)
        return np.array([values[0] - width, values[0] + width], dtype=np.float64)
    mids = 0.5 * (values[:-1] + values[1:])
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = mids
    edges[0] = values[0] - (mids[0] - values[0])
    edges[-1] = values[-1] + (values[-1] - mids[-1])
    return edges


def pick_snapshot_indices(n: int, max_count: int = 4) -> List[int]:
    if n <= 0:
        return []
    raw = np.linspace(0, n - 1, num=min(max_count, n))
    indices = []
    for value in raw:
        idx = int(round(float(value)))
        if idx not in indices:
            indices.append(idx)
    return indices


def choose_selected_snapshot_indices(snapshots: Sequence[Dict[str, object]], max_count: int = 4) -> List[int]:
    if not snapshots:
        return []
    if len(snapshots) <= max_count:
        return list(range(len(snapshots)))

    anchor_pos = None
    for pos, snap in enumerate(snapshots):
        if int(snap.get("index", -1)) == 10:
            anchor_pos = pos
            break

    if anchor_pos is None:
        return pick_snapshot_indices(len(snapshots), max_count=max_count)

    raw = np.linspace(anchor_pos, len(snapshots) - 1, num=max_count)
    chosen: List[int] = []
    for value in raw:
        idx = int(round(float(value)))
        if idx not in chosen:
            chosen.append(idx)
    return chosen


def positive_floor(values: np.ndarray) -> float:
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return 1.0
    return float(np.nanmax([np.nanmin(positive) * 0.5, 1.0e-300]))


def fit_negative_energy_tail_powerlaw(
    hist: np.ndarray,
    energy_max: float = ENERGY_TAIL_FIT_MAX,
    min_points: int = ENERGY_TAIL_FIT_MIN_POINTS,
) -> Dict[str, np.ndarray | float] | None:
    if hist.size == 0:
        return None
    centers = hist[:, 2]
    density = hist[:, 4]
    mask = (
        np.isfinite(centers)
        & np.isfinite(density)
        & (density > 0.0)
        & (centers < energy_max)
    )
    if np.count_nonzero(mask) < min_points:
        return None
    x_mag = -centers[mask]
    y_val = density[mask]
    valid = np.isfinite(x_mag) & np.isfinite(y_val) & (x_mag > 0.0) & (y_val > 0.0)
    if np.count_nonzero(valid) < min_points:
        return None
    x_mag = x_mag[valid]
    y_val = y_val[valid]
    logx = np.log10(x_mag)
    logy = np.log10(y_val)
    slope, intercept = np.polyfit(logx, logy, 1)
    x_fit_mag = np.geomspace(np.min(x_mag), np.max(x_mag), 256)
    y_fit = 10.0 ** (intercept + slope * np.log10(x_fit_mag))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "x_fit": -x_fit_mag[::-1],
        "y_fit": y_fit[::-1],
    }


def iter_positive_histogram_segments(hist: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    if hist.size == 0:
        return []
    left = hist[:, 0]
    right = hist[:, 1]
    density = hist[:, 4]
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(density) & (density > 0.0)
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    start: int | None = None
    for idx, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = idx
        elif not is_valid and start is not None:
            edges = np.concatenate([left[start:idx], right[idx - 1:idx]])
            segments.append((edges, density[start:idx].copy()))
            start = None
    if start is not None:
        edges = np.concatenate([left[start:], right[-1:]])
        segments.append((edges, density[start:].copy()))
    return segments


def plot_histogram_steps(
    ax: plt.Axes,
    hist: np.ndarray,
    *,
    quantity: str,
    color: str | None = None,
    lw: float = 1.4,
    ls: object = "-",
    label: str | None = None,
    alpha: float = 1.0,
    zorder: float | None = None,
) -> bool:
    segments = iter_positive_histogram_segments(hist)
    if not segments:
        return False
    handle = None
    for seg_idx, (edges, values) in enumerate(segments):
        kwargs = {
            "where": "post",
            "lw": lw,
            "ls": ls,
            "alpha": alpha,
        }
        if color is not None:
            kwargs["color"] = color
        if zorder is not None:
            kwargs["zorder"] = zorder
        if seg_idx == 0 and label is not None:
            kwargs["label"] = label
        y = np.concatenate([values, values[-1:]])
        handle = ax.step(edges, y, **kwargs)
    configure_quantity_axis(ax, quantity, hist[:, 2])
    ax.set_yscale("log")
    return handle is not None


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def add_axis_legend(ax: plt.Axes, **kwargs: object) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    unique_handles = []
    unique_labels = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or label == "_nolegend_" or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, **kwargs)


def add_figure_legend(fig: plt.Figure, axes: Sequence[plt.Axes], **kwargs: object) -> None:
    unique_handles = []
    unique_labels = []
    seen: set[str] = set()
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if not label or label == "_nolegend_" or label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        fig.legend(unique_handles, unique_labels, **kwargs)


def prepare_heatmap_row(quantity: str, hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if hist.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if quantity == "j":
        left = hist[:, 0]
        right = hist[:, 1]
        dm = hist[:, 3]
        valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(dm) & (left > 0.0) & (right > left)
        if not np.any(valid):
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        left = left[valid]
        right = right[valid]
        dm = dm[valid]
        log_left = np.log10(left)
        log_right = np.log10(right)
        log_centers = 0.5 * (log_left + log_right)
        dlog = log_right - log_left
        values = np.zeros_like(dm)
        good = dlog > 0.0
        values[good] = dm[good] / dlog[good]
        return log_centers, values
    centers = hist[:, 2]
    values = hist[:, 4]
    return np.array(centers, dtype=np.float64, copy=True), np.array(values, dtype=np.float64, copy=True)


def build_heatmap_display_grid(quantity: str, hist: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float | None, str]:
    if hist.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), None, r"$\log_{10}(dM/dX)$"
    if quantity == "j":
        positive = hist[(hist[:, 0] > 0.0) & (hist[:, 1] > hist[:, 0])]
        if positive.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), None, r"$\log_{10}(dM/d\log_{10}|j|)$"
        log_edges = np.log10(np.concatenate([positive[:, 0], positive[-1:, 1]]))
        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        display_centers = np.linspace(log_centers[0], log_centers[-1], max(log_centers.size * 4, 800), dtype=np.float64)
        display_edges = 10.0 ** build_edges_from_centers(display_centers)
        return display_edges, display_centers, None, r"$\log_{10}(dM/d\log_{10}|j|)$"
    if quantity in {"energy", "Be"}:
        edges = np.concatenate([hist[:, 0], hist[-1:, 1]])
        nonzero = np.abs(edges[np.isfinite(edges) & (edges != 0.0)])
        scale = float(np.nanmin(nonzero)) if nonzero.size else 1.0
        scale = max(scale, 1.0e-30)
        u_edges = np.arcsinh(edges / scale)
        u_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
        display_centers = np.linspace(u_centers[0], u_centers[-1], max(u_centers.size * 4, 800), dtype=np.float64)
        display_edges = scale * np.sinh(build_edges_from_centers(display_centers))
        return display_edges, display_centers, scale, r"$\log_{10}(dM/dX)$"
    centers = np.array(hist[:, 2], dtype=np.float64, copy=True)
    display_centers = np.linspace(centers[0], centers[-1], max(centers.size * 4, 600), dtype=np.float64)
    display_edges = build_edges_from_centers(display_centers)
    return display_edges, display_centers, None, r"$\log_{10}(dM/dX)$"


def resample_heatmap_row(quantity: str, hist: np.ndarray, display_centers: np.ndarray, aux_scale: float | None) -> np.ndarray:
    if hist.size == 0 or display_centers.size == 0:
        return np.zeros(display_centers.size, dtype=np.float64)
    if quantity == "j":
        base_centers, base_values = prepare_heatmap_row(quantity, hist)
        if base_centers.size == 0:
            return np.zeros(display_centers.size, dtype=np.float64)
        return np.interp(display_centers, base_centers, base_values, left=0.0, right=0.0)
    if quantity in {"energy", "Be"}:
        left = hist[:, 0]
        right = hist[:, 1]
        values = hist[:, 4]
        scale = 1.0 if aux_scale is None else aux_scale
        u_left = np.arcsinh(left / scale)
        u_right = np.arcsinh(right / scale)
        base_centers = 0.5 * (u_left + u_right)
        return np.interp(display_centers, base_centers, values, left=0.0, right=0.0)
    base_centers = hist[:, 2]
    base_values = hist[:, 4]
    return np.interp(display_centers, base_centers, base_values, left=0.0, right=0.0)


def configure_quantity_axis(ax: plt.Axes, quantity: str, centers: np.ndarray) -> None:
    if quantity in {"energy", "Be"}:
        nonzero = np.abs(centers[np.isfinite(centers) & (centers != 0.0)])
        linthresh = float(np.nanmin(nonzero)) if nonzero.size else 1.0
        ax.set_xscale("symlog", linthresh=max(linthresh, 1.0e-30))
    elif quantity == "j":
        positive = centers[np.isfinite(centers) & (centers > 0.0)]
        if positive.size:
            ax.set_xscale("log")


def plot_mass_budget(global_data: Dict[str, np.ndarray], outdir: Path, time_scale: float, time_label: str) -> None:
    t = global_data["snapshot_time"] / time_scale
    initial_mass = float(global_data["M_total_accounted"][0]) if t.size else 1.0
    mass_scale = initial_mass if initial_mass > 0.0 else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.9), sharex=True, constrained_layout=True)

    axes[0, 0].plot(t, global_data["M_domain"] / mass_scale, lw=1.7, ls="-", color=COLOR["domain"], label=r"$M_{\rm domain}$")
    axes[0, 0].plot(t, global_data["M_bound_domain"] / mass_scale, lw=1.4, ls="--", color=COLOR["bound"], label=r"$M_{\rm bound}$")
    axes[0, 0].plot(t, global_data["M_unbound_domain"] / mass_scale, lw=1.4, ls="-.", color=COLOR["unbound"], label=r"$M_{\rm unbound}$")
    axes[0, 0].set_ylabel(r"Mass / $M_{\rm initial}$")
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_title("In-Domain Mass")
    add_axis_legend(axes[0, 0], loc="best")

    axes[0, 1].plot(t, global_data["M_sink_cumulative"] / mass_scale, lw=1.6, ls="--", color=COLOR["sink"], label="Sink cumulative")
    axes[0, 1].plot(t, global_data["M_outer_cumulative"] / mass_scale, lw=1.8, ls=":", color=COLOR["outer"], label="Outer cumulative")
    axes[0, 1].set_ylabel(r"Mass / $M_{\rm initial}$")
    axes[0, 1].set_title("Cumulative Loss Channels")
    add_axis_legend(axes[0, 1], loc="best")

    axes[1, 0].plot(t, global_data["M_total_accounted"] / mass_scale, lw=1.8, ls="-", color=COLOR["total"], label="Accounted total")
    axes[1, 0].axhline(1.0, color="0.35", lw=0.9, ls="--", label="Initial mass")
    axes[1, 0].set_xlabel(f"Time [{time_label}]")
    axes[1, 0].set_ylabel(r"Mass / $M_{\rm initial}$")
    axes[1, 0].set_ylim(0.97, 1.03)
    axes[1, 0].set_title("Accounted Total")
    add_axis_legend(axes[1, 0], loc="best")

    axes[1, 1].plot(t, global_data["mass_conservation_error"] / mass_scale, lw=1.6, ls="-", color="#7F3C8D", label="Residual")
    axes[1, 1].axhline(0.0, color="0.35", lw=0.9, ls="--", label="Zero residual")
    axes[1, 1].set_xlabel(f"Time [{time_label}]")
    axes[1, 1].set_ylabel(r"Residual / $M_{\rm initial}$")
    axes[1, 1].set_title("Mass Accounting Residual")
    add_axis_legend(axes[1, 1], loc="best")

    save_figure(fig, outdir, "figure_01_mass_budget")


def plot_rates(global_data: Dict[str, np.ndarray], outdir: Path, time_scale: float, time_label: str) -> None:
    mask = np.isfinite(global_data["interval_mid_time"])
    if not np.any(mask):
        return
    t = global_data["interval_mid_time"][mask] / time_scale
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True, constrained_layout=True)

    msun_per_day = DAY_S / MSUN_G
    for key, label, color, ls in [
        ("Mdot_sink", "Sink", COLOR["sink"], "-"),
        ("Mdot_outer", "Outer", COLOR["outer"], "--"),
        ("Mdot_control", r"Inward flux at $r_{\rm ctrl}$", COLOR["control"], "-."),
    ]:
        y = np.abs(global_data[key][mask]) * msun_per_day
        positive = y > 0.0
        if np.any(positive):
            axes[0].plot(t[positive], y[positive], lw=1.6, ls=ls, color=color, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Rate [$M_\odot$ day$^{-1}$]")
    axes[0].set_title("Flux Rates")
    add_axis_legend(axes[0], loc="best")

    axes[1].plot(t, global_data["dM_bound_domain_dt"][mask] * msun_per_day, lw=1.6, ls="-", color=COLOR["bound"], label=r"$dM_{\rm bound}/dt$")
    axes[1].axhline(0.0, color="0.35", lw=0.9, ls="--", label="Zero line")
    axes[1].set_xlabel(f"Time [{time_label}]")
    axes[1].set_ylabel(r"$dM_{\rm bound}/dt$ [$M_\odot$ day$^{-1}$]")
    axes[1].set_title("Bound-Mass Evolution")
    add_axis_legend(axes[1], loc="best")

    save_figure(fig, outdir, "figure_02_rates")


def plot_circularization(global_data: Dict[str, np.ndarray], outdir: Path, time_scale: float, time_label: str) -> None:
    t = global_data["snapshot_time"] / time_scale
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.0), sharex=True, constrained_layout=True)

    axes[0, 0].plot(t, global_data["mass_fraction_e_lt_ecrit"], lw=1.6, ls="-", color="#4C78A8", label=r"$e_{\rm est}<e_{\rm crit}$")
    axes[0, 0].plot(t, global_data["mass_fraction_Rcirc_lt_Rcrit"], lw=1.6, ls="--", color="#F58518", label=r"$R_{\rm circ}<R_{\rm crit}$")
    axes[0, 0].set_ylabel("Bound-Mass Fraction")
    axes[0, 0].set_ylim(bottom=0.0)
    axes[0, 0].set_title("Circularization Fractions")
    add_axis_legend(axes[0, 0], loc="best")

    axes[0, 1].plot(t, global_data["mean_j_bound"], lw=1.6, ls="-", color=COLOR["domain"], label="Mean")
    axes[0, 1].plot(t, global_data["median_j_bound"], lw=1.6, ls="--", color=COLOR["sink"], label="Median")
    axes[0, 1].set_ylabel(r"$|j|$ [cm$^2$ s$^{-1}$]")
    axes[0, 1].set_title("Bound Angular Momentum")
    add_axis_legend(axes[0, 1], loc="best")

    axes[1, 0].plot(t, global_data["mean_eps_bound"], lw=1.6, ls="-", color=COLOR["domain"], label="Mean")
    axes[1, 0].plot(t, global_data["median_eps_bound"], lw=1.6, ls="--", color=COLOR["sink"], label="Median")
    axes[1, 0].axhline(0.0, color="0.35", lw=0.9, ls=":", label="Zero energy")
    axes[1, 0].set_xlabel(f"Time [{time_label}]")
    axes[1, 0].set_ylabel(r"$\epsilon_{\rm orb}$ [erg g$^{-1}$]")
    axes[1, 0].set_title("Bound Orbital Energy")
    add_axis_legend(axes[1, 0], loc="best")

    axes[1, 1].plot(t, global_data["M_bound_domain"] / MSUN_G, lw=1.7, ls="-", color=COLOR["bound"], label=r"$M_{\rm bound}$")
    axes[1, 1].set_xlabel(f"Time [{time_label}]")
    axes[1, 1].set_ylabel(r"$M_{\rm bound}$ [$M_\odot$]")
    axes[1, 1].set_title("Bound Mass")
    add_axis_legend(axes[1, 1], loc="best")

    save_figure(fig, outdir, "figure_03_circularization")


def plot_distribution_budget(cumulative_hists: Dict[Tuple[str, str], np.ndarray], outdir: Path) -> None:
    quantities = [q for q in ("energy", "j", "Be") if ("approximate_total_latest", q) in cumulative_hists]
    if not quantities:
        return
    fig, axes = plt.subplots(1, len(quantities), figsize=(5.1 * len(quantities), 4.0), constrained_layout=True)
    if len(quantities) == 1:
        axes = [axes]
    for ax, quantity in zip(axes, quantities):
        for channel in [
            "in_domain_latest",
            "sink_cumulative",
            "outer_cumulative",
            "approximate_total_latest",
        ]:
            data = cumulative_hists.get((channel, quantity))
            if data is None or data.size == 0:
                continue
            style = BUDGET_STYLES[channel]
            plot_histogram_steps(
                ax,
                data,
                quantity=quantity,
                lw=float(style["lw"]),
                ls=style["ls"],
                color=style["color"],
                label=CHANNEL_LABELS[channel],
            )
        ax.set_xlabel(QUANTITY_LABELS[quantity])
        ax.set_ylabel(r"$dM/dX$")
        ax.set_title(f"{quantity} Budget")
    add_figure_legend(
        fig,
        axes,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.4,
        handlelength=2.8,
    )
    save_figure(fig, outdir, "figure_04_distribution_budget")


def plot_selected_snapshots(snapshots: Sequence[Dict[str, object]], outdir: Path, time_scale: float, time_label: str) -> None:
    if not snapshots:
        return
    chosen = choose_selected_snapshot_indices(snapshots, max_count=4)
    quantities = [q for q in ("energy", "j", "Be", "e") if ("in_domain", q) in snapshots[0]["histograms"]]
    if not quantities:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), constrained_layout=True)
    axes_list = axes.ravel()
    legend_axes: List[plt.Axes] = []
    energy_tail_fit_lines: List[Tuple[str, str]] = []
    for ax, quantity in zip(axes_list, quantities):
        legend_axes.append(ax)
        if quantity == "energy":
            ax.plot([], [], color="0.25", lw=1.2, ls=(0, (3, 2)),
                    label=r"Tail fit ($E<-5\times10^{16}$)")
        for style_rank, idx in enumerate(chosen):
            snap = snapshots[idx]
            data = snap["histograms"].get(("in_domain", quantity))
            if data is None or data.size == 0:
                continue
            label = f"{snap['index']:05d} ({snap['time_s'] / time_scale:.2f} {time_label})"
            style = SNAPSHOT_STYLES[style_rank % len(SNAPSHOT_STYLES)]
            plot_histogram_steps(
                ax,
                data,
                quantity=quantity,
                lw=float(style["lw"]),
                ls=style["ls"],
                color=style["color"],
                label=label,
            )
            if quantity == "energy":
                fit = fit_negative_energy_tail_powerlaw(data)
                if fit is not None:
                    ax.plot(
                        fit["x_fit"],
                        fit["y_fit"],
                        color=style["color"],
                        lw=max(1.2, float(style["lw"]) - 0.1),
                        ls=(0, (3, 2)),
                        alpha=0.95,
                    )
                    energy_tail_fit_lines.append(
                        (style["color"], f"{snap['index']:05d}: $\\alpha={fit['slope']:.2f}$")
                    )
        ax.set_xlabel(QUANTITY_LABELS[quantity])
        ax.set_ylabel(r"$dM/dX$")
        ax.set_title(f"In-Domain {quantity}")
        if quantity == "energy" and energy_tail_fit_lines:
            ax.text(
                0.03,
                0.97,
                "Tail power-law fit\n" + r"$dM/dE \propto (-E)^\alpha$" + "\n"
                + r"$E < -5\times10^{16}\ {\rm erg\ g^{-1}}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.3,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "0.75", "alpha": 0.82},
            )
            y_anchor = 0.72
            for line_idx, (color, text) in enumerate(energy_tail_fit_lines):
                ax.text(
                    0.05,
                    y_anchor - 0.055 * line_idx,
                    text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=7.1,
                    color=color,
                )
    for ax in axes_list[len(quantities):]:
        ax.set_visible(False)
    add_figure_legend(
        fig,
        legend_axes,
        loc="upper center",
        ncol=min(4, len(chosen)),
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.4,
        handlelength=2.8,
    )
    save_figure(fig, outdir, "figure_05_selected_snapshots")


def plot_histogram_evolution(
    snapshots: Sequence[Dict[str, object]],
    channels: Sequence[str],
    outdir: Path,
    stem: str,
    title_prefix: str,
    time_scale: float,
    time_label: str,
    global_data: Dict[str, np.ndarray] | None = None,
) -> None:
    if not snapshots:
        return
    quantities = []
    for quantity in ("energy", "j", "Be", "e"):
        if any((channel, quantity) in snap["histograms"] for snap in snapshots for channel in channels):
            quantities.append(quantity)
    if not quantities:
        return

    ncols = 2
    nrows = int(math.ceil(len(quantities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 3.2 * nrows), constrained_layout=True)
    axes_list = np.atleast_1d(axes).ravel()

    times = np.array([snap["time_s"] for snap in snapshots], dtype=np.float64) / time_scale
    time_edges = build_edges_from_centers(times)

    for ax, quantity in zip(axes_list, quantities):
        y_edges = None
        display_centers = None
        display_label = QUANTITY_LABELS[quantity]
        colorbar_label = r"$\log_{10}(dM/dX)$"
        aux_scale = None
        for snap in snapshots:
            data = None
            for channel in channels:
                candidate = snap["histograms"].get((channel, quantity))
                if candidate is not None:
                    data = candidate
                    break
            if y_edges is None:
                if data is None or data.size == 0:
                    continue
                y_edges, display_centers, aux_scale, colorbar_label = build_heatmap_display_grid(quantity, data)
                if y_edges.size == 0 or display_centers.size == 0:
                    continue
                break
        if y_edges is None:
            ax.set_visible(False)
            continue
        matrix = []
        for snap in snapshots:
            data = None
            for channel in channels:
                candidate = snap["histograms"].get((channel, quantity))
                if candidate is not None:
                    data = candidate
                    break
            if data is None or data.size == 0:
                matrix.append(np.zeros(y_edges.size - 1, dtype=np.float64))
            else:
                matrix.append(resample_heatmap_row(quantity, data, display_centers, aux_scale))
        matrix_arr = np.array(matrix, dtype=np.float64)
        floor = positive_floor(matrix_arr)
        image = np.log10(np.maximum(matrix_arr, floor))
        pcm = ax.pcolormesh(time_edges, y_edges, image.T, cmap="magma", shading="auto")
        fig.colorbar(pcm, ax=ax, pad=0.01, label=colorbar_label)
        if quantity in {"energy", "Be"}:
            linthresh = 1.0 if aux_scale is None else aux_scale
            ax.set_yscale("symlog", linthresh=max(linthresh, 1.0e-30))
        elif quantity == "j":
            positive = y_edges[y_edges > 0.0]
            if positive.size:
                ax.set_yscale("log")
            if global_data is not None and channels == ["in_domain"]:
                t_overlay = global_data["snapshot_time"] / time_scale
                for key, label, color, ls in [
                    ("mean_j_bound", "Mean bound $j$", "white", "--"),
                    ("median_j_bound", "Median bound $j$", "#80DEEA", "-"),
                ]:
                    y_overlay = global_data[key]
                    mask = np.isfinite(t_overlay) & np.isfinite(y_overlay) & (y_overlay > 0.0)
                    if np.any(mask):
                        ax.plot(
                            t_overlay[mask],
                            y_overlay[mask],
                            color=color,
                            ls=ls,
                            lw=1.1,
                            alpha=0.95,
                            label=label,
                        )
                add_axis_legend(ax, loc="lower right")
        ax.set_xlabel(f"Time [{time_label}]")
        ax.set_ylabel(display_label)
        ax.set_title(f"{title_prefix} {quantity}")
    for ax in axes_list[len(quantities):]:
        ax.set_visible(False)
    save_figure(fig, outdir, stem)


def write_figure_manifest(outdir: Path) -> None:
    lines = [
        "# Paper Figures",
        "",
        "- `figure_01_mass_budget`: in-domain masses, cumulative sink/outer losses, accounted total, mass residual.",
        "- `figure_02_rates`: sink/outer/control rates and bound-mass time derivative.",
        "- `figure_03_circularization`: circularization fractions plus bound-gas mean/median `j` and `eps` evolution.",
        "- `figure_04_distribution_budget`: latest in-domain, cumulative sink, cumulative outer, and approximate latest total distributions.",
        "- `figure_05_selected_snapshots`: selected in-domain histogram overlays for representative snapshots.",
        "- `figure_06_in_domain_evolution`: time-evolution heatmaps of in-domain distributions.",
        "- `figure_07_flux_evolution`: time-evolution heatmaps of sink/outer interval distributions.",
    ]
    (outdir / "README_FIGURES.md").write_text("\n".join(lines) + "\n", encoding="ascii")


def main(argv: Sequence[str]) -> int:
    if len(argv) > 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    script_dir = Path(__file__).resolve().parent
    analysis_dir = Path(argv[1]).expanduser().resolve() if len(argv) >= 2 else (Path.cwd() / "analysis_dat")
    figure_dir = Path(argv[2]).expanduser().resolve() if len(argv) >= 3 else (analysis_dir / "paper_figures")

    global_path = analysis_dir / "analysis_global_timeseries.dat"
    cumulative_path = analysis_dir / "analysis_cumulative_distributions.dat"
    if not global_path.exists():
        cached = sorted(analysis_dir.glob("*.analysis.dat"))
        if cached:
            print(
                "error: missing "
                f"{global_path}\n"
                "found per-snapshot *.analysis.dat files but no global time-series file.\n"
                "This usually means the analysis run is still in progress or was produced by an older "
                "script version that only wrote global outputs at the end.\n"
                "Rerun analyze_tde.py on the same output directory after the analysis process stops; "
                "the current version will reuse cached per-snapshot files and rebuild the global outputs.",
                file=sys.stderr,
            )
            return 1
        print(f"error: missing {global_path}", file=sys.stderr)
        return 1
    if not cumulative_path.exists():
        print(f"error: missing {cumulative_path}", file=sys.stderr)
        return 1

    set_paper_style()
    if figure_dir.exists():
        shutil.rmtree(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    global_data = read_numeric_table(global_path)
    _, _, cumulative_hists = parse_combined_histogram_file(cumulative_path)
    snapshot_files = collect_snapshot_analysis_files(analysis_dir)
    snapshots = []
    for path in snapshot_files:
        metadata, _, histograms = parse_combined_histogram_file(path)
        snapshots.append({
            "path": path,
            "index": int(metadata.get("snapshot_index", "-1")),
            "time_s": float(metadata.get("snapshot_time_s", "nan")),
            "histograms": histograms,
        })
    snapshots.sort(key=lambda item: item["index"])

    time_scale, time_label = choose_time_unit(global_data["snapshot_time"])
    plot_mass_budget(global_data, figure_dir, time_scale, time_label)
    plot_rates(global_data, figure_dir, time_scale, time_label)
    plot_circularization(global_data, figure_dir, time_scale, time_label)
    plot_distribution_budget(cumulative_hists, figure_dir)
    plot_selected_snapshots(snapshots, figure_dir, time_scale, time_label)
    plot_histogram_evolution(
        snapshots,
        ["in_domain"],
        figure_dir,
        "figure_06_in_domain_evolution",
        "In-Domain",
        time_scale,
        time_label,
        global_data=global_data,
    )
    plot_histogram_evolution(snapshots, ["sink_interval", "outer_interval"], figure_dir, "figure_07_flux_evolution", "Flux", time_scale, time_label)
    write_figure_manifest(figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
