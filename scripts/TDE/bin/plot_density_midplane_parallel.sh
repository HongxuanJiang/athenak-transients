#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plot_density_midplane_parallel.sh [jobs] [output_dir]
#
# Defaults:
#   jobs       = number of available CPU cores
#   output_dir = dens_midplane_slices
#
# This script only plots the midplane density slice as a single panel with optional zoom
# insets, code-unit color limits, and a secondary cgs colorbar scale.

default_jobs() {
  nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
}

JOBS="${1:-$(default_jobs)}"
OUT_DIR="${2:-dens_midplane_slices}"

PLOT_SCRIPT="${PLOT_SCRIPT:-./plot_slice.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DENS_VMIN="${DENS_VMIN:-${VMIN:-1.e-10}}"   # code units
DENS_VMAX="${DENS_VMAX:-${VMAX:-1.e-0}}"    # code units
DENS_CMAP="${DENS_CMAP:-Spectral_r}"
PLOT_X1_MIN="${PLOT_X1_MIN:--4}"
PLOT_X1_MAX="${PLOT_X1_MAX:-4}"
PLOT_X2_MIN="${PLOT_X2_MIN:--4}"
PLOT_X2_MAX="${PLOT_X2_MAX:-4}"
# BH_ZOOM_RMAX="${BH_ZOOM_RMAX:-15}"
# CENTER_ZOOM_RMAX="${CENTER_ZOOM_RMAX:-1}"
# DENS_MID_BH_ZOOM_RMAX="${DENS_MID_BH_ZOOM_RMAX:-${BH_ZOOM_RMAX}}"
# DENS_MID_CENTER_ZOOM_RMAX="${DENS_MID_CENTER_ZOOM_RMAX:-}"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "Error: plot script not found: $PLOT_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

plot_if_needed() {
  local bin_file="$1"
  local out_file="$2"
  shift 2

  if [[ -f "$out_file" ]]; then
    echo "skip: $out_file"
    return 0
  fi

  echo "plot: $bin_file -> $out_file"
  "$PYTHON_BIN" "$PLOT_SCRIPT" "$bin_file" dens "$out_file" "$@"
}

plot_one() {
  local bin_file="$1"
  local base_name stem out_file
  local -a args

  base_name="$(basename "$bin_file")"
  stem="${base_name%.bin}"
  out_file="${OUT_DIR}/${stem}.dens_midplane.png"

  args=(
    -c "$DENS_CMAP" -n log --vmin "$DENS_VMIN" --vmax "$DENS_VMAX"
    --x1_min "$PLOT_X1_MIN" --x1_max "$PLOT_X1_MAX"
    --x2_min "$PLOT_X2_MIN" --x2_max "$PLOT_X2_MAX"
    --notex
  )
  if [[ -n "$DENS_MID_BH_ZOOM_RMAX" ]]; then
    args+=(--bh_zoom_rmax "$DENS_MID_BH_ZOOM_RMAX")
  fi
  if [[ -n "$DENS_MID_CENTER_ZOOM_RMAX" ]]; then
    args+=(--center_zoom_rmax "$DENS_MID_CENTER_ZOOM_RMAX")
  fi

  plot_if_needed "$bin_file" "$out_file" "${args[@]}"
}

export OUT_DIR PLOT_SCRIPT PYTHON_BIN DENS_VMIN DENS_VMAX DENS_CMAP
export PLOT_X1_MIN PLOT_X1_MAX PLOT_X2_MIN PLOT_X2_MAX
export DENS_MID_BH_ZOOM_RMAX DENS_MID_CENTER_ZOOM_RMAX
export -f plot_if_needed plot_one

mapfile -d '' BIN_FILES < <(find . -maxdepth 1 -type f -name '*.bin' -print0 | sort -z)

if (( ${#BIN_FILES[@]} == 0 )); then
  echo "No .bin files found in $(pwd)"
  exit 0
fi

printf '%s\0' "${BIN_FILES[@]}" | xargs -0 -n1 -P "$JOBS" bash -c 'plot_one "$1"' _

echo "Done. Images are in: $OUT_DIR"
