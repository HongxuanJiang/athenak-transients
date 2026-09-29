#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plot_density_slices_parallel.sh [jobs] [output_dir]
#
# Defaults:
#   jobs       = number of available CPU cores
#   output_dir = density_slices
#
# Additional products:
#   dens_midplane_slices: log-scale midplane density plots with code-unit ranges and a
#     secondary cgs colorbar scale
#   vel_norm_slices : linear speed orthogonal-triptych plots
#   grav_phi_slices : log-scale |grav_phi| orthogonal-triptych plots
#   c_s_slices     : log-scale sound-speed orthogonal-triptych plots
#   mach_slices    : log-scale Mach-number orthogonal-triptych plots
#   div_v_slices   : stream-focused xoy/yoz plots of div(v) and div(v) r_t/c_s
#   pgas_slices    : log-scale gas-pressure orthogonal-triptych plots
#   temp_slices    : log-scale temperature orthogonal-triptych plots from the active EOS table
#   xh2_slices     : linear H2-fraction orthogonal-triptych plots (T13 chemistry proxy where enabled)
#   xion_slices    : log-scale hydrogen-ionization orthogonal-triptych plots (T13 chemistry proxy where enabled)
#   xhe1_slices    : linear singly ionized helium orthogonal-triptych plots (T13 chemistry proxy)
#   xhe2_slices    : linear doubly ionized helium orthogonal-triptych plots (T13 chemistry proxy)
#   gamma1_slices  : linear Gamma1 orthogonal-triptych plots from the active EOS table
#   gamma3m1_slices: linear Gamma3-1 orthogonal-triptych plots from the active EOS table
#   mu_slices      : linear mean-molecular-weight orthogonal-triptych plots
#   beta_rad_slices: linear radiation-pressure-fraction orthogonal-triptych plots from the active EOS table

default_jobs() {
  nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4
}

JOBS="${1:-$(default_jobs)}"
OUT_DIR="${2:-density_slices}"
DENS_MID_OUT_DIR="${DENS_MID_OUT_DIR:-dens_midplane_slices}"
VELNORM_OUT_DIR="${VELNORM_OUT_DIR:-${VELXYZ_OUT_DIR:-vel_norm_slices}}"
GRAV_PHI_OUT_DIR="${GRAV_PHI_OUT_DIR:-grav_phi_slices}"
CS_OUT_DIR="${CS_OUT_DIR:-c_s_slices}"
MACH_OUT_DIR="${MACH_OUT_DIR:-mach_slices}"
DIVV_OUT_DIR="${DIVV_OUT_DIR:-div_v_slices}"
PGAS_OUT_DIR="${PGAS_OUT_DIR:-pgas_slices}"
TEMP_OUT_DIR="${TEMP_OUT_DIR:-temp_slices}"
XION_OUT_DIR="${XION_OUT_DIR:-xion_slices}"
XH2_OUT_DIR="${XH2_OUT_DIR:-xh2_slices}"
XHE1_OUT_DIR="${XHE1_OUT_DIR:-xhe1_slices}"
XHE2_OUT_DIR="${XHE2_OUT_DIR:-xhe2_slices}"
GAMMA1_OUT_DIR="${GAMMA1_OUT_DIR:-gamma1_slices}"
GAMMA3M1_OUT_DIR="${GAMMA3M1_OUT_DIR:-gamma3m1_slices}"
MU_OUT_DIR="${MU_OUT_DIR:-mu_slices}"
BETA_RAD_OUT_DIR="${BETA_RAD_OUT_DIR:-beta_rad_slices}"

PLOT_SCRIPT="${PLOT_SCRIPT:-./plot_slice.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DENS_VMIN="${DENS_VMIN:-${VMIN:-1.e-10}}"   # code units
DENS_VMAX="${DENS_VMAX:-${VMAX:-1.e-2}}"    # code units
PLOT_X1_MIN="${PLOT_X1_MIN:--80}"
PLOT_X1_MAX="${PLOT_X1_MAX:-120}"
PLOT_X2_MIN="${PLOT_X2_MIN:--70}"
PLOT_X2_MAX="${PLOT_X2_MAX:-180}"
PLOT_X3_MIN="${PLOT_X3_MIN:--20}"
PLOT_X3_MAX="${PLOT_X3_MAX:-20}"
DENS_ORTHO_X_MIN="${DENS_ORTHO_X_MIN:-${PLOT_X1_MIN}}"
DENS_ORTHO_X_MAX="${DENS_ORTHO_X_MAX:-${PLOT_X1_MAX}}"
DENS_ORTHO_Y_MIN="${DENS_ORTHO_Y_MIN:-${PLOT_X2_MIN}}"
DENS_ORTHO_Y_MAX="${DENS_ORTHO_Y_MAX:-${PLOT_X2_MAX}}"
DENS_ORTHO_Z_MIN="${DENS_ORTHO_Z_MIN:-${PLOT_X3_MIN}}"
DENS_ORTHO_Z_MAX="${DENS_ORTHO_Z_MAX:-${PLOT_X3_MAX}}"
DENS_ORTHO_BH_PANEL_RMAX="${DENS_ORTHO_BH_PANEL_RMAX:-16}"
DENS_ORTHO_YOZ_Y_MIN="${DENS_ORTHO_YOZ_Y_MIN:--45}"
DENS_ORTHO_YOZ_Y_MAX="${DENS_ORTHO_YOZ_Y_MAX:--35}"
DENS_ORTHO_YOZ_Z_MIN="${DENS_ORTHO_YOZ_Z_MIN:--3}"
DENS_ORTHO_YOZ_Z_MAX="${DENS_ORTHO_YOZ_Z_MAX:-3}"
DENS_CMAP="${DENS_CMAP:-Spectral_r}"
DENS_STREAM_COLOR="${DENS_STREAM_COLOR:-black}"
DENS_STREAM_ALPHA="${DENS_STREAM_ALPHA:-0.3}"
DENS_STREAM_DENSITY="${DENS_STREAM_DENSITY:-3.375}"
DENS_STREAM_ARROWSIZE="${DENS_STREAM_ARROWSIZE:-0.35}"
DENS_STREAM_RESOLUTION="${DENS_STREAM_RESOLUTION:-128}"
VELNORM_CMAP="${VELNORM_CMAP:-${VELXYZ_CMAP:-viridis}}"
GRAV_PHI_CMAP="${GRAV_PHI_CMAP:-cividis}"
CS_CMAP="${CS_CMAP:-jet}"
MACH_CMAP="${MACH_CMAP:-magma}"
MACH_NORM="${MACH_NORM:-log}"
MACH_VMIN="${MACH_VMIN:-1.e-1}"
MACH_VMAX="${MACH_VMAX:-1.e3}"
DIVV_CMAP="${DIVV_CMAP:-coolwarm_r}"
DIVV_NORM="${DIVV_NORM:-symlog}"
DIVV_LEFT_LINTHRESH="${DIVV_LEFT_LINTHRESH:-1.e-2}"
DIVV_LEFT_VMAX="${DIVV_LEFT_VMAX:-1}"
DIVV_X1_MIN="${DIVV_X1_MIN:--100}"
DIVV_X1_MAX="${DIVV_X1_MAX:-100}"
DIVV_X2_MIN="${DIVV_X2_MIN:--100}"
DIVV_X2_MAX="${DIVV_X2_MAX:-100}"
DIVV_ORTHO_Z_MIN="${DIVV_ORTHO_Z_MIN:--10}"
DIVV_ORTHO_Z_MAX="${DIVV_ORTHO_Z_MAX:-10}"
DIVV_BH_CENTER_RMAX="${DIVV_BH_CENTER_RMAX:-10}"
DIVV_SHOW_GRID="${DIVV_SHOW_GRID:-false}"
PGAS_VMIN="${PGAS_VMIN:-1.e-12}"
PGAS_VMAX="${PGAS_VMAX:-1.e-4}"
PGAS_CMAP="${PGAS_CMAP:-viridis}"
TEMP_VMIN="${TEMP_VMIN:-3.e3}"
TEMP_VMAX="${TEMP_VMAX:-1.e8}"
TEMP_CMAP="${TEMP_CMAP:-inferno}"
XH2_VMIN="${XH2_VMIN:-0.0}"
XH2_VMAX="${XH2_VMAX:-1.0}"
XH2_CMAP="${XH2_CMAP:-viridis}"
XION_VMIN="${XION_VMIN:-1.e-12}"
XION_VMAX="${XION_VMAX:-1.0}"
XION_CMAP="${XION_CMAP:-viridis}"
XHE1_VMIN="${XHE1_VMIN:-0.0}"
XHE1_VMAX="${XHE1_VMAX:-1.0}"
XHE1_CMAP="${XHE1_CMAP:-viridis}"
XHE2_VMIN="${XHE2_VMIN:-0.0}"
XHE2_VMAX="${XHE2_VMAX:-1.0}"
XHE2_CMAP="${XHE2_CMAP:-magma}"
GAMMA1_VMIN="${GAMMA1_VMIN:-1.0}"
GAMMA1_VMAX="${GAMMA1_VMAX:-1.67}"
GAMMA1_CMAP="${GAMMA1_CMAP:-cividis}"
GAMMA3M1_VMIN="${GAMMA3M1_VMIN:-0.0}"
GAMMA3M1_VMAX="${GAMMA3M1_VMAX:-0.67}"
GAMMA3M1_CMAP="${GAMMA3M1_CMAP:-cividis}"
MU_VMIN="${MU_VMIN:-0.5}"
MU_VMAX="${MU_VMAX:-1.4}"
MU_CMAP="${MU_CMAP:-plasma}"
BETA_RAD_VMIN="${BETA_RAD_VMIN:-0.0}"
BETA_RAD_VMAX="${BETA_RAD_VMAX:-1.0}"
BETA_RAD_CMAP="${BETA_RAD_CMAP:-inferno}"
PLOT_PRODUCTS="${PLOT_PRODUCTS:-dens,dens_midplane,vel_norm,grav_phi,c_s,mach,divv,pgas,temp,xh2,xion,xhe1,xhe2,gamma1,gamma3m1,mu,beta_rad}"
# BH_ZOOM_RMAX="${BH_ZOOM_RMAX:-15}"
# CENTER_ZOOM_RMAX="${CENTER_ZOOM_RMAX:-1}"
# DENS_BH_ZOOM_RMAX="${DENS_BH_ZOOM_RMAX:-${BH_ZOOM_RMAX}}"
# DENS_CENTER_ZOOM_RMAX="${DENS_CENTER_ZOOM_RMAX:-}"
# PGAS_BH_ZOOM_RMAX="${PGAS_BH_ZOOM_RMAX:-${BH_ZOOM_RMAX}}"
# PGAS_CENTER_ZOOM_RMAX="${PGAS_CENTER_ZOOM_RMAX:-}"
# DENS_MID_BH_ZOOM_RMAX="${DENS_MID_BH_ZOOM_RMAX:-${PGAS_BH_ZOOM_RMAX}}"
# DENS_MID_CENTER_ZOOM_RMAX="${DENS_MID_CENTER_ZOOM_RMAX:-${PGAS_CENTER_ZOOM_RMAX}}"

if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "Error: plot script not found: $PLOT_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$DENS_MID_OUT_DIR" "$VELNORM_OUT_DIR" "$GRAV_PHI_OUT_DIR" "$CS_OUT_DIR" "$MACH_OUT_DIR" \
  "$DIVV_OUT_DIR" \
  "$PGAS_OUT_DIR" "$TEMP_OUT_DIR" "$XH2_OUT_DIR" "$XION_OUT_DIR" "$XHE1_OUT_DIR" "$XHE2_OUT_DIR" \
  "$GAMMA1_OUT_DIR" "$GAMMA3M1_OUT_DIR" "$MU_OUT_DIR" "$BETA_RAD_OUT_DIR"

output_current() {
  local bin_file="$1"
  local out_file="$2"
  [[ -f "$out_file" ]]
}

plot_if_needed() {
  local bin_file="$1"
  local variable="$2"
  local out_file="$3"
  shift 3

  if output_current "$bin_file" "$out_file"; then
    echo "skip existing: $out_file"
    return 0
  fi

  echo "plot: $bin_file -> $out_file"
  "$PYTHON_BIN" "$PLOT_SCRIPT" "$bin_file" "$variable" "$out_file" "$@"
}

plot_if_present() {
  local bin_file="$1"
  local required_var="$2"
  local plot_var="$3"
  local out_file="$4"
  shift 4

  if output_current "$bin_file" "$out_file"; then
    echo "skip existing: $out_file"
    return 0
  fi

  if ! bin_has_variable "$bin_file" "$required_var"; then
    echo "skip missing variable $required_var: $bin_file"
    return 0
  fi

  echo "plot: $bin_file -> $out_file"
  "$PYTHON_BIN" "$PLOT_SCRIPT" "$bin_file" "$plot_var" "$out_file" "$@"
}

bin_has_variable() {
  local bin_file="$1"
  local variable="$2"
  local header
  header="$(grep -a -m1 "^  variables:" "$bin_file" 2>/dev/null || true)"
  [[ " $header " == *" $variable "* ]]
}

bin_uses_tabulated_lte() {
  local bin_file="$1"
  grep -a -m1 -Eq '^[[:space:]]*eos[[:space:]]*=[[:space:]]*(saha_table|lte_table_hhe|lte_table_hhe_prad|lte_table_t13|lte_table_t13_prad|lte_table_hybrid_hhe_t13|lte_table_hybrid_hhe_t13_prad|lte_table_scvh_t13_union|lte_table_scvh_t13_union_prad|lte_table_scvh_t13_helm_union|lte_table_scvh_t13_helm_union_prad|lte_table_scvh_t13_cp_union|lte_table_scvh_t13_cp_union_prad|lte_table_scvh_t13_cp_helm_union|lte_table_scvh_t13_cp_helm_union_prad|lte_table_chabrier2021_t13_helm_union|lte_table_chabrier2021_t13_helm_union_prad)[[:space:]]*$' \
    "$bin_file"
}

bin_uses_h2_lte() {
  local bin_file="$1"
  grep -a -m1 -Eq '^[[:space:]]*eos[[:space:]]*=[[:space:]]*(lte_table_t13|lte_table_t13_prad|lte_table_hybrid_hhe_t13|lte_table_hybrid_hhe_t13_prad|lte_table_scvh_t13_union|lte_table_scvh_t13_union_prad|lte_table_scvh_t13_helm_union|lte_table_scvh_t13_helm_union_prad|lte_table_scvh_t13_cp_union|lte_table_scvh_t13_cp_union_prad|lte_table_scvh_t13_cp_helm_union|lte_table_scvh_t13_cp_helm_union_prad|lte_table_chabrier2021_t13_helm_union|lte_table_chabrier2021_t13_helm_union_prad)[[:space:]]*$' \
    "$bin_file"
}

bin_uses_prad_lte() {
  local bin_file="$1"
  grep -a -m1 -Eq '^[[:space:]]*eos[[:space:]]*=[[:space:]]*(lte_table_hhe_prad|lte_table_t13_prad|lte_table_hybrid_hhe_t13_prad|lte_table_scvh_t13_union_prad|lte_table_scvh_t13_helm_union_prad|lte_table_scvh_t13_cp_union_prad|lte_table_scvh_t13_cp_helm_union_prad|lte_table_chabrier2021_t13_helm_union_prad)[[:space:]]*$' \
    "$bin_file"
}

want_product() {
  local product="$1"
  [[ ",$PLOT_PRODUCTS," == *",$product,"* ]]
}

plot_one() {
  local bin_file="$1"
  local base_name stem dens_out dens_mid_out velnorm_out grav_phi_out cs_out mach_out divv_out pgas_out temp_out
  local xh2_out xion_out xhe1_out xhe2_out gamma1_out gamma3m1_out mu_out beta_rad_out
  local -a common_plot_args common_triptych_args non_density_triptych_args
  local -a dens_args dens_mid_args velnorm_args grav_phi_args cs_args pgas_args temp_args
  local -a mach_args divv_args xh2_args xion_args xhe1_args xhe2_args gamma1_args gamma3m1_args mu_args beta_rad_args
  base_name="$(basename "$bin_file")"
  stem="${base_name%.bin}"
  dens_out="${OUT_DIR}/${stem}.dens.png"
  dens_mid_out="${DENS_MID_OUT_DIR}/${stem}.dens_midplane.png"
  velnorm_out="${VELNORM_OUT_DIR}/${stem}.vel_norm.png"
  grav_phi_out="${GRAV_PHI_OUT_DIR}/${stem}.grav_phi.png"
  cs_out="${CS_OUT_DIR}/${stem}.c_s.png"
  mach_out="${MACH_OUT_DIR}/${stem}.mach.png"
  divv_out="${DIVV_OUT_DIR}/${stem}.div_v.png"
  pgas_out="${PGAS_OUT_DIR}/${stem}.pgas.png"
  temp_out="${TEMP_OUT_DIR}/${stem}.temperature.png"
  xh2_out="${XH2_OUT_DIR}/${stem}.xh2.png"
  xion_out="${XION_OUT_DIR}/${stem}.xion.png"
  xhe1_out="${XHE1_OUT_DIR}/${stem}.xhe1.png"
  xhe2_out="${XHE2_OUT_DIR}/${stem}.xhe2.png"
  gamma1_out="${GAMMA1_OUT_DIR}/${stem}.gamma1.png"
  gamma3m1_out="${GAMMA3M1_OUT_DIR}/${stem}.gamma3m1.png"
  mu_out="${MU_OUT_DIR}/${stem}.mu.png"
  beta_rad_out="${BETA_RAD_OUT_DIR}/${stem}.beta_rad.png"

  common_plot_args=(
    --no_bh_mask
  )

  common_triptych_args=(
    "${common_plot_args[@]}"
    --x1_min "$PLOT_X1_MIN" --x1_max "$PLOT_X1_MAX"
    --x2_min "$PLOT_X2_MIN" --x2_max "$PLOT_X2_MAX"
    --orthogonal_triptych
    --ortho_x_min "$DENS_ORTHO_X_MIN" --ortho_x_max "$DENS_ORTHO_X_MAX"
    --ortho_y_min "$DENS_ORTHO_Y_MIN" --ortho_y_max "$DENS_ORTHO_Y_MAX"
    --ortho_z_min "$DENS_ORTHO_Z_MIN" --ortho_z_max "$DENS_ORTHO_Z_MAX"
  )

  non_density_triptych_args=(
    "${common_triptych_args[@]}"
    --notex
  )

  dens_args=(
    -c "$DENS_CMAP" -n log --vmin "$DENS_VMIN" --vmax "$DENS_VMAX"
    "${common_triptych_args[@]}"
    --ortho_yoz_y_min "$DENS_ORTHO_YOZ_Y_MIN" --ortho_yoz_y_max "$DENS_ORTHO_YOZ_Y_MAX"
    --ortho_yoz_z_min "$DENS_ORTHO_YOZ_Z_MIN" --ortho_yoz_z_max "$DENS_ORTHO_YOZ_Z_MAX"
    # --streamlines --stream_color "$DENS_STREAM_COLOR"
    # --stream_alpha "$DENS_STREAM_ALPHA"
    # --stream_density "$DENS_STREAM_DENSITY"
    # --stream_arrowsize "$DENS_STREAM_ARROWSIZE"
    --bound_unbound_contour
    --stream_resolution "$DENS_STREAM_RESOLUTION" --notex --grid
  )
  if [[ -n "$DENS_BH_ZOOM_RMAX" ]]; then
    dens_args+=(--bh_zoom_rmax "$DENS_BH_ZOOM_RMAX")
  fi
  if [[ -n "$DENS_CENTER_ZOOM_RMAX" ]]; then
    dens_args+=(--center_zoom_rmax "$DENS_CENTER_ZOOM_RMAX")
  fi
  if want_product dens; then
    plot_if_needed "$bin_file" dens "$dens_out" "${dens_args[@]}"
  fi

  dens_mid_args=(
    -c "$DENS_CMAP" -n log --vmin "$DENS_VMIN" --vmax "$DENS_VMAX"
    "${common_plot_args[@]}"
    --x1_min "$PLOT_X1_MIN" --x1_max "$PLOT_X1_MAX"
    --x2_min "$PLOT_X2_MIN" --x2_max "$PLOT_X2_MAX"
    --notex
  )
  if [[ -n "$DENS_MID_BH_ZOOM_RMAX" ]]; then
    dens_mid_args+=(--bh_zoom_rmax "$DENS_MID_BH_ZOOM_RMAX")
  fi
  if [[ -n "$DENS_MID_CENTER_ZOOM_RMAX" ]]; then
    dens_mid_args+=(--center_zoom_rmax "$DENS_MID_CENTER_ZOOM_RMAX")
  fi
  if want_product dens_midplane; then
    plot_if_needed "$bin_file" dens "$dens_mid_out" "${dens_mid_args[@]}"
  fi

  velnorm_args=(
    -c "$VELNORM_CMAP"
    "${non_density_triptych_args[@]}"
  )
  if [[ -n "$BH_ZOOM_RMAX" ]]; then
    velnorm_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
  fi
  if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
    velnorm_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
  fi
  if want_product vel_norm; then
    plot_if_needed "$bin_file" derived:vel_norm "$velnorm_out" "${velnorm_args[@]}"
  fi

  grav_phi_args=(
    -c "$GRAV_PHI_CMAP" -n log
    "${non_density_triptych_args[@]}"
  )
  if [[ -n "$BH_ZOOM_RMAX" ]]; then
    grav_phi_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
  fi
  if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
    grav_phi_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
  fi
  if want_product grav_phi; then
    plot_if_present "$bin_file" grav_phi derived:grav_phi_abs "$grav_phi_out" \
      "${grav_phi_args[@]}"
  fi

  cs_args=(
    -c "$CS_CMAP" -n log
    "${non_density_triptych_args[@]}"
  )
  if [[ -n "$BH_ZOOM_RMAX" ]]; then
    cs_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
  fi
  if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
    cs_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
  fi
  if want_product c_s; then
    plot_if_needed "$bin_file" derived:c_s "$cs_out" "${cs_args[@]}"
  fi

  mach_args=(
    -c "$MACH_CMAP" -n "$MACH_NORM" --vmin "$MACH_VMIN" --vmax "$MACH_VMAX"
    "${non_density_triptych_args[@]}"
  )
  if [[ -n "$BH_ZOOM_RMAX" ]]; then
    mach_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
  fi
  if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
    mach_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
  fi
  if want_product mach; then
    plot_if_needed "$bin_file" derived:mach "$mach_out" "${mach_args[@]}"
  fi

  divv_args=(
    -c "$DIVV_CMAP" -n "$DIVV_NORM"
    "${common_plot_args[@]}"
    --left_linthresh "$DIVV_LEFT_LINTHRESH" --left_vmax "$DIVV_LEFT_VMAX"
    --x1_min "$DIVV_X1_MIN" --x1_max "$DIVV_X1_MAX"
    --x2_min "$DIVV_X2_MIN" --x2_max "$DIVV_X2_MAX"
    --ortho_x_min "$DIVV_X1_MIN" --ortho_x_max "$DIVV_X1_MAX"
    --ortho_y_min "$DIVV_X2_MIN" --ortho_y_max "$DIVV_X2_MAX"
    --ortho_z_min "$DIVV_ORTHO_Z_MIN" --ortho_z_max "$DIVV_ORTHO_Z_MAX"
    --orthogonal_triptych --notex
  )
  if [[ "$DIVV_SHOW_GRID" == "true" || "$DIVV_SHOW_GRID" == "1" ]]; then
    divv_args+=(--grid)
  fi
  if [[ -n "$DIVV_BH_CENTER_RMAX" ]]; then
    divv_args+=(--bh_center_rmax "$DIVV_BH_CENTER_RMAX")
  fi
  if [[ -n "$BH_ZOOM_RMAX" ]]; then
    divv_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
  fi
  if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
    divv_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
  fi
  if want_product divv || want_product div_v; then
    plot_if_needed "$bin_file" derived:div_v_panels "$divv_out" "${divv_args[@]}"
  fi

  pgas_args=(
    -c "$PGAS_CMAP" -n log --vmin "$PGAS_VMIN" --vmax "$PGAS_VMAX"
    "${non_density_triptych_args[@]}"
  )
  if [[ -n "$PGAS_BH_ZOOM_RMAX" ]]; then
    pgas_args+=(--bh_zoom_rmax "$PGAS_BH_ZOOM_RMAX")
  fi
  if [[ -n "$PGAS_CENTER_ZOOM_RMAX" ]]; then
    pgas_args+=(--center_zoom_rmax "$PGAS_CENTER_ZOOM_RMAX")
  fi
  if want_product pgas; then
    plot_if_needed "$bin_file" derived:pgas "$pgas_out" "${pgas_args[@]}"
  fi

  if bin_uses_tabulated_lte "$bin_file"; then
    temp_args=(
      -c "$TEMP_CMAP" -n log --vmin "$TEMP_VMIN" --vmax "$TEMP_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      temp_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      temp_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product temp; then
      plot_if_needed "$bin_file" derived:T "$temp_out" "${temp_args[@]}"
    fi

    xh2_args=(
      -c "$XH2_CMAP" --vmin "$XH2_VMIN" --vmax "$XH2_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      xh2_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      xh2_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product xh2 && bin_uses_h2_lte "$bin_file"; then
      plot_if_needed "$bin_file" derived:xh2 "$xh2_out" "${xh2_args[@]}"
    fi

    xion_args=(
      -c "$XION_CMAP" --vmin "$XION_VMIN" --vmax "$XION_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      xion_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      xion_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product xion; then
      plot_if_needed "$bin_file" derived:xion "$xion_out" "${xion_args[@]}"
    fi

    xhe1_args=(
      -c "$XHE1_CMAP" --vmin "$XHE1_VMIN" --vmax "$XHE1_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      xhe1_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      xhe1_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product xhe1; then
      plot_if_needed "$bin_file" derived:xhe1 "$xhe1_out" "${xhe1_args[@]}"
    fi

    xhe2_args=(
      -c "$XHE2_CMAP" --vmin "$XHE2_VMIN" --vmax "$XHE2_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      xhe2_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      xhe2_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product xhe2; then
      plot_if_needed "$bin_file" derived:xhe2 "$xhe2_out" "${xhe2_args[@]}"
    fi

    gamma1_args=(
      -c "$GAMMA1_CMAP" --vmin "$GAMMA1_VMIN" --vmax "$GAMMA1_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      gamma1_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      gamma1_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product gamma1; then
      plot_if_needed "$bin_file" derived:gamma1 "$gamma1_out" "${gamma1_args[@]}"
    fi

    gamma3m1_args=(
      -c "$GAMMA3M1_CMAP" --vmin "$GAMMA3M1_VMIN" --vmax "$GAMMA3M1_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      gamma3m1_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      gamma3m1_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product gamma3m1; then
      plot_if_needed "$bin_file" derived:gamma3m1 "$gamma3m1_out" "${gamma3m1_args[@]}"
    fi

    mu_args=(
      -c "$MU_CMAP" --vmin "$MU_VMIN" --vmax "$MU_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      mu_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      mu_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product mu; then
      plot_if_needed "$bin_file" derived:mu "$mu_out" "${mu_args[@]}"
    fi

    beta_rad_args=(
      -c "$BETA_RAD_CMAP" --vmin "$BETA_RAD_VMIN" --vmax "$BETA_RAD_VMAX"
      "${non_density_triptych_args[@]}"
    )
    if [[ -n "$BH_ZOOM_RMAX" ]]; then
      beta_rad_args+=(--bh_zoom_rmax "$BH_ZOOM_RMAX")
    fi
    if [[ -n "$CENTER_ZOOM_RMAX" ]]; then
      beta_rad_args+=(--center_zoom_rmax "$CENTER_ZOOM_RMAX")
    fi
    if want_product beta_rad && bin_uses_prad_lte "$bin_file"; then
      plot_if_needed "$bin_file" derived:beta_rad "$beta_rad_out" "${beta_rad_args[@]}"
    fi
  fi
}

export OUT_DIR DENS_MID_OUT_DIR VELNORM_OUT_DIR GRAV_PHI_OUT_DIR CS_OUT_DIR MACH_OUT_DIR DIVV_OUT_DIR PGAS_OUT_DIR TEMP_OUT_DIR
export XH2_OUT_DIR XION_OUT_DIR XHE1_OUT_DIR XHE2_OUT_DIR GAMMA1_OUT_DIR GAMMA3M1_OUT_DIR
export MU_OUT_DIR BETA_RAD_OUT_DIR PLOT_SCRIPT PYTHON_BIN
export PLOT_X1_MIN PLOT_X1_MAX PLOT_X2_MIN PLOT_X2_MAX PLOT_X3_MIN PLOT_X3_MAX
export DENS_VMIN DENS_VMAX DENS_CMAP DENS_STREAM_COLOR DENS_STREAM_ALPHA
export DENS_STREAM_DENSITY DENS_STREAM_ARROWSIZE DENS_STREAM_RESOLUTION
export DENS_ORTHO_X_MIN DENS_ORTHO_X_MAX DENS_ORTHO_Y_MIN DENS_ORTHO_Y_MAX
export DENS_ORTHO_Z_MIN DENS_ORTHO_Z_MAX DENS_ORTHO_BH_PANEL_RMAX
export DENS_ORTHO_YOZ_Y_MIN DENS_ORTHO_YOZ_Y_MAX DENS_ORTHO_YOZ_Z_MIN DENS_ORTHO_YOZ_Z_MAX
export DENS_BH_ZOOM_RMAX DENS_CENTER_ZOOM_RMAX
export DENS_MID_BH_ZOOM_RMAX DENS_MID_CENTER_ZOOM_RMAX
export VELNORM_CMAP GRAV_PHI_CMAP CS_CMAP MACH_CMAP MACH_NORM MACH_VMIN MACH_VMAX
export DIVV_CMAP DIVV_NORM DIVV_LEFT_LINTHRESH DIVV_LEFT_VMAX
export DIVV_X1_MIN DIVV_X1_MAX DIVV_X2_MIN DIVV_X2_MAX DIVV_ORTHO_Z_MIN DIVV_ORTHO_Z_MAX
export DIVV_BH_CENTER_RMAX DIVV_SHOW_GRID
export PGAS_VMIN PGAS_VMAX PGAS_CMAP PGAS_BH_ZOOM_RMAX PGAS_CENTER_ZOOM_RMAX
export TEMP_VMIN TEMP_VMAX TEMP_CMAP
export XH2_VMIN XH2_VMAX XH2_CMAP
export XION_VMIN XION_VMAX XION_CMAP
export XHE1_VMIN XHE1_VMAX XHE1_CMAP
export XHE2_VMIN XHE2_VMAX XHE2_CMAP
export GAMMA1_VMIN GAMMA1_VMAX GAMMA1_CMAP
export GAMMA3M1_VMIN GAMMA3M1_VMAX GAMMA3M1_CMAP
export MU_VMIN MU_VMAX MU_CMAP
export BETA_RAD_VMIN BETA_RAD_VMAX BETA_RAD_CMAP
export PLOT_PRODUCTS BH_ZOOM_RMAX CENTER_ZOOM_RMAX
export -f output_current plot_if_needed plot_if_present bin_has_variable bin_uses_tabulated_lte
export -f bin_uses_h2_lte bin_uses_prad_lte want_product
export -f plot_one

mapfile -d '' BIN_FILES < <(find "$(pwd)" -maxdepth 1 -type f -name '*.bin' -print0 | sort -z)

if (( ${#BIN_FILES[@]} == 0 )); then
  echo "No .bin files found in $(pwd)"
  exit 0
fi

printf '%s\0' "${BIN_FILES[@]}" | xargs -0 -n1 -P "$JOBS" bash -c 'plot_one "$1"' _

echo "Done. Images are in: $OUT_DIR, $DENS_MID_OUT_DIR, $VELNORM_OUT_DIR, $GRAV_PHI_OUT_DIR, $CS_OUT_DIR, $MACH_OUT_DIR, "\
"$DIVV_OUT_DIR, $PGAS_OUT_DIR, $TEMP_OUT_DIR, $XION_OUT_DIR, $XHE1_OUT_DIR, $XHE2_OUT_DIR, "\
"$GAMMA1_OUT_DIR, $GAMMA3M1_OUT_DIR, $MU_OUT_DIR, $BETA_RAD_OUT_DIR"
