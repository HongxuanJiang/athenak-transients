#!/usr/bin/env bash
# Download the Chabrier et al. dense H/He EOS data and build the tabulated H/He equation of
# state used by the TDE workflow, eos_tables/chabrier2021_t13_helm_union_prad_640.table.
#
# The Chabrier et al. tables are not part of this repository because the authors distribute
# them without a license that permits redistribution (see scripts/chabrier2021_data/README.md).
# This script fetches them from the authors' web page, checks them, and runs the generator.
#
# Usage:  ./get_eos_table.sh [--output-dir DIR] [--keep-download]
#   --output-dir DIR   write the table to DIR instead of eos_tables/
#   --keep-download    keep DirEOS2021.tar.gz and its extracted directory
#
# Requirements: bash, curl or wget, tar, sha256sum or shasum, python3 with numpy and scipy.
set -euo pipefail

JOB=chabrier2021_t13_helm_union_prad_640
URL=http://perso.ens-lyon.fr/gilles.chabrier/DirEOS/DirEOS2021.tar.gz
# SHA-256 of the numerical part (from <fieldsend> to the end of the file) of the table used
# in Jiang et al. (ApJS).  The text header may carry extra metadata lines from newer
# generator versions, so only the numbers are compared.
PAPER_TABLE_DATA_SHA256=4281b2f976ef00d4c758d1bf9d8062c71e3c891fd9e474a95011c5fae42f722c

OUTPUT_DIR=""
KEEP_DOWNLOAD=0
while [ $# -gt 0 ]; do
  case "$1" in
    --output-dir) OUTPUT_DIR=${2:?--output-dir needs a directory}; shift 2 ;;
    --keep-download) KEEP_DOWNLOAD=1; shift ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA_DIR="$ROOT/scripts/chabrier2021_data"
case "$OUTPUT_DIR" in ""|/*) ;; *) OUTPUT_DIR="$PWD/$OUTPUT_DIR" ;; esac
cd "$ROOT"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# ---- 1. Python requirements --------------------------------------------------------------
python3 -c 'import numpy, scipy' 2>/dev/null || {
  echo "python3 with numpy and scipy is required (pip install numpy scipy)." >&2; exit 1; }

# ---- 2. Chabrier et al. tables -----------------------------------------------------------
FILES=(TABLEEOS_2021_TP_Y0275_v1 TABLEEOS_2021_TP_Y0292_v1 TABLEEOS_2021_TP_Y0297_v1
       TABLE_H_TP_v1 TABLE_HE_TP_v1)
SUMS=(c4995d114affedddf421b57b847ad4872699e9526f32b4b335013ffcbfb0b938
      436fe580aac6e8572159b59322bf7baf6afb43ec91630779239698bbeaf57a7d
      3a07460158c1b7feeea9484a94684148ac94ebc9b6df4b2f194f8024b1aa50bb
      0215333ff5d727e9059dc474ef2b615e3dee62af5a16a2dab037ee181a2ff892
      fae4720798f6189f5d10020a996c264b72d72406be40b15175a0b9799efc932b)

have_all=1
for f in "${FILES[@]}"; do [ -f "$DATA_DIR/$f" ] || have_all=0; done

if [ "$have_all" = 0 ]; then
  echo "Downloading $URL"
  mkdir -p "$DATA_DIR"
  tarball="$DATA_DIR/DirEOS2021.tar.gz"
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 -o "$tarball" "$URL"
  elif command -v wget >/dev/null 2>&1; then wget -O "$tarball" "$URL"
  else echo "curl or wget is required to download the tables." >&2; exit 1; fi
  tar -xzf "$tarball" -C "$DATA_DIR"
  for f in "${FILES[@]}"; do cp "$DATA_DIR/DirEOS2021/$f" "$DATA_DIR/$f"; done
  if [ "$KEEP_DOWNLOAD" = 0 ]; then rm -rf "$tarball" "$DATA_DIR/DirEOS2021"; fi
else
  echo "Using the Chabrier et al. tables already in $DATA_DIR"
fi

mismatch=0
for i in "${!FILES[@]}"; do
  got=$(sha256 "$DATA_DIR/${FILES[$i]}")
  if [ "$got" != "${SUMS[$i]}" ]; then
    echo "checksum differs for ${FILES[$i]}" >&2; mismatch=1
  fi
done
if [ "$mismatch" = 1 ]; then
  echo "The authors have released tables that differ from the v1 files used for the paper." >&2
  echo "The table will still be built, but it will not match the one of the paper." >&2
fi

# ---- 3. Build the EOS table --------------------------------------------------------------
if [ -n "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
  python3 scripts/generate_lte_table.py "$JOB" --output-dir "$OUTPUT_DIR"
  TABLE="$OUTPUT_DIR/$JOB.table"
else
  python3 scripts/generate_lte_table.py "$JOB"
  TABLE="$ROOT/eos_tables/$JOB.table"
fi

[ -f "$TABLE" ] || { echo "the generator did not write $TABLE" >&2; exit 1; }
data_sha=$(python3 -c 'import hashlib, sys
b = open(sys.argv[1], "rb").read()
print(hashlib.sha256(b[b.find(b"<fieldsend>"):]).hexdigest())' "$TABLE")
if [ "$data_sha" = "$PAPER_TABLE_DATA_SHA256" ]; then
  echo "Wrote $TABLE (its values are identical to the table used in the paper)."
else
  echo "Wrote $TABLE (its values differ from the table used in the paper)."
fi
