# EOS tables

A `<hydro>` or `<mhd>` block with `eos = saha_table` or one of the `lte_table_*` names reads a
tabulated equation of state from the file named by `table = <file>`. The physics, the input
keys and the table format are described in [Tabulated EOS](Tabulated-EOS). This page covers
where the tables come from, how to build them, and where the code looks for them.

The repository contains no `.table` file (`eos_tables/*.table` is git-ignored). You build the
table you need with the generator scripts. The table used by the TDE example decks can also be
downloaded from the release assets of this repository.

## What is in git

* The generators: `scripts/generate_lte_table.py` (every `lte_table_*` table, with the backends
  `scripts/chabrier2021_eos.py` and `scripts/scvh95_eos.py`) and
  `scripts/generate_saha_table.py` (`saha_hydrogen.table`).
* `get_eos_table.sh` in the top-level directory: downloads the Chabrier et al. data and builds
  the default table (next section).
* `scripts/scvh95_data/` (about 1 MB): the Saumon, Chabrier & van Horn (1995) tables as
  distributed with the ApJS paper (`h_tab_i.dat`, `h_tab_p1.dat`, `h_tab_p2.dat`,
  `he_tab_i.dat`, `rho_crit.dat`, the original `readme.doc`, `table.doc` and `read.f`, and a
  `README.md` with the source URL). Only the job `scvh_t13_cp_helm_union_prad_640` reads them.
* `scripts/chabrier2021_data/` holds only a `README.md`. The tables of Chabrier et al. (2019)
  and Chabrier & Debras (2021) are not in git, because the authors distribute them without a
  licence that permits redistribution. The README lists the five files that are needed, their
  SHA-256 sums and the manual download steps.

## Building the default table

The example decks and most tests use `chabrier2021_t13_helm_union_prad_640.table`. From the
repository root:

```bash
./get_eos_table.sh [--output-dir DIR] [--keep-download]
```

The script needs bash, curl or wget, tar, sha256sum or shasum, and Python 3 with numpy and
scipy. It

1. downloads `DirEOS2021.tar.gz` from the authors' web page
   (`http://perso.ens-lyon.fr/gilles.chabrier/DirEOS/DirEOS2021.tar.gz`) and extracts the five
   tables into `scripts/chabrier2021_data/`, unless they are already there
   (`--keep-download` keeps the archive and its extracted directory);
2. checks the SHA-256 sum of each file and warns if the authors have released tables that
   differ from the `v1` files used for the paper (the table is still built);
3. runs the generator for the job `chabrier2021_t13_helm_union_prad_640`, which writes
   `eos_tables/chabrier2021_t13_helm_union_prad_640.table` (or the file in `DIR`);
4. reports whether the numerical values of the new table are identical to those of the table
   used in the paper.

## The table jobs

`scripts/generate_lte_table.py` contains ten jobs. Every job is a 640 x 640 grid, uniform in
ln(rho) and ln(T). The output file is the job name plus `.table`, except for `lte_t13_prad`,
which writes `lte_t13_prad_eos.table`. Use the `eos` name in the third column with the file.

| job | X | Y | `eos` name | log10 rho (g/cm^3) | log10 T (K) |
| --- | --- | --- | --- | --- | --- |
| `lte_t13_prad` | 0.70 | 0.30 | `lte_table_t13_prad` | -20 to 2 | -0.3 to 9 |
| `scvh_t13_cp_helm_union_prad_640` | 0.70 | 0.30 | `lte_table_scvh_t13_cp_helm_union_prad` | -20 to 2 | 0 to 9 |
| `chabrier2021_t13_helm_union_prad_640` | 0.70 | 0.30 | `lte_table_chabrier2021_t13_helm_union_prad` | -20 to 3 | 0 to 10 |
| `chabrier2021_t13_helm_union_640` (no radiation pressure) | 0.70 | 0.30 | `lte_table_chabrier2021_t13_helm_union` | -20 to 3 | 0 to 10 |
| `chabrier2021_t13_helm_union_prad_640_X745Y255` | 0.745 | 0.255 | `lte_table_chabrier2021_t13_helm_union_prad` | -20 to 4 | 0 to 10 |
| `chabrier2021_t13_helm_union_prad_640_X620Y380` | 0.62 | 0.38 | same | -20 to 4 | 0 to 10 |
| `chabrier2021_t13_helm_union_prad_640_X550Y450` | 0.55 | 0.45 | same | -20 to 4 | 0 to 10 |
| `chabrier2021_t13_helm_union_prad_640_X038Y062` | 0.38 | 0.62 | same | -20 to 4 | 0 to 10 |
| `chabrier2021_t13_helm_union_prad_640_X200Y800` | 0.20 | 0.80 | same | -20 to 4 | 0 to 10 |
| `chabrier2021_t13_helm_union_prad_640_X000Y100` | 0 | 1 | same | -20 to 5 | 0 to 10 |

Unless stated otherwise, the tables include radiation pressure (`_prad`). The suffix of a
composition variant gives X and Y in thousandths (`X620Y380` is X = 0.62 and Y = 0.38), with
one exception: `X038Y062` is X = 0.38 and Y = 0.62. Use the X and Y columns above. The generator
requires X + Y = 1, so metals are folded into helium. A comment in the generator gives
(X_H, Y_He, X_CNO) = (0.38, 0.60, 0.02) for `X038Y062` and (0.00, 0.98, 0.02) for `X000Y100`.

The hydrogen-only table `saha_hydrogen.table` (`eos = saha_table`) comes from
`scripts/generate_saha_table.py`. It is a 256 x 512 grid with log10 rho from -20 to 2 and
log10 T from -8 to 12.

The jobs with `chabrier2021` in the name need the five Chabrier files in
`scripts/chabrier2021_data/` and stop with an error if one is missing. The other two jobs need
only the files in git (the SCvH job reads `scripts/scvh95_data/`).

## Generating tables

```
python3 scripts/generate_lte_table.py                        # every job
python3 scripts/generate_lte_table.py chabrier2021_t13_helm_union_prad_640_X745Y255
python3 scripts/generate_lte_table.py <job> --output-dir $ATHENAK_DATA/eos_tables
python3 scripts/generate_saha_table.py                       # eos_tables/saha_hydrogen.table
```

An unknown job name stops the script and prints the list of known jobs. Output goes to
`eos_tables/` in the repository unless `--output-dir` is given. For a quick test of the setup,
`--nrho N --ntemp N` replaces the grid size of the selected jobs. A 640 x 640 job takes about
one to two minutes.

## Where decks look for tables

A relative `table` path is looked up in the working directory first, then under
`$ATHENAK_DATA`, then under `<source tree>/data` (`src/utils/data_path.hpp`). A table kept in
`$ATHENAK_DATA/eos_tables/` is therefore found by a deck with `table = eos_tables/<file>` that
runs from any directory. The TDE example decks use
`table = ../chabrier2021_t13_helm_union_prad_640.table`, one directory above the directory
each step runs in (see `inputs/TDE_examples/README.md`). An absolute path is used as it is,
and any deck value can be overridden on the command line, for example `hydro/table=<path>`.

## Sharing

If you pass a table on, publish it (for example as a GitHub release asset or a Zenodo
record) together with a `SHA256SUMS` file, the generator commit and the job name. Anyone with
the repository can also regenerate it from the inputs above.

## Citations

* Tomida, K., et al. 2013, ApJ, 763, 6 (Appendix 1: the `t13` chemistry)
* Saumon, D., Chabrier, G., & van Horn, H. M. 1995, ApJS, 99, 713 (SCvH)
* Chabrier, G., Mazevet, S., & Soubiran, F. 2019, ApJ, 872, 51
* Chabrier, G., & Debras, F. 2021, ApJ, 917, 4
* Timmes, F. X., & Swesty, F. D. 2000, ApJS, 126, 501, and Timmes, F. X., & Arnett, D.
  1999, ApJS, 125, 277 (HELM electron-positron gas)

The [Tabulated EOS](Tabulated-EOS#further-reading) page lists what each model takes from them.
