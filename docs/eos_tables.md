# EOS tables

`<hydro>` or `<mhd>` with `eos = saha_table` or one of the `lte_table_*` names reads a
tabulated equation of state from `table = <file>`. The physics, the keys and the table
format are in `docs/wiki/Tabulated-EOS.md`. This page covers where the tables come from
and where they live.

## What is in git

* The generators: `scripts/generate_lte_table.py` (every `lte_table_*` table, with the
  backends `scripts/chabrier2021_eos.py` and `scripts/scvh95_eos.py`) and
  `scripts/generate_saha_table.py` (`saha_hydrogen.table`).
* Their raw inputs, which are small (40 MB):
  * `scripts/chabrier2021_data/`: the H, He and H/He mixture tables of Chabrier et al.
    (2019) and Chabrier & Debras (2021), byte-identical to the `DirEOS2021` archive at
    http://perso.ens-lyon.fr/gilles.chabrier/DirEOS/DirEOS2021 (sha256
    `45f316790ce20d5d1ce0abee4db308521b5bfdc5526d0997141a2784834feeff`,
    retrieved 2026-09-26; the pure H and He tables are also in `DirEOS2019`);
  * `scripts/scvh95_data/`: the Saumon, Chabrier & van Horn (1995) tables, from the
    ApJS CD-ROM volume 5 (URL in `scvh95_eos.py`).

## Generating

```
python3 scripts/generate_lte_table.py                        # every job
python3 scripts/generate_lte_table.py chabrier2021_t13_helm_union_prad_640_X745Y255
python3 scripts/generate_lte_table.py <job> --output-dir $ATHENAK_DATA/eos_tables
python3 scripts/generate_saha_table.py                       # eos_tables/saha_hydrogen.table
```

Needs Python 3 with numpy and scipy. An unknown job name prints the list. Output goes to
`eos_tables/` unless `--output-dir` is given. A 640 × 640 job takes about a minute on one
core. The jobs, all 640 × 640 in $(\ln\rho, \ln T)$:

| job | X | Y |
|---|---|---|
| `lte_t13_prad` | 0.70 | 0.30 |
| `scvh_t13_cp_helm_union_prad_640` | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_640` (no radiation) | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_prad_640` | 0.70 | 0.30 |
| `chabrier2021_t13_helm_union_prad_640_X745Y255` | 0.745 | 0.255 |
| `chabrier2021_t13_helm_union_prad_640_X620Y380` | 0.62 | 0.38 |
| `chabrier2021_t13_helm_union_prad_640_X550Y450` | 0.55 | 0.45 |
| `chabrier2021_t13_helm_union_prad_640_X200Y800` | 0.20 | 0.80 |
| `chabrier2021_t13_helm_union_prad_640_X038Y062` | 0.38 | 0.62 |
| `chabrier2021_t13_helm_union_prad_640_X000Y100` | 0 | 1 |

The generator requires X + Y = 1: metals are folded into helium.

## Where they live and how decks name them

Decks name EOS tables by a relative path, `table = eos_tables/<file>` (or
`../../eos_tables/<file>` from a deck's own directory). AthenaK looks for a relative path
in the working directory, then under `$ATHENAK_DATA`, then under `<source tree>/data`
(`src/utils/data_path.hpp`). So a table kept outside the repo, in
`$ATHENAK_DATA/eos_tables/`, is found by a deck run from any directory.

## Sharing

Share them as GitHub release assets or a
Zenodo record, with a `SHA256SUMS` file, the generator commit and the job names. Anyone
with the repository can also regenerate them from the inputs above.

## Citations

* Tomida, K., et al. 2013, ApJ, 763, 6 (Appendix 1: the `t13` chemistry)
* Saumon, D., Chabrier, G., & van Horn, H. M. 1995, ApJS, 99, 713 (SCvH)
* Chabrier, G., Mazevet, S., & Soubiran, F. 2019, ApJ, 872, 51
* Chabrier, G., & Debras, F. 2021, ApJ, 917, 4
* Timmes, F. X., & Swesty, F. D. 2000, ApJS, 126, 501, and Timmes, F. X., & Arnett, D.
  1999, ApJS, 125, 277 (HELM electron-positron gas)

`docs/wiki/Tabulated-EOS.md` (Further reading) lists what each model takes from them.
