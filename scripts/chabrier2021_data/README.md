# Chabrier et al. dense H/He EOS tables (not included)

`scripts/generate_lte_table.py` builds the dense-fluid part of the H/He equation of state
table `chabrier2021_t13_helm_union_prad_640.table` from five tables published by
G. Chabrier and collaborators.  The authors make the tables publicly available, but they
are distributed without a license that permits redistribution, so this repository does not
include them.  Download them yourself and place them in this directory before you run the
generator.

| File | Content | Reference |
|---|---|---|
| `TABLEEOS_2021_TP_Y0275_v1` | H/He mixture with H-He interactions, `Y = 0.275` | Chabrier & Debras (2021) |
| `TABLEEOS_2021_TP_Y0292_v1` | H/He mixture with H-He interactions, `Y = 0.292` | Chabrier & Debras (2021) |
| `TABLEEOS_2021_TP_Y0297_v1` | H/He mixture with H-He interactions, `Y = 0.297` | Chabrier & Debras (2021) |
| `TABLE_H_TP_v1` | pure hydrogen | Chabrier, Mazevet & Soubiran (2019) |
| `TABLE_HE_TP_v1` | pure helium | Chabrier, Mazevet & Soubiran (2019) |

All five are the version `v1` (June 2021) tables on the `(log T, log P)` grid, with
`2 <= log10(T / K) <= 8` and `-9 <= log10(P / GPa) <= 13`.  They are contained in the
archive `DirEOS2021.tar.gz` on the authors' web page, which also holds tables that the
generator does not use.

## How to get them

The script `get_eos_table.sh` in the top-level directory of the repository does all of the
steps below and then builds the EOS table.  To do it by hand, run from the repository root:

```bash
cd scripts/chabrier2021_data
curl -LO http://perso.ens-lyon.fr/gilles.chabrier/DirEOS/DirEOS2021.tar.gz
tar -xzf DirEOS2021.tar.gz
cp DirEOS2021/TABLEEOS_2021_TP_Y0275_v1 DirEOS2021/TABLEEOS_2021_TP_Y0292_v1 \
   DirEOS2021/TABLEEOS_2021_TP_Y0297_v1 DirEOS2021/TABLE_H_TP_v1 \
   DirEOS2021/TABLE_HE_TP_v1 .
sha256sum -c <<'EOF'
c4995d114affedddf421b57b847ad4872699e9526f32b4b335013ffcbfb0b938  TABLEEOS_2021_TP_Y0275_v1
436fe580aac6e8572159b59322bf7baf6afb43ec91630779239698bbeaf57a7d  TABLEEOS_2021_TP_Y0292_v1
3a07460158c1b7feeea9484a94684148ac94ebc9b6df4b2f194f8024b1aa50bb  TABLEEOS_2021_TP_Y0297_v1
0215333ff5d727e9059dc474ef2b615e3dee62af5a16a2dab037ee181a2ff892  TABLE_H_TP_v1
fae4720798f6189f5d10020a996c264b72d72406be40b15175a0b9799efc932b  TABLE_HE_TP_v1
EOF
cd ../..
```

The checksums are those of the files used to build the table of the paper.  If a check
fails, the authors have released a newer version of the tables, and the generated EOS
table will differ from the one of the paper.

The same tables are also distributed as a `.tar.gz` package with Chabrier & Debras
(2021) on the journal web site.  The authors' README names `chabrier@ens-lyon.fr` as
the contact for requests.

Then build the EOS table:

```bash
python3 scripts/generate_lte_table.py chabrier2021_t13_helm_union_prad_640
```

The generator looks for the five files in this directory by default
(`scripts/chabrier2021_eos.py`).

## References

Please cite both papers when you use EOS tables built from these data.

* Chabrier, G., Mazevet, S., & Soubiran, F. 2019, ApJ, 872, 51 (arXiv:1902.01852)
* Chabrier, G., & Debras, F. 2021, ApJ, 917, 4 (arXiv:2107.04434)
