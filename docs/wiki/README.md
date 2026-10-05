# AthenaK-HXJ wiki

This wiki documents the physics modules and problem generators that this fork adds to
[AthenaK](https://github.com/IAS-Astrophysics/athenak).

## Pages

| page | what it covers |
|---|---|
| [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping.md) ([notes](Local-Adaptive-Time-Stepping-Implementation-Notes.md)) | per-block time-step bins (LAT): binning, ticks and windows, partition and rebalance, which modules support it |
| [Remapping](Remapping.md) ([notes](Remapping-Implementation-Notes.md)) | how loading one run's state onto another mesh works. To run a remap, see [Remap Usage](../remap_usage.md) |
| [Tabulated EOS](Tabulated-EOS.md) ([notes](Tabulated-EOS-Implementation-Notes.md)) | LTE/Saha tables, their generator, composition tables |
| [Multigrid Self-Gravity](Multigrid-Self-Gravity.md) ([notes](Multigrid-Self-Gravity-Implementation-Notes.md)) | the Poisson solver, its coupling, and what happens under LAT |
| [Dual Energy](Dual-Energy.md) ([notes](Dual-Energy-Implementation-Notes.md)) | the auxiliary internal-energy variable and the solvers that carry it |

## Where things live

- Test decks: `inputs/<module>/`; test drivers and their notes: `tst/run_test_suite.py`,
  `tst/test_suite/`.
- Table generators: `scripts/generate_lte_table.py`. Tables are not in git: [EOS Tables](../eos_tables.md).
- Remap user guide: [Remap Usage](../remap_usage.md).

## How to read a page

Each main page opens with a short summary and a quick start, then gives the full parameter
table, a conceptual explanation, and practical guidance (what is refused, how to check it works,
common problems). A short developer overview (how the pieces fit, key files, known issues, how to test) is in the
matching `<Name>-Implementation-Notes` page.
