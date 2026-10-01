# AthenaK-HXJ wiki

This wiki documents the physics modules and problem generators that this fork adds to
[AthenaK](https://github.com/IAS-Astrophysics/athenak).

## Pages

| page | what it covers |
|---|---|
| [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping.md) ([notes](Local-Adaptive-Time-Stepping-Implementation-Notes.md)) | per-block time-step bins (LAT): binning, ticks and windows, partition and rebalance, which modules support it |
| [Remapping](Remapping.md) ([notes](Remapping-Implementation-Notes.md)) | loading one run's state onto another mesh |
| [Tabulated EOS](Tabulated-EOS.md) ([notes](Tabulated-EOS-Implementation-Notes.md)) | LTE/Saha tables, their generator, composition tables |
| [Multigrid Self-Gravity](Multigrid-Self-Gravity.md) ([notes](Multigrid-Self-Gravity-Implementation-Notes.md)) | the Poisson solver, its coupling, and what happens under LAT |
| [Dual Energy](Dual-Energy.md) ([notes](Dual-Energy-Implementation-Notes.md)) | the auxiliary internal-energy variable and the solvers that carry it |

## Where things live

- Test decks: `inputs/<module>/`; test drivers and their notes: `tst/run_test_suite.py`,
  `tst/test_suite/`.
- Table generators: `scripts/generate_lte_table.py`. Tables are not in git: `docs/eos_tables.md`.

## How to read a page

Each main page opens with a short summary and a quick start, then gives the full parameter
table, a conceptual explanation, and practical guidance (what is refused, how to check it works,
common problems). Code-level detail (function names, code map, edge cases, known issues, tests)
is in the matching `<Name>-Implementation-Notes` page.
