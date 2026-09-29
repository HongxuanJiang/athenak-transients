# AthenaK-HXJ wiki

This wiki documents the physics modules and problem generators that this fork adds to
[AthenaK](https://github.com/IAS-Astrophysics/athenak).

## Pages

| page | what it covers |
|---|---|
| [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping.md) | per-block time-step bins (LAT): binning, ticks and windows, partition and rebalance, which modules support it |
| [Remapping](Remapping.md) | loading one run's state onto another mesh |
| [Tabulated EOS](Tabulated-EOS.md) | LTE/Saha tables, their generator, composition tables |
| [Multigrid Self-Gravity](Multigrid-Self-Gravity.md) | the Poisson solver, its coupling, and what happens under LAT |
| [Dual Energy](Dual-Energy.md) | the auxiliary internal-energy variable and the solvers that carry it |

## Where things live

- Test decks: `inputs/<module>/`; test drivers and their notes: `tst/run_test_suite.py`,
  `tst/test_suite/`.
- Table generators: `scripts/generate_lte_table.py`. Tables are not in git: `docs/eos_tables.md`.

## How to read a page

Each page has the same sections: Summary, Physics and algorithm, Code map, Configuration
(every key with its default and where it is read), How it runs, Interactions (LAT,
AMR/SMR, restart, FOFC, units, other modules), Limitations and known issues, Tests, an example
deck fragment, References. Statements the author could not verify against the code are marked
"(unverified)".
