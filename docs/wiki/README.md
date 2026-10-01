<!-- Source of the wiki Home page. Wiki page names: README.md -> Home; docs/eos_tables.md -> EOS-Tables; docs/remap_usage.md -> Remap-Usage; docs/wiki/<Name>.md -> <Name>. All links use wiki page names without `.md`. -->
# AthenaK-HXJ wiki

This wiki explains the physics modules and problem generators that this fork adds to
[AthenaK](https://github.com/IAS-Astrophysics/athenak). Parameters, defaults and refusals are
checked against the code in this repository.

## Which page do I need?

| I want to ... | read |
| --- | --- |
| let each block of the mesh take its own time step | [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping) |
| start a new run from the state of an old one, on a different mesh | [Remap usage](Remap-Usage), then [Remapping](Remapping) |
| use ionization, H2 dissociation or radiation pressure in the gas | [Tabulated EOS](Tabulated-EOS) and [EOS tables](EOS-Tables) |
| add self-gravity to a hydro or MHD run | [Multigrid Self-Gravity](Multigrid-Self-Gravity) |
| keep the gas temperature right in fast, cold flow | [Dual Energy](Dual-Energy) |

## All pages

Each topic has a main page for users and an implementation-notes page for people who want to
change the code.

| topic | main page | implementation notes |
| --- | --- | --- |
| Local adaptive time stepping (LAT) | [Local Adaptive Time Stepping](Local-Adaptive-Time-Stepping) | [notes](Local-Adaptive-Time-Stepping-Implementation-Notes) |
| Restart remap | [Remap usage](Remap-Usage) (how to use it), [Remapping](Remapping) (how it works) | [notes](Remapping-Implementation-Notes) |
| Tabulated EOS (LTE / Saha) | [Tabulated EOS](Tabulated-EOS), [EOS tables](EOS-Tables) (building the tables) | [notes](Tabulated-EOS-Implementation-Notes) |
| Multigrid self-gravity | [Multigrid Self-Gravity](Multigrid-Self-Gravity) | [notes](Multigrid-Self-Gravity-Implementation-Notes) |
| Dual energy | [Dual Energy](Dual-Energy) | [notes](Dual-Energy-Implementation-Notes) |

## How a main page is laid out

1. **Summary**: what the module does, when to use it and when not to.
2. **Quick start**: a few lines of input you can copy.
3. **Full parameter table**: every key with its type and default.
4. **How it works**: the idea, in plain words.
5. **Practical guidance**: what is refused, how to check that it works, common problems.

## Where things live

- Input decks: `inputs/<module>/`. Test decks: `tst/inputs/`. Tests: `tst/test_suite/`, run
  with `tst/run_test_suite.py`.
- Table generators: `scripts/generate_lte_table.py` and `scripts/generate_saha_table.py`. The
  tables themselves are not in git: see [EOS tables](EOS-Tables).
