Official SCvH 1995 source tables used by `scripts/scvh95_eos.py`.

Files in this directory are the original paper-distributed inputs from:

- Saumon, Chabrier, and Van Horn 1995, ApJS, 99, 713
- AAS CD-ROM distribution:
  `https://aas.org/sites/default/files/cdrom/volume5/volume5/apjs/v99/p713`

Included files:

- `readme.doc`
- `table.doc`
- `read.f`
- `rho_crit.dat`
- `h_tab_i.dat`
- `h_tab_p1.dat`
- `h_tab_p2.dat`
- `he_tab_i.dat`

These are the native SCvH tabulations. AthenaK does not regenerate the SCvH
physics from scratch; it resamples and mixes these official source tables onto
the local `(ln rho, ln T)` export grid.
