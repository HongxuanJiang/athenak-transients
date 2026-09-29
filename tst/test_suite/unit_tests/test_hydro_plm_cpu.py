"""Thermodynamic PLM invariants and chunked/LAT hydro flux regression."""

from pathlib import Path

import pytest
import test_suite.testutils as testutils


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("fofc", [False, True])
def test_hydro_plm(dual, fofc):
    repo = Path(__file__).resolve().parents[3]
    table = repo / "eos_tables/chabrier2021_t13_helm_union_640.table"
    testutils.run(
        str(repo / "tst/inputs/hydro_plm_unit.athinput"),
        [
            f"hydro/table={table}",
            f"hydro/dual_energy={str(dual).lower()}",
            f"hydro/fofc={str(fofc).lower()}",
            f"mesh/nghost={3 if fofc else 2}",
        ],
    )
