"""athenak_rt: radiative post-processing of AthenaK snapshots.

Modes: grey tau=1 photosphere (``tau1``), grey formal transfer (``grey``),
grey transfer with a thermalization depth (``grey-therm``) and multifrequency
H/He continuum transfer (``multifreq``).  See README.md and ``python -m athenak_rt -h``.
"""

__version__ = "1.0.0"

from .config import MODES, RTSettings  # noqa: E402,F401
from .pipeline import RTResult, process_snapshot, run  # noqa: E402,F401
