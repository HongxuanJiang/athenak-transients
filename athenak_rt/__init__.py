"""athenak_rt: radiative post-processing of AthenaK snapshots.

Modes: grey tau=1 photosphere (``tau1``), grey formal transfer (``grey``),
grey transfer with a thermalization depth (``grey-therm``) and multifrequency
H/He continuum transfer (``multifreq``).  Runs are driven by a parameter file,
``python -m athenak_rt PARAMETER_FILE``; see README.md and docs/athenak_rt.md.
"""

__version__ = "1.0.0"

from .config import MODES, RTSettings  # noqa: E402,F401
from .pipeline import RTResult, process_snapshot, run  # noqa: E402,F401
from .params import (  # noqa: E402,F401
    ParameterError,
    RunConfig,
    load_parameter_file,
    parse_parameters,
    run_parameter_file,
)
