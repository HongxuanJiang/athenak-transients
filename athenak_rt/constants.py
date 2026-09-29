"""Physical constants (CGS) and luminosity conventions shared by the package.

The numerical values are those of the validated post-processing scripts used for
the paper; they are deliberately not taken from astropy/scipy so that results
stay bit-identical with the archived runs.
"""

from __future__ import annotations

import math

K_B_CGS = 1.380649e-16
H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
SIGMA_SB_CGS = 5.670374419e-5
HYDROGEN_MASS_CGS = 1.6735575e-24
EV_CGS = 1.602176634e-12
SAFE_POSITIVE = 1.0e-300

# H/He continuum physics (rt_continuum_experiments.py conventions).
ELECTRON_MASS_CGS = 9.1093837e-28
PROTON_MASS_CGS = 1.67262192e-24
SIGMA_THOMSON_CGS = 6.6524587e-25
X_H = 0.7
Y_HE = 0.3
CHI_H = 13.598 * EV_CGS
CHI_HE1 = 24.587 * EV_CGS
CHI_HE2 = 54.418 * EV_CGS
NU_H = CHI_H / H_CGS
NU_HE1 = CHI_HE1 / H_CGS
NU_HE2 = CHI_HE2 / H_CGS

# Projected/isotropic-equivalent normalization from Yang et al. (2026),
# equation (10).  A blackbody patch contributes pi B_nu dA as its one-sided
# hemispheric spectral power, while the line-of-sight isotropic-equivalent
# luminosity is 4 pi B_nu dA.  Hence both spectral and bolometric pixel powers
# acquire the same factor of four relative to the hemispheric convention.
PROJECTED_LUMINOSITY_FACTOR = 4.0
LUMINOSITY_NORMALIZATION = "projected_isotropic_equivalent"
LUMINOSITY_REFERENCE = "Yang et al. (2026), equation (10)"
LUMINOSITY_BOLOMETRIC_FORMULA = (
    "L_bol = 4 integral sigma_SB T_ph^4 dA  (= 4 pi integral I dA)"
)
LUMINOSITY_SPECTRAL_FORMULA = (
    "L_nu = 4 pi integral I_nu dA  (blackbody: I_nu = B_nu(T_ph))"
)

SIGMA_OVER_PI = SIGMA_SB_CGS / math.pi
