"""Runner for PoreDiameter block — solves for pore diameter D.

From:  dR/R = (d^3 / (D^2 * L)) * 1 / (1 - 0.8 * (d/D)^3)

Rearranged to cubic in D:
    dR/R * L * D^3  -  d^3 * D  -  0.8 * dR/R * L * d^3  =  0

Solved numerically per element using Brent's method.
"""

import numpy as np
from scipy.optimize import brentq


def _solve_D(dR_R_val, d_val, L_val):
    """Solve for D given scalar dR/R, d, L."""
    if dR_R_val <= 0 or d_val <= 0 or L_val <= 0:
        return np.nan
    a = dR_R_val * L_val

    def f(D):
        return a * D**3 - d_val**3 * D - 0.8 * a * d_val**3

    # D must be > d (particle fits inside pore)
    lo = d_val * 1.001
    hi = d_val * 100
    try:
        # Ensure bracket: f(lo) should be negative, f(hi) positive
        if f(lo) * f(hi) > 0:
            # Expand search range
            hi = d_val * 1000
            if f(lo) * f(hi) > 0:
                return np.nan
        return brentq(f, lo, hi, xtol=1e-12)
    except (ValueError, RuntimeError):
        return np.nan


def run(inputs, params, block):
    dR_R = np.asarray(inputs.get("dROverR"), dtype=float)
    d = np.asarray(inputs.get("d"), dtype=float)
    L = np.asarray(inputs.get("L"), dtype=float)

    # Broadcast scalars
    d_scalar = d.item(0) if d.size == 1 else None
    L_scalar = L.item(0) if L.size == 1 else None

    dR_R = np.atleast_1d(dR_R).ravel()
    result = np.empty_like(dR_R)

    for i in range(len(dR_R)):
        di = d_scalar if d_scalar is not None else d.ravel()[i]
        Li = L_scalar if L_scalar is not None else L.ravel()[i]
        result[i] = _solve_D(dR_R[i], di, Li)

    return {"D": result}
