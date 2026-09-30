"""
    NumericalUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""

import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

OUTLIER_SCALE = 2
OUTLIER_IGNORE_LIMIT = 1e-6
UV_DOMAIN_MASK_MIN_VALID = 10

def compute_uv_domain_mask(uv_x, uv_domain_lo, uv_domain_hi, min_valid=UV_DOMAIN_MASK_MIN_VALID):
    """Boolean mask selecting frames within the really-measured UV domain.

    ``uv_x`` maps every XR frame into UV coordinates (``uv_x = a*x+b`` in each
    objective function), but UV is only actually measured within
    ``[uv_domain_lo, uv_domain_hi]``; frames outside that range are
    spline/interpolation extrapolation, not real data, and inflate
    UV_2D_fitting/UV_LRF_residual if not excluded (see BasicOptimizer#286,
    molass-library#270).

    Falls back to an all-True mask if fewer than ``min_valid`` frames would
    remain, to avoid degenerate (near-empty) norms during optimization when
    the mapping is briefly far off (e.g. early BH/NS iterations).

    Parameters
    ----------
    uv_x : array-like
        Each XR frame's position mapped into UV coordinates.
    uv_domain_lo, uv_domain_hi : float
        The real, measured UV coordinate range (inclusive).
    min_valid : int, optional
        Minimum number of True entries required to use the mask as-is.

    Returns
    -------
    ndarray of bool
        Same shape as ``uv_x``.
    """
    uv_x = np.asarray(uv_x)
    mask = (uv_x >= uv_domain_lo) & (uv_x <= uv_domain_hi)
    if mask.sum() < min_valid:
        return np.ones_like(mask, dtype=bool)
    return mask

def safe_ratios_debug_plot(x, y, xr_ty, xr_cy_list, rg_curve, rg_params):
    return

    with plt.Dp():
        fig, ax = plt.subplots()
        axt = ax.twinx()
        axt.grid(False)

        ax.plot(x, y, color="orange")
        for cy in xr_cy_list[:-1]:
            ax.plot(x, cy, ":")

        ax.plot(x, ty, ":", color="red")

        # axt.plot(ratios, "-", color="C1")
        fig.tight_layout()
        plt.show()

def safe_ratios(ones, cy, ty, debug=False):
    # ty can be zero/near-zero at the edges of the elution range; the resulting
    # nan/inf is intentionally overwritten below, so the divide-by-zero and
    # invalid-value RuntimeWarnings are expected noise, not a real problem.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = cy/ty
    ratios[ty==0] = 1

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("safe_ratios entry")
            axt = ax.twinx()
            axt.grid(False)

            ax.plot(ty, ":")
            ax.plot(cy, ":")
            axt.plot(ratios, "-", color="C1")
            fig.tight_layout()
            plt.show()

    outliers = np.where(np.abs(ratios) > ones*OUTLIER_SCALE)[0]
    if len(outliers) == 0:
        return ratios

    # outliers appear only where both cy and ty are very small
    outlier_height = np.max(np.abs(ty[outliers]))
    if outlier_height < OUTLIER_IGNORE_LIMIT:
        widened = ty < OUTLIER_IGNORE_LIMIT
    else:
        widened = None

    if debug:
        print("outlier_height=", outlier_height)
        ratios_orig = ratios.copy()
        if widened is None:
            ratios[outliers] = 1
        else:
            ratios[widened] = 1

        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("safe_ratios outliers")

            axt = ax.twinx()
            axt.grid(False)

            ax.plot(cy)
            ax.plot(ty)

            axt.plot(ratios_orig)
            axt.plot(ratios)
            plt.show()
    else:
        if widened is None:
            ratios[outliers] = 1
        else:
            ratios[widened] = 1

    return ratios
