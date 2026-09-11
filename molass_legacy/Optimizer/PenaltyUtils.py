"""
    Optimizer.PenaltyUtils.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
PENALTY_SCALE = 1e3
UV_B_ALLOW_RATIO = 0.1

def compute_mapping_penalty(uv_curve, xr_curve, init_mapping, mapping, uv_x_size, xr_scales, uv_scales, debug=False):
    if debug:
        from importlib import reload
        import molass_legacy.Optimizer.XrUvScaleRatio
        reload(molass_legacy.Optimizer.XrUvScaleRatio)
    from molass_legacy.Optimizer.XrUvScaleRatio import xruv_scale_ratio_penalty
    a_init, b_init = init_mapping
    a, b = mapping
    ratio = a/a_init
    a_deviation = min(0, ratio - 0.99)**2 + max(0, ratio - 1.05)**2
    b_allowance = uv_x_size * UV_B_ALLOW_RATIO
    b_deviation = max(0, abs(b - b_init) - b_allowance)**2

    ratio_penalty = xruv_scale_ratio_penalty(xr_scales, uv_scales, debug=debug)
    mapping_penalty = PENALTY_SCALE * (a_deviation + b_deviation ) + ratio_penalty

    if debug:
        print("ratio=", ratio)
        print("a_deviation=", a_deviation)
        print("b_allowance=", b_allowance)
        print("ratio_penalty=", ratio_penalty)
        print("mapping_penalty=", mapping_penalty)
        xr_x = xr_curve.x
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("compute_mapping_penalty: debug")
            ax.set_xlabel("XR")
            ax.set_ylabel("UV")
            ax.plot(xr_x, xr_x*a_init + b_init, label='initial mapping')
            ax.plot(xr_x, xr_x*a + b, ':', label='current mapping')
            fig.tight_layout()
            plt.show()

    return mapping_penalty


# LKM mass-transfer floor (molass-library Copilot/refactor/DESIGN_lkm_mass_transfer_floor.md).
# Derived from LkmLinear.lkm_pdf's characteristic function H(s) by Taylor-expanding
# ln H(s) to O(s^2) (moment-generating-function expansion):
#   sigma^2 = 2*t0^2*R^2/Pe        <- axial dispersion, independent of k_MT
#           + 2*t0*(R-1)/k_MT      <- mass-transfer resistance (blows up as k_MT -> 0)
# Requiring sigma <= LKM_MT_TOLERANCE times the axial-only baseline gives, per component:
#   k_MT_floor = (R-1)*Pe / ((tolerance^2-1)*t0*R^2)
# Verified against real SAMPLE5 BH/DE fits to <1% (molass-papers/experiments/
# recipe_runner_notebook_redesign.ipynb, Sections 11-12): predicted vs measured
# component width agreed closely, and this floor correctly separates the two
# known-pathological components (near-zero k_MT, unrealistically broad) from the
# four well-behaved ones.
LKM_MT_TOLERANCE = 1.5   # m: max acceptable width inflation vs the axial-only baseline
LKM_MT_PENALTY_SCALE = 3.0


def compute_lkm_mass_transfer_penalty(Pe, t0, R_values, k_MT_values,
                                       tolerance=LKM_MT_TOLERANCE,
                                       scale=LKM_MT_PENALTY_SCALE, debug=False):
    """Soft-ramp penalty discouraging k_MT collapsing toward zero (LKM broadening pathology).

    Dimensionless per component: 0 when k_MT >= floor, ramping up to `scale`
    as k_MT -> 0. Using a ratio (not the raw k_MT - floor difference) keeps the
    penalty comparable in magnitude across datasets with very different Pe/t0
    scales.

    Parameters
    ----------
    Pe, t0 : float
        Shared LKM column parameters for this trial.
    R_values, k_MT_values : array-like
        Per-component retention factor and mass-transfer rate (same order).
    tolerance : float, optional
        m in the derivation above. Default 1.5 (empirically checked against
        real data -- see module docstring).
    scale : float, optional
        Penalty weight per component at full collapse (k_MT -> 0).

    Returns
    -------
    float
        Total penalty (>= 0), to be added to the objective's penalty list.
    """
    import numpy as np
    R = np.asarray(R_values, dtype=float)
    k_MT = np.asarray(k_MT_values, dtype=float)
    floor = (R - 1.0) * Pe / ((tolerance**2 - 1.0) * t0 * R**2)
    shortfall = np.maximum(0.0, 1.0 - k_MT / floor)
    penalty = scale * float(np.sum(shortfall**2))
    if debug:
        print("k_MT_floor=", floor)
        print("k_MT_values=", k_MT)
        print("shortfall=", shortfall)
        print("mass_transfer_penalty=", penalty)
    return penalty