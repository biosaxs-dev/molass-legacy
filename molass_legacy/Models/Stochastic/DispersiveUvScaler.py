"""
    Models.Stochastic.DispersiveUvScaler.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf 
from molass_legacy.Models.Stochastic.DispersiveUtils import NUM_SDMCUV_PARAMS

def adjust_to_uv_scales(x, y, uv_x, uv_y_, sdm_params, rgs, avoid_vanishing=True, debug=False, optimizer=None):
    print("adjust_to_uv_scales: xr_init_params=", sdm_params, "rgs=", rgs)
    N, K, x0, poresize, N0, tI = sdm_params[0:NUM_SDMCUV_PARAMS]
    T = K/N
    me = 1.5
    mp = 1.5
    xr_scales = sdm_params[NUM_SDMCUV_PARAMS:]
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    cy_list = []
    for rho in rhov:
        np_ = N*(1 - rho)**me
        tp_ = T*(1 - rho)**mp
        cy = dispersive_monopore_pdf(x, np_, tp_, N0, x0)   # note that the scale is not applied here
        cy_list.append(cy)

    def uv_scales_objective(scales, return_cy=False):
        uv_cy_list = []
        for cy, scale in zip(cy_list, scales):
            uv_cy = scale * cy
            uv_cy_list.append(uv_cy)
        uv_ty = np.sum(uv_cy_list, axis=0)
        if return_cy:
            return uv_cy_list, uv_ty
        dev = np.sum((uv_ty - uv_y_)**2)
        
        # Add strong penalty for near-zero scales to prevent degeneracy under high overlap
        # Use logarithmic penalty in scale space: -(log(s) + 2)^2 approximation
        # This heavily penalizes scales < ~0.1 while leaving healthy scales (>0.3) mostly unaffected
        safe_scales = np.maximum(scales, 1e-8)
        log_penalty = np.sum(-np.log(safe_scales + 0.1)**2) * 10  # log penalty with weight
        
        return dev + log_penalty

    xr2uv_scale = np.max(uv_y_)/np.max(y)   # task: get earlier and keep this scale

    # Try objective-based UV scale refinement if optimizer is available
    optimized_scales = None
    if optimizer is not None and hasattr(optimizer, '_refine_uv_scales') and optimizer._refine_uv_scales:
        print(f"[DEBUG] Using objective-based UV scale refinement in adjust_to_uv_scales")
        try:
            uv_w = optimize_uv_scales_via_objective(optimizer, sdm_params, optimizer._uv_scale_indices, xr_scales, xr_scales, debug=debug)
            if uv_w is not None:
                print(f"[DEBUG] Objective-based refinement succeeded: {uv_w}")
                optimized_scales = uv_w
                uv_cy_list, uv_ty = uv_scales_objective(uv_w, return_cy=True)
            else:
                print(f"[DEBUG] Objective-based refinement returned None, falling back to 2D fit")
        except Exception as e:
            print(f"[DEBUG] Objective-based refinement failed: {e}, falling back to 2D fit")

    # If objective-based refinement didn't work, use 2D fit
    if optimized_scales is None:
        if avoid_vanishing:
            min_scale = np.min(xr_scales)*xr2uv_scale
        else:
            min_scale = 0
        bounds = [(min_scale, 100)] * len(xr_scales)
        print(f"[DEBUG] 2D fit: avoid_vanishing={avoid_vanishing}, min_scale={min_scale:.6g}, bounds={bounds}")
        res = minimize(uv_scales_objective, xr_scales, method="Nelder-Mead", bounds=bounds)
        print(f"[DEBUG] Nelder-Mead result: x={res.x}, fun={res.fun}, success={res.success}")
        optimized_scales = res.x
        uv_cy_list, uv_ty = uv_scales_objective(res.x, return_cy=True)

    if debug:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
            fig.suptitle("adjust_to_uv_scales")
            ax1.set_title("UV")

            ax1.plot(uv_x, uv_y_, color="blue")
            for k, cy in enumerate(uv_cy_list):
                ax1.plot(uv_x, cy, ":", label='component-%d' % k)
            ax1.plot(uv_x, uv_ty, ":", color="red", label='model total')
            ax1.legend()

            ax2.set_title("XR")
            ax2.plot(x, y, color="orange")
            xr_cy_list = []
            for k, (cy, scale) in enumerate(zip(cy_list, xr_scales)):
                xr_cy = scale * cy
                ax2.plot(x, xr_cy, ":", label='component-%d' % k)
                xr_cy_list.append(xr_cy)
            xr_ty = np.sum(xr_cy_list, axis=0)
            ax2.plot(x, xr_ty, ":", color="red", label='model total')
            ax2.legend()

            fig.tight_layout()
            ret = plt.show()
    else:
        ret = True

    if ret:
        return optimized_scales, uv_ty
    else:
        return


def optimize_uv_scales_via_objective(optimizer, init_params, uv_scale_indices, 
                                     uv_w_initial, xr_scales, debug=False):
    """
    Refine UV scales by minimizing the full objective function, keeping all other params fixed.
    
    This uses the real rigorous objective (with penalties) instead of simple NNLS on the UV matrix,
    preventing zero-scales when components are highly overlapping (similar Rgs → near-identical SDM peaks).
    
    Uses bounds derived from median(xr_scales), not min, so the bound doesn't degrade when one
    component already has a very small XR scale.
    
    Includes light L2 regularization (in log space) toward the initial peak-based estimate,
    keeping the solution physically reasonable.
    
    Parameters
    ----------
    optimizer : BasicOptimizer
        The optimizer object with objective_func method.
    init_params : 1-D array
        Full initial parameter vector.
    uv_scale_indices : 1-D array of int
        Indices of uv_w in init_params.
    uv_w_initial : 1-D array
        Initial UV scales (from peak-based estimate or prior NNLS).
    xr_scales : 1-D array
        XR component scales (used to set bounds).
    debug : bool
        Print debug info.
    
    Returns
    -------
    uv_w_optimized : 1-D array or None
        Optimized UV scales, or None if optimization failed.
    """
    # Bounds: use median(xr_scales) instead of min to avoid degradation under overlap
    median_xr = np.median(xr_scales)
    # Estimate XR→UV scale ratio from initial scales
    init_min_xr = np.min(xr_scales[xr_scales > 1e-10])
    init_min_uv = np.min(uv_w_initial[uv_w_initial > 1e-10])
    xr2uv_ratio = init_min_uv / init_min_xr if init_min_xr > 1e-10 else 1.0
    
    lower_bound = max(1e-6, median_xr * xr2uv_ratio * 0.05)
    upper_bound = median_xr * xr2uv_ratio * 100
    bounds = [(lower_bound, upper_bound)] * len(uv_w_initial)
    
    if debug:
        print(f"  optimize_uv_scales_via_objective:")
        print(f"    median(xr_scales)={median_xr:.6g}, xr2uv_ratio={xr2uv_ratio:.6g}")
        print(f"    bounds=[{lower_bound:.6g}, {upper_bound:.6g}]")
        print(f"    uv_w_initial={uv_w_initial}")
    
    def objective_wrt_uv(uv_w_candidate):
        """Evaluate full objective with only UV scales varying."""
        p_candidate = init_params.copy()
        p_candidate[uv_scale_indices] = uv_w_candidate
        fv = optimizer.objective_func(p_candidate)
        
        # Light L2 regularization in log space toward initial estimate
        # Prevents solution from drifting too far from physical guess
        safe_initial = np.clip(uv_w_initial, 1e-10, 1e10)
        safe_candidate = np.clip(uv_w_candidate, 1e-10, 1e10)
        reg = 0.5 * np.sum((np.log(safe_candidate) - np.log(safe_initial))**2)
        
        return fv + reg
    
    try:
        res = minimize(objective_wrt_uv, uv_w_initial, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 150})
        
        # Check for success: optimization converged AND all scales are healthy
        if res.success and np.all(res.x > lower_bound * 0.1):
            if debug:
                print(f"    ✓ optimization converged: fv={res.fun:.6g}, x={res.x}")
            return res.x
        else:
            if debug:
                reason = "convergence failure" if not res.success else "near-zero scales detected"
                print(f"    ✗ optimization failed ({reason}): fv={res.fun:.6g}")
            return None
    except Exception as e:
        if debug:
            print(f"    ✗ exception during optimization: {e}")
        return None