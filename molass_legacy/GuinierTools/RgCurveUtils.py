"""
    GuinierTools.RgCurveUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Optimizer.NumericalUtils import safe_ratios
# from molass_legacy.Trimming.Sigmoid import sigmoid

VALID_QUIALTY_LIMIT = 0.01
VALID_BASE_QUALITY = 0.3

def convert_to_milder_qualities(qualities):
    # task: this is a temporary fix. deeper improvement is required. 
    ret_qualities = qualities.copy()
    valid = qualities > VALID_QUIALTY_LIMIT
    ret_qualities[valid] = VALID_BASE_QUALITY + (1 - VALID_BASE_QUALITY)*qualities[valid]
    return ret_qualities

def get_connected_curve_info(rg_curve, debug=False):
    segments = rg_curve.get_curve_segments()
    qualities = np.concatenate(rg_curve.qualities)      # basic qualiteis; see RgProcess.RgCurve.py

    x_list = []
    y_list = []
    rg_list = []
    valid_bool_all = np.zeros(len(rg_curve.x), dtype=bool)
    valid_bool_seg_list = []
    k = 0
    for state, slice_ in zip(rg_curve.states, rg_curve.slices):
        if state == 0:           
            continue

        x_, y_, rg_ = segments[k]
        x_list.append(x_)
        y_list.append(y_)
        rg_list.append(rg_)
        bvec = rg_curve.qualities[k] > VALID_QUIALTY_LIMIT
        valid_bool_all[slice_] = bvec
        valid_bool_seg_list.append(bvec)
        k += 1

    x_ = np.concatenate(x_list)
    y_ = np.concatenate(y_list)
    rgv = np.concatenate(rg_list)
    valid_bool_seg = np.concatenate(valid_bool_seg_list)

    if debug:
        print(len(x_), len(y_), len(rgv), len(qualities))
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("get_connected_curve_info: debug")
            axt = ax.twinx()
            axt.grid(False)
            for dx, dy, drg in zip(x_list, y_list, rg_list):
                ax.plot(dx, dy)
                axt.plot(dx, drg, color='gray')
            # axt.plot(x_[valid_bools], rgv[valid_bools], 'o', color="green", alpha=0.3, label="valid $R_g's$")
            axt.plot(x_, qualities*100, color='green', alpha=0.5)
            fig.tight_layout()
            plt.show()

    return x_, y_, rgv, qualities, (valid_bool_all, valid_bool_seg)

def get_reconstructed_curve(size, valid_bools, Cxr, rg_params):
    valid_bool_all, valid_bool_seg = valid_bools
    ty_ = np.sum(Cxr, axis=0)[valid_bool_all]
    rrgv = np.zeros(size)
    for cy, rg in zip(Cxr[:-1], rg_params):
        rrgv += cy[valid_bool_all]/ty_*rg
    return rrgv

def compute_rg_curves(x, xr_weights, rg_params, xr_cy_list, xr_ty, rg_curve, debug=False):
    # recompute without mask for plot

    t_rg_ = np.zeros(len(x))
    ones = np.ones(len(t_rg_))
    for w, r, xr_cy in zip(xr_weights, rg_params, xr_cy_list):  # weighted sum without baseline part
        if w > 0.001:
            t_rg_ += r * safe_ratios(ones, xr_cy, xr_ty, debug=False)   # using safe_ratios to avoid inconveniences by outliers

    rg_curves1 = []
    rg_curves2 = []
    segments = rg_curve.segments
    k = 0
    for state, slice_ in zip(rg_curve.states, rg_curve.slices):
        if state == 0:
            continue
        x_, y_, rg_ = segments[k]
        rg_curves1.append((x_, rg_))
        rg_curves2.append((x[slice_], t_rg_[slice_]))
        k += 1

    if debug:
        with plt.Dp():
            fig, ax = plt.subplots()
            fig.suptitle("compute_rg_curves debug")
            axt = ax.twinx()
            axt.grid(False)

            for cy in xr_cy_list:
                ax.plot(x, cy, ":")
            ax.plot(x, xr_ty, ":", color="red")

            for (x1_, rg1_), (x2_, rg2_) in zip(rg_curves1, rg_curves2):
                axt.plot(x1_, rg1_)
                axt.plot(x2_, rg2_, ":")
            fig.tight_layout()
            plt.show()

    return rg_curves1, rg_curves2

def plot_rg_curves(ax, xrh_params, rg_params, x, xr_cy_list, xr_ty, rg_curve):
    if len(xr_cy_list) > len(rg_params):
        xr_ty_ = xr_ty - xr_cy_list[-1]      # remove baseline component
    else:
        # assuming baseline component has been removed
        xr_ty_ = xr_ty
    weights = xrh_params/np.max(xrh_params)
    num_components = len(xrh_params)
    assert len(rg_params) == num_components

    rg_curves1, rg_curves2 = compute_rg_curves(x, weights, rg_params, xr_cy_list[:num_components], xr_ty_, rg_curve)

    k = 0
    for (x1, rg1), (x2, rg2) in zip(rg_curves1, rg_curves2):
        label = 'observed rg' if k == 0 else None
        ax.plot(x1, rg1, color='gray', alpha=0.5, label=label)
        label = 'reconstructed rg' if k == 0 else None
        ax.plot(x2, rg2, ':', color='black', label=label)
        k += 1

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(min(10, ymin), max(50, ymax))
    ax.legend(loc='upper left')

def rg_deviation_inspect_impl(gdev, valid_components, rg_params, lrf_rgs):
    print("rg_deviation_inspect_impl")
    print("valid_components=", valid_components)
    print("rg_params=", rg_params)
    print("lrf_rgs=", lrf_rgs)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        ax1.set_title("Rg Plot", fontsize=16)
        ax2.set_title("Rg Curve", fontsize=16)
        ax1.plot(rg_params, "o", label='Rg params')
        ax1.plot(lrf_rgs, "o", label='LRF Rgs')
        ax1.legend()
        fig.tight_layout()
        plt.show()