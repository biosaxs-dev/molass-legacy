"""
    SecTheory.NumPlates.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Elution.CurveUtils import simple_plot
from .RetensionTime import compute_retention_time
from .MonoPore import compute_mu_sigma

NUM_PLATES = 48000*0.3

def contrast_sigma_info(in_folder, sd, job, nc, fv, seccol_params, sigmas, monopore_params, rg_params, debug=True):
    xr_curve = sd.get_xray_curve()

    emg_peaks = xr_curve.get_emg_peaks()
    p = xr_curve.primary_peak_no
    ppeak = emg_peaks[p]
    x = xr_curve.x
    params = ppeak.get_params()
    print("params=", params)
    mean = params[1]
    stdev = params[2]

    # get the nearest Rg
    trs = compute_retention_time(seccol_params, rg_params)
    n = np.argmin((trs - mean)**2)
    rg = rg_params[n]
    if sigmas is None:
        # stc
        mu, sigma = compute_mu_sigma(*monopore_params, rg)
    else:
        # egh
        mu = mean
        sigma = sigmas[n]

    tr = compute_retention_time_from_N(NUM_PLATES, sigma)
    print("tr=", tr)

    t0, K, rp, m = seccol_params

    if debug:
        from DataUtils import get_in_folder
        in_folder = get_in_folder(in_folder)
        with plt.Dp():
            fig, ax = plt.subplots(figsize=(18,4))
            ax.set_title("SEC Parameter Plot at Job %s processing Data %s" % (job, in_folder), fontsize=16)
            simple_plot(ax, xr_curve, legend=False)
            y = ppeak.get_model_y(x)
            ax.plot(x, y, label="egh")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            ti = params[1] - tr
            for j, color, label in (ti,"green", "injection"), (t0, "yellow", "t0"), (t0+K, "yellow", "t0+K"):
                ax.plot([j, j], [ymin, ymax], color=color, label=label)

            xmin, xmax = ax.get_xlim()
            dx = (xmax - xmin)*0.02
            dy = (ymax - ymin)*0.05

            px = mean
            py = xr_curve.spline(px)
            ax.text(px-dx, py-dy, "Rg = %.3g" % rg, ha="right", va="center", fontsize=20, alpha=0.5)

            tx = xmin*0.5 + xmax*0.5
            ty = ymin*0.2 + ymax*0.8
            ax.text(tx, ty, "Pore Size = %.3g" % rp, ha="center", va="center", fontsize=20, alpha=0.5)
            ax.legend(loc="lower center")
            fig.tight_layout()
            plt.show()

    return mean, stdev, mu, sigma, tr, rg

def compute_retention_time_from_N(num_plates, sigma):
    """
    N = (tr/stdev)**2
    tr = sqrt(N)*sigma
    """
    return np.sqrt(num_plates) * sigma
