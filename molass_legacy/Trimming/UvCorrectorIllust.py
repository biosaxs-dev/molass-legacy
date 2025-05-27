"""
    UvCorrectorIllust.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.EghSupples import egh, d_egh

def illust():
    with plt.Dp():
        x = np.arange(600)
        sigma = 20
        params = np.array([1, 300, sigma, 30])
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle("UV Distortion Model", fontsize=20)
        ax1.set_title("egh", fontsize=16)
        ax2.set_title("reflected derivative of egh", fontsize=16)

        for tau in [20, 30, 40]:
            params[2] = tau
            ax1.plot(x, egh(x, *params), label="sigma=%g, tau=%g" % (sigma, tau))
            params_ = params.copy()
            params_[0] = -1
            ax2.plot(x, d_egh(x, *params_), label="sigma=%g, tau=%g" % (sigma, tau))
        for ax in [ax1, ax2]:
            ax.legend()
        fig.tight_layout()
        plt.show()

def debug_plot(title_words, uv_corrector, x, y, params, fit_slice, init_params=None, fig_file=None):
    from DataUtils import get_in_folder
    from molass_legacy.Elution.CurveUtils import simple_plot
    print("params=", params)
    in_folder = get_in_folder()
    title = "UV Correction with %s for %s" % (title_words, in_folder)

    slope, intercept = params[0:2]
    dy_ = uv_corrector.correction_curve(x, *params)
    y_ = uv_corrector.get_corrected_y(params=params)
    if False:
        egh_params = params[2:].copy()
        if egh_params[0] < 0:
           egh_params[0] *= -1

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle(title, fontsize=20)
        ax1.set_title("Elution at λ=%g" % 280, fontsize=16)
        ax2.set_title("Elution at λ=%g" % 340, fontsize=16)

        simple_plot(ax1, uv_corrector.curve1, legend=False)
        # ax1.plot(x, egh(x, *egh_params), label="egh")
        ax1.legend()

        if init_params is not None:
            scale = params[2]/init_params[2]
            tx = np.average(ax1.get_xlim())
            ty = np.average(ax1.get_ylim())
            ax1.text(tx, ty, "Scale: %.3g" % scale, ha="center", fontsize=30, alpha=0.3)

        ax2.plot(x, y, alpha=0.5, label="data")
        ax2.plot(x, dy_, color="cyan", lw=2, label="distotion model")
        ax2.plot(x, y_, color="green", alpha=0.5, label="corrected data")
        ax2.plot(x, x*slope + intercept, ":", color="red", lw=2, label="linear regression")

        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        for i, p in enumerate([fit_slice.start, fit_slice.stop]):
            label = "fit range ends" if i == 0 else None
            ax2.plot([p, p], [ymin, ymax], ":", color="gray", label=label)

        ax2.legend()
        fig.tight_layout()
        if fig_file is None:
            plt.show()
        else:
            from time import sleep
            plt.show(block=False)
            fig = plt.gcf()
            fig.savefig(fig_file)
            sleep(1)
