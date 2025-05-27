"""
    CharacteristicDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import time
from DataUtils import get_in_folder

def demo(in_folder, sd):
    import molass_legacy.KekLib.DebugPlot as plt
    from importlib import reload
    import Models.Characteristic
    reload(Models.Characteristic)    
    from molass_legacy.Models.Characteristic import CfSpace
    from molass_legacy.Models.CfsEvalPeaks import get_cfs_evaluated_peaks
    from molass_legacy.Models.ElutionCurveModels import egha

    print(in_folder)
    ecurve = sd.get_xray_curve()
    x = ecurve.x
    y = ecurve.y

    cfs = CfSpace(use_np_fft=True)
    t0 = time.perf_counter()
    w = cfs.get_w()
    cft = cfs.compute_cf(x, y)
    t1 = time.perf_counter()

    print("it took:", t1 - t0)
    print("len(cft)=", len(cft))

    num_peaks = len(ecurve.peak_info)
    egha_params, emga_params = get_cfs_evaluated_peaks(x, y, num_peaks, debug=False)

    cy_list = []
    for p in egha_params:
        cy = egha(x, *p)
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)

    ty_cft = cfs.compute_cf(x, ty)

    in_folder_ = get_in_folder(in_folder)
    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
        fig.suptitle("Decomposition Demo using CF Space with %s" % in_folder_, fontsize=20)
        ax1, ax2, ax3, ax4 = axes[0,:]

        ax1.set_title("Data", fontsize=16)
        ax2.set_title("cft.real vs w", fontsize=16)
        ax3.set_title("cft.imag vs w", fontsize=16)
        ax4.set_title("cft.imag vs cft.real", fontsize=16)

        ax1.plot(x, y)

        ax2.plot(w, cft.real)
        ax3.plot(w, cft.imag)
        ax4.plot(cft.real, cft.imag, "o", markersize=3)

        ax5, ax6, ax7, ax8 = axes[1,:]

        ax5.plot(x, y)
        for cy in cy_list:
            ax5.plot(x, cy, ":")
        ax5.plot(x, ty, ":", lw=3, color="red")

        ax6.plot(w, ty_cft.real, color="red")
        ax7.plot(w, ty_cft.imag, color="red")
        ax8.plot(ty_cft.real, ty_cft.imag, "o", markersize=3, color="red")

        fig.tight_layout()
        plt.show()
