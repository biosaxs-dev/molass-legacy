"""
    SimpleGuinierDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
from bisect import bisect_right
import numpy as np
import time
import molass_legacy.KekLib.DebugPlot as plt
from DataUtils import get_in_folder
from SvdDenoise import get_denoised_data
from SimpleGuinier import SimpleGuinier

def compute_lrf_info(D, E, qv, i_smp, aslice, eslice, rank=2):
    D_ = D[:,eslice]
    E_ = E[:,eslice]

    D_ = get_denoised_data(D_, rank=rank)
    c = D_[i_smp, :]
    if rank == 1:
        C = np.array([c])
    else:
        C = np.array([c, c**2])
    Cinv = np.linalg.pinv(C)
    P = D_ @ Cinv
    Dinv = np.linalg.pinv(D_)
    W = np.dot(Dinv, P)
    Pe = np.sqrt(np.dot(E_**2, W**2))
    data = np.array([qv, P[:,0], Pe[:,0]]).T
    sg = SimpleGuinier(data)
    z_ = np.log(data[aslice,1])
    return sg, z_

g_counter = 0

def demo(in_folder, sd, fig_folder=None, counter=None, for_all_peaks=False, index_csv_fh=None):

    print(in_folder)

    if counter is None:
        global g_counter
        counter = g_counter
        g_counter += 1

    if index_csv_fh is not None:
        index_csv_fh.write(','.join([str(counter), in_folder]) + "\n")
        index_csv_fh.flush()

    D, E, qv, ecurve = sd.get_xr_data_separate_ly()

    for k, pinfo in enumerate(ecurve.peak_info):
        if for_all_peaks:
            pass
        else:
            if k != ecurve.primary_peak_no:
                continue

        eno = pinfo[1]
        eslice = slice(pinfo[0], pinfo[2]+1)

        y = D[:,eno]
        e = E[:,eno]
        data = np.array([qv, y, e]).T
        sg = SimpleGuinier(data)
        i_smp = sd.xray_index

        i = bisect_right(qv, 2/max(10, sg.Rg))

        aslice = slice(0, i)
        qv2 = qv[aslice]**2
        y_ = np.log(y[aslice])

        lrf1_sg, z1_ = compute_lrf_info(D, E, qv, i_smp, aslice, eslice, rank=1)

        lrf2_sg, z2_ = compute_lrf_info(D, E, qv, i_smp, aslice, eslice, rank=2)

        peak_suffix = "-%d" % k if for_all_peaks else ""

        with plt.Dp():
            in_folder_ = get_in_folder(in_folder)
            fig, ax = plt.subplots()

            ax.set_title("Guinier Plots of Curves in %s%s" % (in_folder_, peak_suffix))

            Iz = np.log(sg.Iz)
            ax.plot(qv2, y_ - Iz, "o", markersize=1, label="main-peak curve")
            ax.plot(sg.guinier_x, sg.guinier_y - Iz, color="red", alpha=0.5, lw=2, label="main-peak guinier region")

            for lrf_sg, z_, rank in [(lrf1_sg, z1_, 1), (lrf2_sg, z2_, 2)]:
                Iz = np.log(lrf_sg.Iz)
                ax.plot(qv2, z_ - Iz, "o", markersize=1, label="rank%d-lrf curve" % rank)
                color = "pink" if rank == 1 else "cyan"
                ax.plot(lrf_sg.guinier_x, lrf_sg.guinier_y - Iz, color=color, alpha=0.5, lw=2, label="rank%d-lrf guinier region" % rank)

            ax.legend()

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(min(-1.2, ymin), ymax)
            tx = xmin*0.9 + xmax*0.1
            w = 0.3; ty = ymin*(1-w) + ymax*w
            ax.text(tx, ty, "Peak Rg=%.1f min_qRg=%.3g" % (sg.Rg, sg.min_qRg), fontsize=16, alpha=0.5, va="center")
            w = 0.2; ty = ymin*(1-w) + ymax*w
            ax.text(tx, ty, "LRF1 Rg=%.1f min_qRg=%.3g" % (lrf1_sg.Rg, lrf1_sg.min_qRg), fontsize=16, alpha=0.5, va="center")
            w = 0.1; ty = ymin*(1-w) + ymax*w
            ax.text(tx, ty, "LRF2 Rg=%.1f min_qRg=%.3g" % (lrf2_sg.Rg, lrf2_sg.min_qRg), fontsize=16, alpha=0.5, va="center")

            fig.tight_layout()

            if fig_folder is None:
                plt.show()
            else:
                plt.show(block=False)
                dp = plt.get_dp()
                fig_file = os.path.join(fig_folder, "fig-%03d%s.png" % (counter, peak_suffix))
                dp.fig.savefig(fig_file)
                time.sleep(1)
