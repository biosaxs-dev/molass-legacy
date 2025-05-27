"""
    ReconsiderDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
from bisect import bisect_right
from scipy.stats import linregress
from matplotlib.patches import Rectangle
import molass_legacy.KekLib.DebugPlot as plt
from Theory.SolidSphere import phi

def demo_impl(sd, f, t):
    from DataUtils import get_in_folder
    from importlib import reload
    import BoundedLRF.NaiiveLrfSolver
    reload(BoundedLRF.NaiiveLrfSolver)
    from .NaiiveLrfSolver import NaiiveLrfSolver
    import BoundedLRF.BoundedLrfSolver
    reload(BoundedLRF.BoundedLrfSolver)
    from .BoundedLrfSolver import BoundedLrfSolver

    print("demo_impl")

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y
    data_title = get_in_folder()

    i = bisect_right(qv, 0.02)
    cy = D[i,:]
    j = np.argmax(cy)

    print("len(cy)=", len(cy), "j=", j)
    pv1 = D[:,j]
    # pv1 = pv1/np.max(pv1)

    range_ = slice(f, t+1)

    solver = NaiiveLrfSolver(qv, D[:,range_], E[:,range_])
    P, C = solver.solve()
    c1, c2 = C[:,j-f]
    aq, bq = P.T
    aq1 = aq*c1
    bq1 = bq

    range_ = slice(f,t+1)
    debug_info = sd, range_
    solver = BoundedLrfSolver(qv, D[:,range_], E[:,range_], debug_info=debug_info)
    P_, C_, Rg, R_, L_, hK, hL, bq_bounds_, coerced_bq_ = solver.solve(debug=True)
    aq_, bq_ = P_.T
    aqf = aq_*c1

    print("L_=", L_)

    TxT = True

    with plt.Dp():
        if TxT:
            fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(21,10))
            ax0 = axes[0,0]
            ax1 = axes[0,1]
            ax2 = axes[0,2]
            ax3 = axes[1,1]
            ax4 = axes[1,0]
        else:
            fig, (ax0, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(22,5))
        fig.suptitle("Bounded LRF Demo for %s" % (data_title), fontsize=20)
        ax0.set_title("Elution Data and Range", fontsize=16)
        ax2.set_title("LRF Result (Linear)", fontsize=16)
        ax3.set_title("LRF Result (Log)", fontsize=16)
        ax4.set_title("LRF Result (Guinier Plot)", fontsize=16)

        ax0.plot(x, y)
        ymin, ymax = ax0.get_ylim()
        p = Rectangle(
                (f, ymin),  # (x,y)
                t - f,   # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax0.add_patch(p)

        

        ax2.plot(qv, bq_*c2, color="pink", label="bounded LRF B(q)")
        if False:
            h_bq = -hK*phi(qv, 2*hL*R_)*aqf/c1
            ax2.plot(qv, h_bq*c2, color="yellow", label="Hard Sphere fit B(q)")

        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        for k, bound in enumerate(bq_bounds_):
            label = "estimated B(q) bound" if k == 0 else None
            ax2.plot(qv, bound*c2, color="red", alpha=0.5, label=label)

        ax2.legend(fontsize=14)

        ax3.set_yscale("log")
        for ax in ax2, ax3:
            ax.plot(qv, pv1, label="measured data")
            ax.plot(qv, aq1, label="naiive LRF A(q)", alpha=0.5)
            ax.plot(qv, aqf, label="bounded LRF A(q)", alpha=0.5)
        ax3.legend(fontsize=14)

        glim_i = bisect_right(qv, 2/Rg)
        gslice_disp = slice(0,glim_i)
        qv2 = qv[gslice_disp]**2

        glim_i = bisect_right(qv, 1.3/Rg)
        gslice = slice(0,glim_i)
        qv2_ = qv[gslice]**2
        intercept_list = []
        rgs = []
        for pv in pv1, aq1, aqf:
            glny_ = np.log(pv[gslice])
            slope, intercept = linregress(qv2_, glny_)[0:2]
            Rg = np.sqrt(-3*slope)
            rgs.append(Rg)
            # I0 = np.exp(intercept)
            intercept_list.append(intercept)

        ax4.plot(qv2, np.log(pv1[gslice_disp]) - intercept_list[0], label="measured data")
        ax4.plot(qv2, np.log(aq1[gslice_disp]) - intercept_list[1], label="naiive LRF A(q)", alpha=0.5)
        ax4.plot(qv2, np.log(aqf[gslice_disp]) - intercept_list[2], label="bounded LRF A(q)", alpha=0.5)
        ymin, ymax = ax4.get_ylim()
        ax4.set_ylim(ymin, ymax)
        qg = qv[glim_i]**2
        ax4.plot([qg, qg], [ymin, ymax], color="yellow", label="$qR_g=1.3$")
        xmin, xmax = ax4.get_xlim()
        tx = xmin*0.8 + xmax*0.2
        ty = ymin*0.7 + ymax*0.3
        ax4.text(tx, ty, "$R_g=%.3g$" % rgs[-1], ha="center", va="center", alpha=0.5, fontsize=20)
        ax4.legend(fontsize=14)

        fig.tight_layout()
        plt.show()

def demo(caller):
    dialog = caller.dialog
    pdata, popts = caller.get_preview_data(with_update=False)

    print("pdata.cnv_ranges=", pdata.cnv_ranges)
    paired_range = pdata.cnv_ranges[1]
    ranges = paired_range.get_fromto_list()
    print(ranges)
    f, t = ranges[1]
    sd = dialog.sd
    demo_impl(sd, f, t)

def demo_bak2(caller):
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
    from molass_legacy.Tools.EmbedCushionUtils import get_caller_attr

    in_folder_save = get_setting("in_folder")

    gi_in_folder = r"E:\PyTools\Data\20180526\GI"
    if not os.path.exists(gi_in_folder):
        gi_in_folder = r"D:\PyTools\Data\20180526\GI"
    cushion_dict = get_caller_attr(caller, "cushion_dict", {})
    gi_sd = cushion_dict.get("gi_sd")
    if gi_sd is None:
        from molass_legacy.Tools.SdUtils import get_sd
        cushion_dict["gi_sd"] = gi_sd = get_sd(gi_in_folder)

    set_setting("in_folder", gi_in_folder)

    f = 170
    t = 200
    demo_impl(gi_sd, f, t)

    set_setting("in_folder", in_folder_save)

def demo_bak3(caller):
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
    from molass_legacy.Tools.EmbedCushionUtils import get_caller_attr

    in_folder_save = get_setting("in_folder")

    gi_in_folder = r"E:\PyTools\Data\20180526\OA"
    if not os.path.exists(gi_in_folder):
        gi_in_folder = r"D:\PyTools\Data\20180526\OA"
    cushion_dict = get_caller_attr(caller, "cushion_dict", {})
    gi_sd = cushion_dict.get("oa_sd")
    if gi_sd is None:
        from molass_legacy.Tools.SdUtils import get_sd
        cushion_dict["oa_sd"] = gi_sd = get_sd(gi_in_folder)

    set_setting("in_folder", gi_in_folder)

    f = 250
    t = 310
    demo_impl(gi_sd, f, t)

    set_setting("in_folder", in_folder_save)


def demo_bak4(caller):
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
    from molass_legacy.Tools.EmbedCushionUtils import get_caller_attr

    in_folder_save = get_setting("in_folder")

    temp_in_folder = r"E:\PyTools\Data\20221227"
    if not os.path.exists(temp_in_folder):
        temp_in_folder = r"D:\PyTools\Data\20221227"
    cushion_dict = get_caller_attr(caller, "cushion_dict", {})
    temp_sd = cushion_dict.get("temp_sd")
    if temp_sd is None:
        from molass_legacy.Tools.SdUtils import get_sd
        cushion_dict["temp_sd"] = temp_sd = get_sd(temp_in_folder)

    set_setting("in_folder", temp_in_folder)

    f = 220
    t = 260
    demo_impl(temp_sd, f, t)

    set_setting("in_folder", in_folder_save)
