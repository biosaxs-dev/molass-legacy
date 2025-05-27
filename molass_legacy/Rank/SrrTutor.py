"""

    Rank.SrrTutor.py

    Copyright (c) 2022, SAXS Team, KEK-PF

"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from matplotlib.gridspec import GridSpec
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.Batch.StandardProcedure import StandardProcedure
from molass_legacy.DataStructure.MatrixData import simple_plot_3d
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def rankview_plot(fig, in_folder, sd, M, vec, data_type, xr_type,
                    pre_recog=None, trim=False, i_only=False, j_only=False,
                    data_label="",
                    view_kw=None, return_axes=False):
    if trim:
        if pre_recog is None:
            from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
            pre_recog = PreliminaryRecognition(sd)
        if xr_type:
            islice, jslice = pre_recog.get_xr_slices()
        else:
            islice, jslice = pre_recog.get_uv_slices()
        if i_only:
            jslice = slice(None, None)
        if j_only:
            islice = slice(None, None)
        D = M[islice,jslice]
        vec = vec[islice]
        trim_label = " trim:" + str([(s.start, s.stop) for s in [islice, jslice]]).replace("None", "_")
        trim_label = trim_label.replace("[", "").replace("]", "")
    else:
        D = M
        trim_label = ""

    N = np.linalg.norm(D)

    U, s, VT = np.linalg.svd(D)
    reconst = []
    residuals = []
    residuals_norms = []
    for rank in range(1,5):
        D_ = U[:,0:rank] @ np.diag(s[0:rank]) @ VT[0:rank,:]
        reconst.append(D_)
        R = D - D_
        residuals.append(R)
        residuals_norms.append(np.linalg.norm(R))

    gs = GridSpec(5,12)
    in_folder = get_in_folder(in_folder)
    fig.suptitle("SVD Reconstruction with various Ranks of %s (%s%s)%s" % (in_folder, data_label, data_type, trim_label), fontsize=24)
    ax0 = fig.add_subplot(gs[0:4,0:4], projection="3d")
    ax0.set_title("Whole Data", fontsize=20)
    ax0.set_axis_off()

    def my_plot_3d(ax, D, x, view_kw=None):
        simple_plot_3d(ax, D, x=x)
        ax.margins(0)
        if view_kw is not None:
            ax.view_init(**view_kw)

    my_plot_3d(ax0, D, vec, view_kw=view_kw)

    axes = []
    for i in range(2):
        i2 = i*2
        axes_row = []
        if i == 0:
            z_limits = []
        else:
            z_limits = np.array(z_limits)
            zmin = np.min(z_limits[:,0])
            zmax = np.max(z_limits[:,1])
        for j in range(4):
            j2 = 4 + j*2
            ax = fig.add_subplot(gs[i2:i2+2,j2:j2+2], projection="3d")
            ax.set_axis_off()
            if i == 0:
                ax.set_title("Rank %d" % (j+1), fontsize=20)
                my_plot_3d(ax, reconst[j], vec, view_kw=view_kw)
                z_limits.append(ax.get_zlim())
            else:
                my_plot_3d(ax, residuals[j], vec, view_kw=view_kw)
                ax.set_zlim(zmin, zmax)
            axes_row.append(ax)
        axes.append(axes_row)
    axes = np.array(axes)

    ax1 = fig.add_subplot(gs[4,0:4])
    ax1.set_axis_off()

    srr1 = residuals_norms[0]/N
    srr2 = residuals_norms[1]/N
    srr21 = srr2/srr1
    ax1.text(0.5, 0.7, r"$ SRR(1) = %.2g, \;\; SRR(2) = %.2g, \;\; SRR(2:1) = \frac{SRR(2)}{SRR(1)} = %.2g$" % (srr1, srr2, srr21), ha="center", fontsize=14)
    ax1.text(1.0, 0.4, "Norms of residual matrices are plotted in the right figures.", ha="right", fontsize=14)
    ax1.text(1.0, 0.2, r"Note that $ \| A + B \| \leq \| A \| + \| B \| $.", ha="right", fontsize=14)

    b_axes = []
    b_twin_axes = []
    y_limits = []
    for j in range(4):
        j2 = 4 + j*2
        ax = fig.add_subplot(gs[4,j2:j2+2])
        ax.set_axis_off()
        ax.bar(0, residuals_norms[j], width=0.5, alpha=0.5, label="residual norm (relative)")
        ax.legend()
        y_limits.append(ax.get_ylim())
        b_axes.append(ax)

        axt = ax.twinx()
        axt.set_axis_off()
        axt.bar(0, residuals_norms[j], width=0.5, alpha=0.5, color="C1", label="residual norm (proportion)")
        axt.legend(bbox_to_anchor=(1, 0.8), loc="upper right")
        b_twin_axes.append(axt)

    y_limits = np.array(y_limits)
    ymin = np.min(y_limits[:,0])
    ymax = np.max(y_limits[:,1])
    Nymax = N*ymax/residuals_norms[0]
    for j in range(4):
        ax = b_axes[j]
        ax.set_ylim(ymin, ymax)
        axt = b_twin_axes[j]
        axt.set_ylim(ymin, Nymax)

    fig.tight_layout()
    fig.subplots_adjust(top=0.88, left=0.01)

    if return_axes:
        return ax0, axes, ax1, b_axes, b_twin_axes

def srr_turtorial(in_folder, **kwargs):
    set_setting("test_pattern", 0)
    sp = StandardProcedure()
    sp.load_old_way(in_folder)
    correct = kwargs.pop("correct", False)
    trim = kwargs.pop("trim", False)
    if correct:
        sd = sp.get_corrected_sd(proxy=False)
    else:
        sd = sp.get_sd(whole=True)

    data_type = kwargs.pop("data_type", "xr")
    xr_type = data_type.lower().find("x") >= 0
    if xr_type:
        M, E, vec, ecurve = sd.get_xr_data_separate_ly()
    else:
        M, _, vec, ecurve = sd.get_uv_data_separate_ly()

    fig = plt.figure(figsize=(20,9))
    rankview_plot(fig, in_folder, sd, M, vec, data_type, xr_type, **kwargs)
    plt.show()
