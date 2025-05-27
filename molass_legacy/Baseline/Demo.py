# coding: utf-8
"""
    Baseline.Demo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Polygon
from MeasuredData import MeasuredData
import molass_legacy.KekLib.DebugPlot as plt
from ThreeDimViewer import ThreeDimViewer
from MatrixData import simple_plot_3d
from .Baseline import compute_baseline, better_integrative_curve
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from CurveDecomposer import decompose

BUG_FIX_WITH_REDRAW = False
TITLE_FONTSIZE = 16

def demo(root, in_folder, xray=True, show_preview=False):
    print(in_folder)
    md = MeasuredData(in_folder)

    if show_preview:
        dialog = ThreeDimViewer(root, md)
        dialog.show()

    if xray:
        rd = md.xr
        ecolor = 'orange'
    else:
        rd = md.uv
        ecolor = 'blue'

    rd.set_elution_curve()
    i_slice = rd.i_slice
    j_slice = rd.j_slice

    whole_data = rd.data
    data = whole_data[i_slice,j_slice]
    q = rd.vector[i_slice]

    fig = plt.figure(figsize=(21, 11))
    gs = GridSpec(2,3)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax10 = fig.add_subplot(gs[1,0])
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])

    ax00.set_title("3D Data View", y=1.09, fontsize=TITLE_FONTSIZE)
    simple_plot_3d(ax00, data, x=q)
    z = rd.e_curve.y[j_slice]

    q1 = rd.vector[rd.e_index]
    size = data.shape[1]
    x = np.ones(size)*q1
    y = np.arange(size)

    ax00.plot(x, y, z, color=ecolor)

    baseline = None
    for ax1, ax2, integral in [
            (ax01, ax02, False), 
            (ax11, ax12, True),
            ]:
        if integral:
            title = "LPM Integral Baseline"
        else:
            title = "LPM Linear Baseline"
        b = compute_baseline(z, integral=integral)
        if baseline is None:
            baseline = b
        z_ = z - b
        ec = ElutionCurve(z_)
        recs = decompose(ec)

        ax1.set_title(title, fontsize=TITLE_FONTSIZE)
        ax1.plot(z, color=ecolor)
        ax1.plot(b, ':', color='red')
        ax2.set_title("Decomposition after Correction", fontsize=TITLE_FONTSIZE)
        ax2.plot(z_, color=ecolor)
        for k, rec in enumerate(recs):
            func = rec.evaluator
            ax2.plot(y, func(y), ':', color='C%d' % k)

    # ax10.set_ylim(-0.01, 0.2)

    anim = make_anim(fig, ax10, y, z, baseline, ecolor)
    fig.tight_layout()
    # anim.save("anim.mp4", writer="ffmpeg")
    plt.show()

def draw_anim_init(ax, x, y, b, ecolor, n):
    ax.set_title("LPM Integral Baseline Algorithm Animation", fontsize=TITLE_FONTSIZE)
    ax.plot(x, y, color=ecolor)
    line0, = ax.plot(x, b, ':', color='green')
    area_points = [(x[0], b[0])] + list(zip(x,y)) + [(x[-1], b[0])]
    area1 = Polygon(area_points, alpha=0.3, color='cyan')
    ax.add_patch(area1)

    tx = np.average(ax.get_xlim())
    ymin, ymax = ax.get_ylim()
    ty = ymin*0.6 + ymax*0.4
    num_text = ax.text(tx, ty, str(n), alpha=0.05, fontsize=300, ha="center", va="center")
    return line0, area1, num_text

def make_anim(fig, ax, x, y, b, ecolor):

    line0, area1, num_text = draw_anim_init(ax, x, y, b, ecolor, 1)

    num_frames1 = 60
    num_frames2 = 60
    num_frames3 = 30
    num_frames4 = 60
    num_frames = num_frames1 + num_frames2 + num_frames3 + num_frames4

    ymin, ymax = ax.get_ylim()

    smear_rate = 0.05
    end_height = b[-1]  # or y[-1]
    area2 = None
    integ_height = ymax*0.2

    start_y = y[0]
    y_ = y - b
    cy = np.cumsum(y_*smear_rate)
    height = cy[-1] - cy[0]
    ratio = (y[-1] - start_y)/height
    integ_y = cy*ratio

    b2 = start_y + integ_y
    y2_ = y - b2
    cy2 = np.cumsum(y2_)
    height = cy2[-1] - cy2[0]
    ratio = y2_[-1]/height
    integ_y2 = cy2*ratio

    lineb = None

    def make_area_points(y1, y2):
        r = list(zip(x,y2))
        r.reverse()
        return list(zip(x,y1)) + r

    def init():
        line0.set_data(x, b)
        num_text.set_text("1")
        return line0, area1, num_text

    def animate_impl(i, a_integ_y, a_baseline):
        nonlocal area2, lineb
        if i < num_frames1:
            ratio = smear_rate + (1 - smear_rate)*(1 - i/num_frames1)
            # print([i], ratio)
            yp = a_baseline + y_*ratio
            area_points = make_area_points(yp, a_baseline)
            area1.set_xy(area_points)
            return line0, area1, num_text

        if i < num_frames1 + num_frames2:
            i_ = i - num_frames1
            if i_ == 0:
                area_points = [(x[0], start_y), (x[0], start_y)]
                area2 = Polygon(area_points, alpha=0.3, color='pink')
                ax.add_patch(area2)
            else:
                # i.e. i_ > 0
                j = int(len(x)*i_/num_frames2)
                yp = a_baseline[0] + a_integ_y[0:j]
                area_points = [(x[0], start_y)] + list(zip(x[0:j], yp)) + [(x[max(0,j-1)], start_y)]
                area2.set_xy(area_points)
            return line0, area1, area2, num_text

        if i < num_frames1 + num_frames2 + num_frames3:
            i_ = i - (num_frames1 + num_frames2)
            w = (i_+1)/num_frames3
            yp = a_baseline[0] + a_integ_y*( 1*(1-w) + (a_baseline[-1]-a_baseline[0])/a_integ_y[-1]*w )
            area_points = [(x[0], start_y)] + list(zip(x, yp)) + [(x[-1], start_y)]
            area2.set_xy(area_points)
            return line0, area1, area2, num_text

        i_ = i - (num_frames1 + num_frames2 + num_frames3)
        if i_ == 0:
            lineb, = ax.plot(x, b2, ':', color='red')

        return line0, area1, area2, lineb, num_text

    def animate(i):
        nonlocal line0, area1, num_text

        if i < num_frames:
            return animate_impl(i, integ_y*integ_height/integ_y[-1], b)

        i -= num_frames
        if i == 0:
            if BUG_FIX_WITH_REDRAW:
                ax.cla()
                line0, area1, num_text = draw_anim_init(ax, x, y, b2, ecolor, 2)
            else:
                line0.set_data(x, b2)
                area2.remove()
                num_text.set_text("2")

        return animate_impl(i, integ_y2*integ_height/integ_y2[-1], b2)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_frames*2, interval=100, blit=True)
    return anim

def proof_plot(in_folder, xray=True, better=True):
    from DataUtils import get_in_folder

    md = MeasuredData(in_folder)

    if xray:
        rd = md.xr
        ecolor = 'orange'
    else:
        rd = md.uv
        ecolor = 'blue'

    j_slice = rd.j_slice
    x = rd.e_curve.x[j_slice]
    y = rd.e_curve.y[j_slice]

    end_slices = rd.e_curve.get_end_slices()
    if better:
        b, convex = better_integrative_curve(y, end_slices=end_slices)
    else:
        b = compute_baseline(y, integral=True, end_slices=end_slices)

    fig, ax = plt.subplots()
    ax.set_title("Proof Plot of Integral Basecurve for " + get_in_folder(in_folder), fontsize=20)
    ax.plot(x, y, color=ecolor, label='data')
    ax.plot(x, b, ':', color='red', label='integral basecurve')

    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.show()
