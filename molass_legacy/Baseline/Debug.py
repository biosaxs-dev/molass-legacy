# coding: utf-8
"""
    Baseline.Debug.py

    Copyright (c) 2020-2022, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.patches import Polygon
from MeasuredData import MeasuredData
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from CurveDecomposer import decompose
from LPM import LPM_3d
# from molass_legacy.Baseline.LpmProxy import LpmProxy as LPM_3d
USING_LPM_PROXY = False
from Extrapolation.ExSolver import ExSolver
from molass_legacy.KekLib.NumpyUtils import np_savetxt
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve

BUG_FIX_WITH_REDRAW = False
TITLE_FONTSIZE = 16

def demo(root, in_folder, xray=True, show_preview=False):
    print(in_folder)
    md = MeasuredData(in_folder)

    if show_preview:
        from molass_legacy.Tools.ThreeDimViewer import ThreeDimViewer
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
    j_start = j_slice.start
    if j_start is None:
        j_start = 0

    print('i_slice=', i_slice)
    print('j_slice=', j_slice)

    whole_data = rd.data
    data = whole_data[i_slice,j_slice]
    print('data.shape=', data.shape)

    q = rd.vector[i_slice]
    q1 = rd.vector[rd.e_index]

    fig = plt.figure(figsize=(21, 11))
    fig.suptitle("Inspection of Deviation in the Wide Angle Region", fontsize=24)

    gs = GridSpec(2,3)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')
    ax01 = fig.add_subplot(gs[0,1])
    ax02 = fig.add_subplot(gs[0,2])
    ax10 = fig.add_subplot(gs[1,0], projection='3d')
    ax11 = fig.add_subplot(gs[1,1])
    ax12 = fig.add_subplot(gs[1,2])
    z = rd.e_curve.y[j_slice]
    pt_i = rd.e_curve.primary_peak_i - j_start
    hw = 5
    top_slice = slice(pt_i-hw, pt_i+hw)
    range_slice = top_slice
    if USING_LPM_PROXY:
        mapper = md.create_mapper(jslice=j_slice)
    else:
        mapper = None

    i_start = i_slice.start
    if i_start is None:
        i_start = 0

    e_index = rd.e_index - i_start
    demo_impl(ax00, ax01, ax02, data, q, q1, z, top_slice, range_slice, e_index, ecolor, mapper)

    j_slice = slice(846, 1185)
    j_start = j_slice.start
    if j_start is None:
        j_start = 0

    print('j_slice=', j_slice)

    data = whole_data[i_slice,j_slice]
    z = rd.e_curve.y[j_slice]
    pt_i = rd.e_curve.primary_peak_i - j_start
    top_slice = slice(pt_i-hw, pt_i+hw)
    range_slice = slice(900-j_start, 1016-j_start)
    if USING_LPM_PROXY:
        mapper = md.create_mapper(jslice=range_slice)
    else:
        mapper = None

    demo_impl(ax10, ax11, ax12, data, q, q1, z, top_slice, range_slice, e_index, ecolor, mapper, save=False)

    x = rd.vector
    for ax, j0 in [(ax00, 0), (ax10, j_slice.start)]:
        for y in [j_slice.start, j_slice.stop]:
            y_ = y - j0
            ax.plot(x[[0,-1]], [y_, y_], [0, 0], color='yellow' )

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

def demo_impl(ax00, ax01, ax02, data, q, q1, z, top_slice, range_slice, e_index, ecolor, mapper, save=False):
    ax00.set_title("3D Data View", y=1.09, fontsize=TITLE_FONTSIZE)
    simple_plot_3d(ax00, data, x=q)

    size = data.shape[1]
    x = np.ones(size)*q1
    y = np.arange(size)

    ax00.plot(x, y, z, color=ecolor)

    # raw pt average
    pt_y = np.average(data[:,top_slice], axis=1)

    if save:
        save_data = np.array([q, pt_y, pt_y*0.03]).T
        np_savetxt('temp.dat', save_data)

    # lpm pt average
    if USING_LPM_PROXY:
        lpm = LPM_3d(data, ecurve_y=z, integral=False, for_all_q=True, e_index=e_index, q=q, mapper=mapper)
    else:
        lpm = LPM_3d(data, ecurve_y=z, integral=False, for_all_q=True, e_index=e_index)

    lp_y = np.average(lpm.data[:,top_slice], axis=1)

    M0 = data[:,range_slice]
    M1 = lpm.data[:,range_slice].copy()

    e_curve = ElutionCurve(z)
    lpm.adjust_with_mf(e_index, e_curve)
    lp_y2 = np.average(lpm.data[:,top_slice], axis=1)
    M2 = lpm.data[:,range_slice].copy()
    # solver
    solver = ExSolver()
    c = z[range_slice]
    c = c/np.max(c)

    # raw pt extrapolation (cd=1)
    ex_y0 = solver.solve(M0, c=c, cd=1)

    # lpm pt extrapolation (cd=1)
    ex_y1 = solver.solve(M1, c=c, cd=1)
    ex_y2 = solver.solve(M2, c=c, cd=1)

    ax01.set_title("Peak Top Averages", fontsize=16)
    ax01.set_yscale('log')
    ax01.plot(q, pt_y, label="raw average")
    ax01.plot(q, lp_y, label="lpm average")
    ax01.plot(q, lp_y2, label="lpm+MF average")
    ax01.legend()

    ax02.set_title("Ascending-side Extrapolations", fontsize=16)
    ax02.set_yscale('log')
    ax02.plot(q, ex_y0, label="raw extrapolation (cd=1)")
    ax02.plot(q, ex_y1, label="lpm extrapolation (cd=1)")
    ax02.plot(q, ex_y2, label="lpm+MF extrapolation (cd=1)")
    ax02.legend()
