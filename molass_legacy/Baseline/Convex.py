# coding: utf-8
"""
    Baseline.Convex.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from matplotlib.gridspec import GridSpec
from molass_legacy.DataStructure.MatrixData import simple_plot_3d

TITLE_FONTSIZE = 16

def get_baseline_vertices(size, vertices):
    n0 = np.argmin(vertices)
    n1 = np.argmax(vertices)
    assert vertices[n0] == 0
    assert vertices[n1] == size - 1

    if n1 > n0:
        return vertices[n0:n1+1]
    else:
        return np.hstack([vertices[n0:], vertices[0:n1+1]])

def compute_baseline(y, vertices):
    i = vertices[0]
    yi = y[i]
    ret_y = np.zeros(len(y))

    for j in vertices[1:]:
        w = j-i
        w_ = np.arange(w)/w
        yj = y[j]
        dy = yj - yi
        ret_y[i:j] = yi + dy*w_
        i = j
        yi = yj

    ret_y[i] = yi
    return ret_y

def get_best_adjust_params(y, ys, end_limits=(10, -10)):
    """
    it is better to change end_limits to suite the situation
    """

    zeros = np.zeros(len(y))
    init_rate = y[-1]/ys[-1]
    ls = slice(0, end_limits[0])
    ms = slice(end_limits[0], end_limits[1])
    rs = slice(end_limits[1], None)

    def obj_func(x):
        rate, base = x
        return (np.sum((y[ls] - (base + ys[ls]*rate))**2)
                    + np.sum(np.min([y[ms] - (base + ys[ms]*rate), zeros[ms]], axis=0)**2)
                    + np.sum((y[rs] -(base + ys[rs]*rate))**2)
                    )

    res = minimize(obj_func, (init_rate, 0))
    return res.x

def demo(root, in_folder, xray=True, show_preview=False, debug=False):
    import molass_legacy.KekLib.DebugPlot as plt
    from MeasuredData import MeasuredData

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

    whole_data = rd.data
    data = whole_data[i_slice,j_slice]
    q = rd.vector[i_slice]

    fig = plt.figure(figsize=(21, 11))
    gs = GridSpec(3,4)
    ax00 = fig.add_subplot(gs[0,0], projection='3d')

    axes = []
    for i in range(3):
        for j in range(4):
            if i+j == 0:
                pass
            else:
                axes.append(fig.add_subplot(gs[i,j]))

    ax00.set_title("3D Data View", y=1.09, fontsize=TITLE_FONTSIZE)
    simple_plot_3d(ax00, data, x=q)
    z = rd.e_curve.y[j_slice]

    q1 = rd.vector[rd.e_index]
    size = data.shape[1]
    x = np.ones(size)*q1
    y = np.arange(size)

    ax00.plot(x, y, z, color=ecolor)

    baseline = None
    zeros = np.zeros(size)
    for ax in axes:
        ax.plot(y, z, color=ecolor)

        if baseline is None:
            points = np.array(list(zip(y,z)))
            hull = ConvexHull(points)
            ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'o-', color='yellow')

            print('hull.vertices=', hull.vertices)
            vertices = get_baseline_vertices(size, hull.vertices)
            ax.plot(points[vertices,0], points[vertices,1], 'o', color='red')
            baseline = compute_baseline(z, vertices)

        ax.plot(y, baseline, ':', color="red")

        zc = np.max([z - baseline, zeros], axis=0)
        # zc = z - baseline
        zs = np.cumsum(zc)
        rate, base = get_best_adjust_params(z, zs)

        if debug:
            ax.plot(y, zs*0.01, color='cyan')
            print('rate, base=', rate, base)

        zs_ = zs*rate + base
        ax.plot(y, zs_, '-', color="red")
        baseline = zs_

    fig.tight_layout()
    plt.show()

def integrative_curve_convex(y, baseline=None, epsilon=1e-4, return_hull=False):
    size = len(y)

    if baseline is None:
        x = np.arange(size)
        points = np.array(list(zip(x,y)))
        hull = ConvexHull(points)
        vertices = get_baseline_vertices(size, hull.vertices)
        baseline = compute_baseline(y, vertices)
        if False:
            print('hull.vertices=', hull.vertices)
            plt.push()
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'o-', color='yellow')
            ax.plot(x, baseline, ':', color='red')
            fig.tight_layout()
            plt.show()
            plt.pop()

    min_e = None
    min_baseline = None
    zeros = np.zeros(size)
    for k in range(5):
        y_ = np.max([y - baseline, zeros], axis=0)
        # y_ = y - baseline
        cy = np.cumsum(y_)
        height = cy[-1] - cy[0]

        rate, base = get_best_adjust_params(y, cy)
        new_baseline = cy*rate + base

        e = np.max(np.abs(new_baseline - baseline))/abs(height)
        # print([k], 'e=', e)
        if e < epsilon:
            break

        if min_e is None or e < min_e:
            min_e = e
            min_baseline = new_baseline
        else:
            new_baseline = min_baseline
            break

        baseline = new_baseline

    if return_hull:
        return new_baseline, points[hull.vertices, :]
    else:
        return new_baseline

def proof_plot(in_folder, xray=True):
    import molass_legacy.KekLib.DebugPlot as plt
    from MeasuredData import MeasuredData
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

    b, vertices = integrative_curve_convex(y, return_hull=True)

    j0 = j_slice.start
    if j0 is None:
        j0 = 0
    fig, ax = plt.subplots()
    ax.set_title("Proof Plot of Convex Hull Method for " + get_in_folder(in_folder), fontsize=20)
    ax.plot(x, y, color=ecolor, label='data')
    ax.plot(j0+vertices[:,0], vertices[:,1], 'o-', color='yellow', label='convex hull')
    ax.plot(x, b, ':', color='red', label='integral basecurve')

    ax.legend(fontsize=16)
    fig.tight_layout()
    plt.show()
