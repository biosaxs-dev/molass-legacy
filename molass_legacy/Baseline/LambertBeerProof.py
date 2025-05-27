# coding: utf-8
"""

    LambertBeerProof.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""
import numpy as np
from matplotlib.collections import PolyCollection
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from ThreeDimUtils import compute_plane
from ModeledData import ModeledData, simple_plot_3d
from .LambertBeer import BasePlane, get_base_plane, get_args
from LmfitThreadSafe import minimize, Parameters

def proof_modeled_data_demo(inset=False):
    qvector = np.linspace(0.01, 0.6, 600)
    n_elutions = 300
    pd = ModeledData(qvector, n_elutions)
    M = pd.get_data()

    index = bisect_right(qvector, 0.02)
    bp1 = get_base_plane(M, index)

    a = 0
    b = 0.1/n_elutions
    c = 0.03
    x = np.arange(len(qvector))
    y = np.arange(n_elutions)
    BP = compute_plane(a, b, c, x, y)
    M_ = M + BP
    bp2 = get_base_plane(M_, index)

    n = 180
    M1 = M_[:,0:n]
    bp3 = get_base_plane(M1, index)

    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    view_init = (25, -10)
    # view_init = (-10, -10)

    bp1.debug_plot(ax1, view_init=view_init)
    bp2.debug_plot(ax2, view_init=view_init)
    bp2.debug_plot(ax3, plane_color='yellow')
    bp3.debug_plot(ax3, view_init=view_init)

    y_ = np.ones(len(x))*n
    z_ = M_[:,n]
    ax3.plot(x, y_, z_, color='green', linewidth=5)

    xmin, xmax = ax3.get_xlim()
    zmin, zmax = ax3.get_zlim()
    ax3.set_xlim(xmin, xmax)
    ax3.set_zlim(zmin, zmax)

    verts = [ [ [x[0], zmin], [x[0], zmax], [x[-1], zmax], [x[-1], zmin] ] ]
    poly = PolyCollection(verts, facecolor='green', alpha=0.2)
    ax3.add_collection3d(poly, zs=[n], zdir='y')

    if inset:
        x_ = np.ones(len(y))*150
        z_ = M_[150,:]
        px1 = []
        py1 = []
        pz1 = []
        for ax in [ax1, ax2, ax3]:
            ax.plot(x_,y, z_, color='orange')

    fig.tight_layout()
    plt.show()

def proof_feasibility_demo(sd):
    from DataUtils import get_in_folder
    data = sd.intensity_array[:,:,1].T
    xray_slice = sd.xray_slice
    index = (xray_slice.start + xray_slice.stop)//2
    ecurve = sd.xray_curve
    lb = BasePlane(data, index, ecurve)

    def compute_dvector(pname, vvector):
        params = {}
        params['A'] = 0
        params['B'] = 0
        params['C'] = 0

        d_list = []
        for v in vvector:
            params[pname] = v
            d_list.append(np.sum(lb.objective(params)**2))
        dvector = np.array(d_list)
        return dvector

    a_vvector = np.linspace(-0.0002, 0.0002, 100)
    a_dvector = compute_dvector('A', a_vvector)
    b_vvector = np.linspace(-0.00005, 0.00005, 100)
    b_dvector = compute_dvector('B', b_vvector)
    c_vvector = np.linspace(-0.002, 0.002, 100)
    # c_vvector = np.linspace(-0.02, 0.02, 100)
    c_dvector = compute_dvector('C', c_vvector)

    nrows = 2
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(21,10))

    in_folder = get_in_folder()
    fig.suptitle(r"Variation of the Objective Fuction $ \| P \cdot C - M \|_{a,b,c}$ with Data " + in_folder, fontsize=30)

    for j in range(ncols):
        axes[1,j].set_yscale('log')
    for i in range(nrows):
        axes[i,0].plot(a_vvector, a_dvector)
        axes[i,1].plot(b_vvector, b_dvector)
        axes[i,2].plot(c_vvector, c_dvector)

    texts = ["a", "b", "c"]
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i,j]
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            ax.plot([0,0], [ymin, ymax], ':', color='red')
            tx = (xmin+xmax)/2
            if i == 0:
                ty = (ymin+ymax)/2
            else:
                ty = np.sqrt(ymin*ymax)
            ax.text(tx, ty, texts[j], alpha=0.05, ha='center', va='center', fontsize=200)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()

class BaseSurfaceSpike:
    def __init__(self, data, index, ecurve):
        self.data = data
        self.index = index
        self.ecurve = ecurve

        self.compute_divided_belts()
        self.compute_soomth_surface()

    def compute_divided_belts(self):
        length = self.data.shape[0]
        belt_width = 10
        params_list = []
        for i in range(0, length, belt_width):
            start = i
            stop = min(length, i+belt_width)
            bp = BasePlane(self.data[start:stop,:], 0, self.ecurve)
            bp.solve(debug=False, debug_info=[start, stop])
            params_list.append(bp.get_params())

        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for k, i in enumerate(range(0, length, belt_width)):
                a, b, c = params_list[k]
                x = np.array([i, i])
                y = np.array([0, self.data.shape[1]-1])
                z = a*x + b*y + c
                ax.plot(x, y, z, color='red')
            fig.tight_layout()
            plt.show()

    def compute_soomth_surface(self):
        pass

def compute_basesurface(sd, pre_recog, debug=False):
    args = get_args(sd, pre_recog, debug)
    bs = BaseSurfaceSpike(*args)

def disproof_2d():
    qvector = np.linspace(0.01, 0.6, 600)
    n_elutions = 300
    pd = ModeledData(qvector, n_elutions)
    M = pd.get_data()

    index = bisect_right(qvector, 0.02)

    x = np.arange(n_elutions)
    y = M[index,:]

    A = 0.0005
    B = -0.01
    b_ = A*x + B
    y_ = y + b_

    def objective(params):
        a = params['A']
        b = params['B']
        base = A*x + B
        yc = y_ - base

        h = yc[150]
        yh = yc*h
        return yc - yh

    params = Parameters()
    params.add('A', value=0,   min=-1e-2,  max=+1e-2 )
    params.add('B', value=0,   min=-1e-2,  max=+1e-2 )

    result =  minimize( objective, params, args=() )
    A_ = result.params['A'].value
    B_ = result.params['B'].value

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
    ax1, ax2, ax3 = axes
    for ax in axes[0:2]:
        ax.plot(x, y)
        ax.plot(x, y_)
    ax1.plot(x, b_, ':', color='red')
    ax2.plot(x, A_*x + B_, color='red')
    fig.tight_layout()
    plt.show()
