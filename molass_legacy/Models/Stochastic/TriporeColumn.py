"""
    Models/Stochastic/TriporeColumn.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.SecPDF import FftInvPdf
from SecTheory.SecCF import gec_tripore_phi
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.Models.Stochastic.Tripore import PORESIZES, POREPROPS

tripore_pdf = FftInvPdf(gec_tripore_phi)

def triporecolumn_func(me, mp, sv, pv, x, scale, N, T, Rg, x0):
    rho = Rg/sv
    rho[rho > 1] = 1
    np_ = N * pv * (1 - rho)**me
    tp_ = T*(1 - rho)**mp
    return scale*tripore_pdf(x, np_[0], tp_[0], np_[1], tp_[1], np_[2], tp_[2], x0)

class TriporeColumn:
    def __init__(self):
        self.N = 2000
        self.T = 8
        self.x0 = 50
        self.me = 1.5
        self.mp = 1.5
        self.sv = PORESIZES
        self.pv = POREPROPS

    def update_column_params(self, params):
        self.N = params[0]
        self.T = params[1]
        self.x0 = params[2]
        self.sv = params[3:6]
        self.pv = params[6:9]

    def estimate_psd(self, x, y, rgs, tRs, area_props, init_scales=None, use_basinhopping=False):
        N, T, x0 = [self.N, self.T, self.x0]
        def scale_objective(scales, debug=False):
            cy_list = []
            for rg, scale in zip(rgs, scales):
                cy = triporecolumn_func(self.me, self.mp, self.sv, self.pv, x, scale, N, T, rg, x0)
                cy_list.append(cy)
            ty = np.sum(cy_list, axis=0)
            props = compute_area_props(cy_list)
            if debug:
                with plt.Dp():
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
                    fig.suptitle("objective debug")
                    ax1.plot(x, y)
                    for cy in cy_list:
                        ax1.plot(x, cy, ":")
                    ax1.plot(x, ty, ":", color="red")
                    ax2.bar(self.sv, self.pv, width=10)
                    fig.tight_layout()
                    plt.show()
            return np.sum((y - ty)**2) + np.sum((props - area_props)**2)

        if init_scales is None:
            init_scales = np.ones(len(rgs))
        scale_objective(init_scales, debug=True)
        res = minimize(scale_objective, init_scales, method='Nelder-Mead')
        scale_objective(res.x, debug=True)
        init_scales = res.x

        def objective(p, debug=False, title=None):
            N = p[0]
            T = p[1]
            x0 = p[2]
            sv = p[3:6]
            pv = p[6:9]
            scales = p[9:]
            cy_list = []
            for rg, scale in zip(rgs, scales):
                cy = triporecolumn_func(self.me, self.mp, sv, pv, x, scale, N, T, rg, x0)
                cy_list.append(cy)
            props = compute_area_props(cy_list)
            ty = np.sum(cy_list, axis=0)
            dsv = np.diff(sv)
            order_penalty = np.sum(dsv[dsv>0])
            if debug:
                print("N, T, x0=", N, T, x0)
                print("sv, pv=", sv, pv)
                print("dsv=", dsv, order_penalty)
                with plt.Dp():
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
                    if title is None:
                        title = "objective debug"
                    fig.suptitle(title)
                    ax1.plot(x, y)
                    for cy, rg in zip(cy_list, rgs):
                        ax1.plot(x, cy, ":", label="Rg=%g" % rg)
                    ax1.plot(x, ty, ":", color="red", label="total")
                    ax1.legend()
                    ax2.bar(sv, pv, width=10)
                    fig.tight_layout()
                    plt.show()
            # return np.sum((y - ty)**2) + abs(1 - np.sum(pv)) + np.sum((props - area_props)**2) + max(100, np.std(sv))
            return np.sum((y - ty)**2) + abs(1 - np.sum(pv)) + np.sum((props - area_props)**2) + order_penalty

        init_params = np.concatenate([[self.N, self.T, self.x0], self.sv, self.pv, init_scales])
        bounds = [(0, 5000), (0, 10), (-100, 100)] + [(10, 300)]*3 + [(0, 1)]*3 + [(0, 100)]*len(rgs)
        objective(init_params, debug=True, title="before minimize")
        if use_basinhopping:
            minimizer_kwargs = dict(method='Nelder-Mead', bounds=bounds)
            res = basinhopping(objective, init_params, minimizer_kwargs=minimizer_kwargs, niter=10)
        else:
            method='Nelder-Mead' 
            res = minimize(objective, init_params, method=method, bounds=bounds)
        objective(res.x, debug=True, title="after minimize")
        return res.x

    def compute_cy_list(self, x, rgs, tripore_params):
        N, T, x0 = tripore_params[:3]
        sv = tripore_params[3:6]
        pv = tripore_params[6:9]
        scales = tripore_params[9:]
        cy_list = []
        for rg, scale in zip(rgs, scales):
            cy = triporecolumn_func(self.me, self.mp, sv, pv, x, scale, N, T, rg, x0)
            cy_list.append(cy)
        return cy_list

    def cover_missing_data(self, rg_list, last_rgs=None):
        if last_rgs is None:
            new_rg_list = rg_list
        else:
            new_rg_list = []
            for rg, last_rg in zip(rg_list, last_rgs):
                if rg is None:
                    new_rg_list.append(last_rg)
                else:
                    new_rg_list.append(rg)
        return np.array(new_rg_list)

    def get_curve(self, x, scale, *params):
        """
        all: N, T, me, mp, pore-distribution, Rg, scale, x0
        pore-distribution:
            pore sizes
            pore proportions

        mid: N, T, Rg, scale, x0  ---- implemented below

        min: Rg, scale, x0
        """
        return triporecolumn_func(self.me, self.mp, self.sv, self.pv, x, scale, *params)
    
def estimate_psd_impl(advanced_frame, devel=True):
    editor = advanced_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    print("estimate_psd_impl", model.get_name(), model.__class__)
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    D, E, qv, ecurve = editor.sd.get_xr_data_separate_ly()
    peak_region = ecurve.get_peak_region(sigma_scale=5)
    print("peak_region=", peak_region)

    def plot_cy_list(cy_list, title):
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.plot(x, y)
            for cy in cy_list:
                ax.plot(x, cy, ":", label="component-%d" % (len(cy_list)))
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", label="model total")
            for x_ in peak_region:
                ax.axvline(x=x_+x[0], color="green")
            fig.tight_layout()
            plt.show()

    slice_ = slice(*[int(x_) for x_ in peak_region])
    num_components = len(params_array)
    print("num_components=", num_components)
    M_ = get_denoised_data(D[:,slice_], rank=num_components)
    E_ = E[:,slice_]
    cy_list = compute_cy_list(model, fx, params_array)
    plot_cy_list(cy_list, "initial state")

    min_msd = None
    column = TriporeColumn()
    init_scales = None
    last_rgs = None
    for k in range(3):
        C = np.array(cy_list)
        C_ = C[:,slice_]
        Cinv = np.linalg.pinv(C_)
        P_ = M_ @ Cinv
        Minv = np.linalg.pinv(M_)
        W = Minv @ P_
        Pe = np.sqrt(E_**2 @ W**2)

        rg_list = []
        for j, p_ in enumerate(P_.T):
            data = np.array([qv, p_, Pe[:,j]]).T
            sg = SimpleGuinier(data)
            rg_list.append(sg.Rg)

        rgs = column.cover_missing_data(rg_list, last_rgs=last_rgs)
        tRs = None
        area_props = compute_area_props(cy_list)
        tripore_params = column.estimate_psd(fx, y, rgs, tRs, area_props, init_scales=init_scales)
        temp_cy_list = column.compute_cy_list(fx, rgs, tripore_params)
        ty = np.sum(temp_cy_list, axis=0)
        msd = np.sum((y - ty)**2)
        print("msd=", msd)
        if min_msd is None or msd < min_msd:
            cy_list = temp_cy_list
            min_msd = msd
            column.update_column_params(tripore_params)
            plot_cy_list(cy_list, "%d-th estimate" % k)
            last_rgs = rgs
            init_scales = tripore_params[9:]
        else:
            break