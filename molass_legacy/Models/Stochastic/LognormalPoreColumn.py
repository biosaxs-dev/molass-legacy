"""
    Models/Stochastic/LognormalPoreColumn.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.optimize import minimize, basinhopping
import time
from matplotlib.widgets import Slider, Button
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.Models.ElutionModelUtils import compute_4moments
from Experiment.ColumnTypes import get_all_poresizes
from molass_legacy.Models.Stochastic.SecModelUtils import (
    DEFAULT_SIGMA, SIGMA_LOWER_BOUND,
    NUM_ENTRIES_LIST, DEFAULT_NUM_ENTRIES, DEFAULT_ENTRY_TIME,
    compute_tRv
    )
from molass_legacy.Models.Stochastic.LognormalPoreFunc import lognormal_pore_cf, lognormal_pore_func, distr_func

def compute_residual_ratio_impl(x, y, rgs, params, debug=False):
    cy_list = []
    for rg, scale in zip(rgs, params[7:]):
        cy = lognormal_pore_func(x, scale, params[0], params[1], params[3], params[4], params[5], params[6], rg, params[2])
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)
    ratio = np.sum(np.abs(y - ty))/np.sum(y)

    if debug:
        print("ratio=", ratio)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(x, y)
            for cy in cy_list:
                ax.plot(x, cy, ":")
            ax.plot(x, ty, ":", color="red")
            fig.tight_layout()
            plt.show()

    return ratio

def plot_state_impl(x, y, title, cy_list, ty, mu, sigma, save_fig_as=None):
    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle(title, fontsize=20)
        ax1.set_title("Elution Curve", fontsize=16)
        ax2.set_title("Pore Size Distribution", fontsize=16)
        ax1.plot(x, y)
        for cy in cy_list:
            ax1.plot(x, cy, ":")
        ax1.plot(x, ty, ":", color="red")
        mode = np.exp(mu - sigma**2)
        max_size = int(mode + 200)
        z = np.arange(max_size)
        ax2.plot(z, distr_func(z, mu, sigma), label="PSD: Lognormal(%.3g, %.3g)" % (mu, sigma))
        xmin, xmax = ax2.get_xlim()
        ymin, ymax = ax2.get_ylim()
        tx = xmin*0.3 + xmax*0.7
        ty = ymin*0.5 + ymax*0.5
        poresize = np.exp(mu - sigma**2)
        ax2.text(tx, ty, "Poresize = %.3g" % poresize, ha="center", alpha=0.5, fontsize=16)
        ax2.legend()            
        fig.tight_layout()
        if save_fig_as is not None:
            plt.show(block=False)
            fig.savefig(save_fig_as)
            ret = True
        else:
            ret = plt.show()
    return ret

class LognormalPoreColumn:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.N = DEFAULT_NUM_ENTRIES
        self.T = DEFAULT_ENTRY_TIME
        self.x0 = 0
        self.me = 1.5
        self.mp = 1.5
        self.mu = 4.5
        self.sigma = DEFAULT_SIGMA
        self.evaluate_area_props = False
        self.abort = False

    def get_column_params(self, T=None, x0=None, mu=None, sigma=None):
        if T is None:
            T = self.T
        if x0 is None:
            x0 = self.x0
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        return np.array([self.N, T, x0, self.me, self.mp, mu, sigma])

    def prepare_for_rgs(self, x, y, rg_recs, debug=False):    
        self.init_scales = None
        W, M1, M2, M3 = compute_4moments(x, y)
        self.M1 = M1
        s = np.sqrt(M2)
        self.x0 = M1 - 8*s
        self.data_width = 3*s
        print("x0=%.3g, data_width=%.3g" % (self.x0, self.data_width))
        self.area = W

        self.rgs = rg_recs[:,1]
        max_rg = np.max(self.rgs)

        poresizes = np.asarray(get_all_poresizes())
        possible_poresizes = poresizes[poresizes > max_rg + 20]
        self.logger.info("trying for rgs=%s against possible_poresizes: %s", str(self.rgs), str(possible_poresizes))

        ret_list = []
        for poresize in possible_poresizes:
            ret = self.prepare_for_poresize(x, y, rg_recs, poresize, debug=False)
            if ret is None:
                return False
            ret_list.append(ret)

        # select the best poresize
        ret_array = np.asarray(ret_list)
        fv_list = ret_array[:,0]
        j = np.argmin(fv_list)
        self.poresize = possible_poresizes[j]
        self.x0 = ret_array[j,1]
        self.N = ret_array[j,2]
        self.logger.info("selected poresize=%.3g with x0=%.3g from evaluations %s", self.poresize, self.x0, str([round(v) for v in fv_list]))
        if debug:
            fv = fv_list[j]
            m = self.me + self.mp
            t_rgs = rg_recs[:,0]
            tRv = compute_tRv(self.N, self.T, self.x0, m, self.poresize, self.rgs)
            ret = self.plot_exclusion_state(x, y, self.poresize, self.rgs, self.x0, t_rgs, tRv, fv, title="selected exclusion state")
            if not ret:
                return False
        self.estimate_mu_sigma_from_poresize(self.poresize, debug=debug)
        return True

    def prepare_for_poresize(self, x, y, rg_recs, poresize, fit_optimize=False, debug=False):
        self.logger.info("preparing for poresize(%.3g)", poresize)
        rgs = rg_recs[:,1]

        m = self.me + self.mp
        t_rgs = rg_recs[:,0]
        def prepare_objective(p, debug=False, title=None):
            tRv = compute_tRv(self.N, self.T, p[0], m, poresize, rgs)
            # fv = np.sum((np.mean(tRv) - self.M1)**2) + ((tRv[-1] - tRv[0]) - self.data_width)**2
            fv = np.sum((tRv - t_rgs)**2)
            if debug:
                ret = self.plot_exclusion_state(x, y, poresize, rgs, p[0], t_rgs, tRv, fv, title=title)
                if not ret:
                    self.abort = True
            return fv

        fv_list = []
        res_list = []
        for N in NUM_ENTRIES_LIST:
            self.N = N
            if debug:
                fv = prepare_objective([self.x0], debug=True, title="before minimize with N=%d" % N)
                if self.abort:
                    return
            # bounds = [(-100, 100)]
            bounds = None
            res = minimize(prepare_objective, [self.x0], bounds=bounds, method="Nelder-Mead")
            if debug:
                fv = prepare_objective([res.x[0]], debug=True, title="after minimize with N=%d" % N)
                if self.abort:
                    return

            if res.success:
                self.logger.info("x0 = %f" % res.x[0])
            else:
                self.logger.error("x0 fitting failed")
                return

            if fit_optimize:
                x0 = res.x[0]
                res = self.prepare_fit_optimize(x, y, N, x0, rgs, debug=debug)
                if res is None:
                    return

            fv = res.fun
            self.logger.info("fv = %.3g" % fv)
            fv_list.append(fv)
            res_list.append(res)

        j = np.argmin(fv_list)
        res = res_list[j]
        min_fv = res.fun
        N = NUM_ENTRIES_LIST[j]
        self.logger.info("min_fv=%.3g, x0=%.3g for N=%d", min_fv, res.x[0], N)
        return min_fv, res.x[0], N

    def plot_exclusion_state(self, x, y, poresize, rgs, x0, t_rgs, tRs, fv, title):
        print("rgs=", rgs)
        print("t_rgs=", t_rgs)
        print("tRs=", tRs)
        m = self.me + self.mp
        rv = np.linspace(20, poresize, 100)
        plot_params = [self.N, self.T, x0, m, poresize].copy()
        slider_specs = [    ("N", 0, 6000, plot_params[0]),
                            ("T", 0, 3, plot_params[1]),
                            ("x0",  -500, 500, plot_params[2]),
                            ("m", 0, 4, plot_params[3]),
                            ("poresize", 0, 500, plot_params[4]),
                            ]
        with plt.Dp():
            fig, ax = plt.subplots(figsize=(15,5))
            if title is not None:
                ax.set_title(title + "; fv=%.3g" % fv, fontsize=20)
            ax.plot(x, y)
            ax.axvline(x0, color="red")
            axt = ax.twinx()
            axt.grid(False)
            excl_curve, = axt.plot(compute_tRv(*plot_params, rv), rv, color="yellow")
            axt.plot(t_rgs, rgs, "o", color="red")
            axt.plot(tRs, rgs, "o", color="green")

            dp = plt.get_dp()
            def slider_update(k, val):
                # print([k], "slider_update", val)
                plot_params[k] = val
                trv = compute_tRv(*plot_params, rv)
                excl_curve.set_data(trv, rv)
                dp.draw()

            slider_axes = []
            sliders = []
            for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
                ax_ = fig.add_axes([0.75, 0.8 - 0.08*k, 0.2, 0.03])
                slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
                slider.on_changed(lambda val, k_=k: slider_update(k_, val))
                slider_axes.append(ax_)
                sliders.append(slider)

            fig.tight_layout()
            fig.subplots_adjust(right=0.65)
            ret = plt.show()
        return ret

    def prepare_fit_optimize(self, x, y, N, x0, rgs, debug=False):

        def prepare_fit_objective(p, debug=False, title=None):
            cy_list = []
            for scale, rg in zip(p, rgs):
                cy = self.compute_curve(x, x0, scale, rg)
                cy_list.append(cy)
            ty = np.sum(cy_list, axis=0)
            fv = np.sum((ty - y)**2)
            if debug:
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.plot(x, y)
                    for cy in cy_list:
                        ax.plot(x, cy, ":")
                    ax.plot(x, ty, ":", color="red")
                    fig.tight_layout()
                    ret = plt.show()
                if not ret:
                    self.abort = True
            return fv

        num_components = len(rgs)
        init_scales = np.ones(num_components)*self.area/num_components
        if debug:
            fv = prepare_fit_objective(init_scales, debug=True, title="before fit minimize with N=%d" % N)
            if self.abort:
                return
        bounds = [(0, self.area*2)]*num_components
        res = minimize(prepare_fit_objective, init_scales, method="Nelder-Mead", bounds=bounds)
        if debug:
            fv = prepare_fit_objective(res.x, debug=True, title="after fit minimize with N=%d" % N)
            if self.abort:
                return

        return res

    def estimate_mu_sigma_from_poresize(self, poresize, debug=False):
        self.sigma = DEFAULT_SIGMA
        self.mu = np.log(poresize) + self.sigma**2      # note that: poresize = mode = exp(mu - sigma**2) 
        self.logger.info("estimated mu and sigma as %.3g, %.3g from poresize(%.3g)", self.mu, self.sigma, poresize)

    def estimate_optimal_ksec(self, x, y, rg_recs, debug=False):
        rgs = rg_recs[:,1]
        self.x = x
        self.y = y

        def ksec_objective(p, debug=False, title="ksec_objective"):
            T = p[0]
            x0 = p[1]
            cy_list = []
            for rg, scale in zip(rgs, p[2:]):
                cy = lognormal_pore_func(x, scale, self.N, T, self.me, self.mp, self.mu, self.sigma, rg, x0)
                cy_list.append(cy)
            ty = np.sum(cy_list, axis=0)
            fv = np.sum((ty - y)**2)
            if debug:
                print("T=", T)
                lnp_params = np.concatenate([self.get_column_params(T=T, x0=x0), p[2:]])
                self.plot_state(title, lnp_params)
            return fv

        num_components = len(rgs)
        init_scale = self.area/num_components
        init_params = [self.T, self.x0] + [init_scale]*num_components
        if debug:
            ksec_objective(init_params, debug=True, title="before minimize ksec_objective")
            if self.abort:
                return False
        t0 = time.time()
        bounds = [(0.01, 10), (-1000, 1000)] + [(0, 10)]*num_components
        ret = minimize(ksec_objective, init_params, bounds=bounds, method="Nelder-Mead")
        t = time.time() - t0
        self.logger.info("optimize ksec took %.3g seconds with %d iterations", t, ret.nit)
        if debug:
            ksec_objective(ret.x, debug=True, title="after minimize ksec_objective")
            if self.abort:
                return False
        self.T = ret.x[0]
        self.x0 = ret.x[1]
        self.init_scales = ret.x[2:]    
        self.logger.info("T=%.3g, x0=%.3g, init_scales=%s", self.T, self.x0, str([round(v,3) for v in self.init_scales]))
        return True

    def compute_curve(self, x, x0, scale, rg):
        return lognormal_pore_func(x, scale, self.N, self.T, self.me, self.mp, self.mu, self.sigma, rg, x0)

    def update_column_params(self, params):
        self.N = params[0]
        self.T = params[1]
        self.x0 = params[2]
        self.me = params[3]
        self.mp = params[4]
        self.mu = params[5]
        self.sigma = params[6]

    def plot_state(self, title, lnp_params, save_fig_as=None):
        from molass_legacy.Models.Stochastic.LnporeUtils import plot_lognormal_fitting_state
        ret = plot_lognormal_fitting_state(self.x, self.y, lnp_params, self.rgs, title=title, save_fig_as=save_fig_as)
        if not ret:
            self.abort = True

    def scale_objective_pdf(self, p, debug=False, title=None):
        x = self.x
        y = self.y
        x0 = p[-1]
        cy_list = []
        for rg, scale in zip(self.rgs, p[:-1]):
            cy = lognormal_pore_func(x, scale, self.N, self.T, self.me, self.mp, self.mu, self.sigma, rg, x0)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)

        if debug:
            lnp_params = np.concatenate([self.get_column_params(x0=x0), p[:-1]])
            self.plot_state(title, lnp_params)

        if self.evaluate_area_props:
            props = compute_area_props(cy_list)
            return np.sum((ty - y)**2) + np.sum((props - self.area_props)**2)
        else:
            return np.sum((ty - y)**2)

    def advanced_objective_pdf(self, p, debug=False, title=None, save_fig_as=None):
        x = self.x
        y = self.y
        x0 = p[0]
        mu_ = p[1]
        sigma_ = p[2]
        scales = p[3:]
        cy_list = []
        for rg, scale in zip(self.rgs, scales):
            cy = lognormal_pore_func(x, scale, self.N, self.T, self.me, self.mp, mu_, sigma_, rg, x0)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        if debug:
            print("x0=", x0)
            print("mu, sigma=", mu_, sigma_)
            lnp_params = np.concatenate([self.get_column_params(x0=x0, mu=mu_, sigma=sigma_), scales])
            self.plot_state(title, lnp_params, save_fig_as=save_fig_as)
        return np.sum((y - ty)**2)

    def estimate_psd_with_pdf(self, x, y, rg_recs, area_props, debug=False):
        rgs = rg_recs[:,1]       
        self.logger.info("estimate_psd_with_pdf with x0=%.3g, poresize=%.3g", self.x0, self.poresize)
        self.abort = False
        self.x = x
        self.y = y
        self.rgs = rgs
        self.area_props = area_props

        method = 'Nelder-Mead'
        # method = None

        num_components = len(rgs)
        if self.init_scales is None:
            init_scale = self.area/num_components
            init_params = np.concatenate([np.ones(num_components)*init_scale, [self.x0]])
            if debug:
                self.scale_objective_pdf(init_params, debug=True, title="before scale")
                if self.abort:
                    return

            t0 = time.time()
            min_scale = init_scale*0.2
            scale_bounds = [(min_scale, None)]*(len(init_params)-1) + [(None, None)]
            res = minimize(self.scale_objective_pdf, init_params, method=method, bounds=scale_bounds)
            t = time.time() - t0
            self.logger.info("scale fitting with PDF took : %.3g seconds with %d iterations", t, res.nit)
            if debug:
                print("ret.x=", res.x)
                self.scale_objective_pdf(res.x, debug=True, title="after scale")
                if self.abort:
                    return
                
            init_scales = res.x[:-1]
            init_x0 = res.x[-1]
        else:
            init_scales = self.init_scales
            init_x0 = self.x0

        init_params = np.concatenate([[init_x0, self.mu, self.sigma], init_scales])
        bounds = [(init_x0-50, init_x0+50), (1, 10), (SIGMA_LOWER_BOUND, 1)] + [(0, 10)]*num_components
        if debug:
            self.advanced_objective_pdf(init_params, debug=True, title="before minimize")
            if self.abort:
                return
        t0 = time.time()
        res = minimize(self.advanced_objective_pdf, init_params, method=method, bounds=bounds)
        t = time.time() - t0
        self.logger.info("advanced optimization with PDF took : %.3g seconds with %d iterations", t, res.nit)
        if debug:
            self.advanced_objective_pdf(res.x, debug=True, title="after minimize")
            if self.abort:
                return
        return self.get_whole_params(res)

    def get_whole_params(self, res):
        x0, mu, sigma = res.x[0:3]
        return np.array([self.N, self.T, x0, self.me, self.mp, mu, sigma] + list(res.x[3:]))

    def get_objective_params(self, whole_params):
        return np.concatenate([whole_params[[2,5,6]], whole_params[7:]])

    def compute_residual_ratio(self, x, y, rgs, params, debug=False):
        return compute_residual_ratio_impl(x, y, rgs, params, debug=debug)

    def estimate_psd_with_cf(self, x, y, rg_recs):
        from SecTheory.SecPDF import BidirectFft
        self.logger.info("estimate_psd_with_cf")
        abort = False

        rgs = rg_recs[:,1] 

        def plot_state_from_cf(title, cz_list, tz, mu, sigma):
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
                fig.suptitle(title)
                ax1.plot(x, y)
                for cz in cz_list:
                    cy = bifft.compute_y(cz)
                    ax1.plot(x, cy, ":")
                ty = bifft.compute_y(tz)
                ax1.plot(x, ty, ":", color="red")
                mode = np.exp(mu - sigma**2)
                max_size = int(mode + 500)
                z = np.arange(max_size)
                ax2.plot(z, distr_func(z, mu, sigma), label="pdf")
                fig.tight_layout()
                plt.show()

        bifft = BidirectFft(x)
        z = bifft.compute_z(y)
        w = bifft.get_w()
        def scale_objective(p, debug=False):
            x0 = p[-1]
            cz_list = []
            for rg, scale in zip(rgs, p[:-1]):
                cz = scale*lognormal_pore_cf(w, self.N, self.T, self.me, self.mp, self.mu, self.sigma, rg, x0)
                cz_list.append(cz)
            tz = np.sum(cz_list, axis=0)

            if debug:
                plot_state_from_cf("scale_objective debug", cz_list, tz, self.mu, self.sigma)

            dz = tz - z
            return np.sum(np.real(dz)**2 + np.imag(dz)**2)

        init_params = np.concatenate([np.ones(len(rgs)),[self.x0]])
        scale_objective(init_params, debug=True)
        t0 = time.time()
        res = minimize(scale_objective, init_params, method='Nelder-Mead')
        t = time.time() - t0
        self.logger.info("scale fitting with CF took : %s seconds with %d iterations", t, res.nit)
        scale_objective(res.x, debug=True)
        init_scales = res.x[:-1]
        init_x0 = res.x[-1]

        def advanced_objective(p, debug=False, title=None):
            N = p[0]
            T = p[1]
            x0 = p[2]
            mu_ = p[3]
            sigma_ = p[4]
            scales = p[5:]
            cz_list = []
            for rg, scale in zip(rgs, scales):
                cz = scale*lognormal_pore_cf(w, self.N, self.T, self.me, self.mp, mu_, sigma_, rg, x0)
                cz_list.append(cz)
            tz = np.sum(cz_list, axis=0)

            if debug:
                print("N, T, x0=", N, T, x0)
                print("mu, sigma=", mu_, sigma_)
                plot_state_from_cf(title, cz_list, tz, mu_, sigma_)

            dz = tz - z
            return np.sum(np.real(dz)**2 + np.imag(dz)**2)

        init_params = np.concatenate([[self.N, self.T, init_x0, self.mu, self.sigma], init_scales])
        bounds = [(0, 5000), (0, 10), (-100, 100)] + [(0, 100)]*len(rgs) + [(1, 10), (0.1, 1)]
        advanced_objective(init_params, debug=True, title="before minimize")
        res = minimize(advanced_objective, init_params, method='Nelder-Mead', bounds=bounds)
        self.logger.info("advanced optimization with CF took : %s seconds with %d iterations", t, res.nit)
        advanced_objective(res.x, debug=True, title="after minimize")
        return res.x

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

def estimate_lognormal_psd_impl(advanced_frame, **kwargs):
    with_cf = kwargs.pop('with_cf', False)
    devel = kwargs.pop('devel',True)
    print("estimate_lognormal_psd_impl: with_cf=", with_cf)

    editor = advanced_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    print("estimate_lognormal_psd_impl", model.get_name(), model.__class__)
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    D, E, qv, ecurve = editor.sd.get_xr_data_separate_ly()
    peak_region = ecurve.get_peak_region(sigma_scale=5)
    print("peak_region=", peak_region)

    def plot_cy_list(cy_list, title, test_y=None):
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
            if test_y is not None:
                ax.plot(x, test_y, ":", color="cyan", label="test curve")
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
    column = LognormalPoreColumn()
    last_rgs = None
    for k in range(5):
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

        x0 = 300
        scale = 3
        print("rg_list=", rg_list)
        column.prepare_for_rgs(fx, y, rg_list)
        rgs = column.cover_missing_data(rg_list, last_rgs=last_rgs)
        tRs = None
        area_props = compute_area_props(cy_list)

        if with_cf:
            pore_params = column.estimate_psd_with_cf(fx, y, rgs)
        else:
            pore_params = column.estimate_psd_with_pdf(fx, y, rgs, area_props)
        break
        temp_cy_list = column.compute_cy_list(fx, rgs, pore_params)
        ty = np.sum(temp_cy_list, axis=0)
        msd = np.sum((y - ty)**2)
        print("msd=", msd)
        break
        if min_msd is None or msd < min_msd:
            cy_list = temp_cy_list
            min_msd = msd
            column.update_column_params(pore_params)
            plot_cy_list(cy_list, "%d-th estimate" % k)
            last_rgs = rgs
            init_scales = pore_params[9:]
        else:
            break