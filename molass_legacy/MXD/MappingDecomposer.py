# coding: utf-8
"""
    MXD.MappingDecomposer.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from molass_legacy.Peaks.RobustPeaks import RobustPeaks
from molass_legacy.Peaks.ElutionModels import egh
from Prob.GaussianMixture import hist_to_source
from Prob.EghMixture import EghMixture
import OurStatsModels as sm

APPLY_PARAM_CONSTRAINTS = True
USE_WLS = True

class CustomMixture(EghMixture):
    def __init__(self, K):
        self.K = K
        self.anim_data = False

    def initialize(self, X, init_params):
        self.N = X.shape[0]
        self.X = X
        self.X_  = X.flatten()
        sorted_params = np.array(sorted(init_params, key=lambda x:x[1]))
        w = np.sum(sorted_params[:,0])
        self.pi = sorted_params[:,0]/w
        self.tR = sorted_params[:,1]
        self.sigma = sorted_params[:,2]
        self.tau = sorted_params[:,3]
        if APPLY_PARAM_CONSTRAINTS:
            self.min_sigma = self.sigma*0.5
            self.max_sigma = self.sigma*1.5
            self.min_tR = np.max([np.zeros(self.K), self.tR - self.sigma], axis=0)
            self.max_tR = self.tR + self.sigma

class MappingDecomposer:
    def __init__(self, uv_rp, xr_rp, num_components=None):
        self.uv_rp = uv_rp
        self.xr_rp = xr_rp
        self.uv_x = uv_rp.x
        self.uv_y = uv_rp.y
        self.xr_x = xr_rp.x
        self.xr_y = xr_rp.y
        self.num_components = num_components
        self.uv_sy = hist_to_source(self.uv_x, self.uv_y)
        self.xr_sy = hist_to_source(self.xr_x, self.xr_y)

    def decompose_separatley(self):
        from Prob.EghMixture import EghMixture
        from Prob.EghMixtureUtils import get_components

        num_components = self.num_components
        x0 = self.uv_x
        y0 = self.uv_y
        sy0 = self.uv_sy
        mm0 = EghMixture(num_components)
        mm0.fit(np.expand_dims(sy0,1))
        cy_list0, ty0 = get_components(x0, y0, mm0)

        x1 = self.xr_x
        y1 = self.xr_y
        sy1 = self.xr_sy
        mm1 = EghMixture(num_components)
        mm1.fit(np.expand_dims(sy1,1))
        cy_list1, ty1 = get_components(x1, y1, mm1)

        plot_mapped_states([
            [x0, y0, cy_list0, ty0],
            [x1, y1, cy_list1, ty1],
            ])

    def decompose(self):
        mapping_params, params_info, w_ = self.guess_initial_params()
        print("mapping_params=", mapping_params)
        x0 = self.uv_x
        y0 = self.uv_y
        x1 = self.xr_x
        y1 = self.xr_y

        if False:
            for i in range(1):
                components_list = []
                mapping_points = []
                for (x, y), params_list in zip([(x0, y0), (x1, y1)], params_info):
                    cy_list = []
                    ty = np.zeros(len(x))
                    for params in params_list:
                        h, tR, sigma, tau = params
                        mapping_points.append((tR-sigma, tR, tR+sigma))
                        cy = egh(x, h, tR, sigma, tau)
                        cy_list.append(cy)
                        ty += ty
                    components_list.append((x, y, cy_list, ty))
                plot_mapped_states(components_list)

        uv_cm = CustomMixture(self.num_components)
        xr_cm = CustomMixture(self.num_components)

        uv_cm.initialize(np.expand_dims(self.uv_sy,1), params_info[0])
        xr_cm.initialize(np.expand_dims(self.xr_sy,1), params_info[1])

        for step in range(201):
            uv_cm._e_step()
            xr_cm._e_step()
            uv_params = uv_cm._m_step(step)
            xr_params = xr_cm._m_step(step)
            mapping_params_, uv_params_, xr_params_ = self.map_adjust_params(mapping_params, uv_params, xr_params, weights=w_)
            components_list = []
            for (x, y), params, pi_list in zip([(x0, y0), (x1, y1)], [uv_params_, xr_params_], [uv_cm.pi, xr_cm.pi]):
                cy_list = []
                ty = np.zeros(len(x))
                for prm, pi in zip(params, pi_list):
                    cy = egh(x, pi, *prm)
                    cy_list.append(cy)
                    ty += cy
                components_list.append((x, y, cy_list, ty))
            if step % 100 == 0:
                plot_mapped_states(components_list, scale=True, step=step, mappinn_info=[mapping_params_, uv_params_, uv_cm.pi, xr_params_, xr_cm.pi])
            uv_cm.update_params(step, uv_params_)
            xr_cm.update_params(step, xr_params_)

    def guess_initial_params(self):
        params_info = []
        mapping_points = []
        for k, rp in enumerate([self.uv_rp, self.xr_rp]):
            params_list = []
            x = rp.x
            y = rp.y
            w = []
            for peak in rp.get_peaks():
                ls, pt, rs = peak
                print([k], ls, pt, rs)
                mapping_points.append(x[peak])
                lsw = x[pt] - x[ls]
                tau = 0
                if self.num_components == 5:
                    point_list = [pt - 2*(pt-ls), ls]
                elif self.num_components in [6, 7]:
                    point_list = [pt - 3*(pt-ls), ls, (ls+pt)//2]
                elif self.num_components in [8]:
                    point_list = [pt - 4*(pt-ls), pt - 3*(pt-ls), ls, (ls+pt)//2]
                else:
                    assert False
                for ls_ in point_list:
                    h = y[ls_]
                    tR = x[ls_]
                    sigma = lsw/3
                    params_list.append((h, tR, sigma, tau))
                    w.append(1)
                rsw = x[rs] - x[pt]
                h = y[pt]
                tR = x[pt]
                sigma = (lsw+rsw)/3.5
                params_list.append((h, tR, sigma, tau))
                w.append(30)
                if self.num_components < 7:
                    point_list = [rs, pt + 2*(rs-pt)]
                elif self.num_components >= 7:
                    point_list = [rs, pt + 2*(rs-pt), pt + int(2.5*(rs-pt))]
                else:
                    assert False

                for rs_ in point_list:
                    h = y[rs_]
                    tR = x[rs_]
                    sigma = rsw/3
                    params_list.append((h, tR, sigma, tau))
                    w.append(1)
            params_info.append(params_list)

        w_ = w if USE_WLS else None
        slope, intercept = linear_regression(mapping_points[1], mapping_points[0])
        return [slope, intercept], params_info, w_

    def map_adjust_params(self, mapping_params, uv_params, xr_params, weights):
        slope, intercept = linear_regression(xr_params[:,0], uv_params[:,0], weights=weights)
        uv_params_ = []
        xr_params_ = []
        for uv_p, xr_p in zip(uv_params, xr_params):
            tR = (uv_p[0] + slope*xr_p[0] + intercept)/2
            sigma = (uv_p[1] + slope*xr_p[1])/2
            tau = (uv_p[2] + slope*xr_p[2])/2
            uv_params_.append((tR, sigma, tau))

            tR = ((uv_p[0] - intercept)/slope + xr_p[0])/2
            sigma = (uv_p[1]/slope + xr_p[1])/2
            tau =  (uv_p[2]/slope + xr_p[2])/2
            xr_params_.append((tR, sigma, tau))
        return [slope, intercept], np.array(uv_params_), np.array(xr_params_)

def linear_regression(x, y, weights=None):
    if weights is None:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
    else:
        X = sm.add_constant(x)
        mod = sm.WLS(y, X, weights=weights)
        res = mod.fit()
        intercept, slope = res.params
    return slope, intercept

def plot_mapped_states(components_list, scale=False, step=None, mappinn_info=None):
    plt.push()
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))

    step_str = "" if step is None else " step %d" % step
    fig.suptitle("UV-Xray Decomposed Mapping Figures (Plan)%s" % step_str, fontsize=30)
    ax1.set_title("UV Elution", fontsize=20)
    ax2.set_title("Xray Elution", fontsize=20)
    ax3.set_title("Mapped Elution", fontsize=20)

    for ax, components, color in zip([ax1, ax2], components_list, ['C0', 'C1']):
        x, y, cy_list, ty = components
        if scale:
            s = np.max(y)/np.max(ty)
        else:
            s = 1

        ax.plot(x, y, color=color)
        ax.plot(x, s*ty, ':', color='red')
        for cy in cy_list:
            ax.plot(x, s*cy, ':')

    if mappinn_info is not None:
        mapping_params, uv_params, uv_pi, xr_params, xr_pi = mappinn_info
        slope, intercept = mapping_params
        x0, y0, cy_list0, ty0 = components_list[0]
        x1, y1, cy_list1, ty1 = components_list[1]
        ax3.plot(x1, y1, color='C1', label='Xray')
        spline = UnivariateSpline(x0, y0, s=0, ext=3)
        x_ = x1*slope + intercept
        y_ = spline(x_)
        scale = np.max(y1)/np.max(y_)
        my = y_*scale
        ax3.plot(x1, my, color='gray', alpha=0.2, label='simply mapped UV')
        smy = make_mapped_uv(x1, my, uv_params, uv_pi, xr_params, xr_pi)
        ax3.plot(x1, smy, ':', color='C0', label='scaling mapped UV')
        ax3.legend(fontsize=16, loc="upper left")

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
    plt.pop()

def make_mapped_uv(x, mapped_y, uv_params, uv_pi, xr_params, xr_pi):
    w = np.zeros(len(mapped_y))
    for uv_p, xr_p, params in zip(uv_pi, xr_pi, xr_params):
        w += egh(x, xr_p/uv_p, *params)
    smy = mapped_y*w
    scale = np.max(mapped_y)/np.max(smy)
    return smy*scale

def spike_demo():
    x0 = np.arange(1000)
    y0 = np.zeros(len(x0))
    for params in [
            [1.0, 401, 50, 5],
            [0.7, 602, 60, 10],
            ]:
        y0 += egh(x0, *params)

    x1 = np.arange(500)
    y1 = np.zeros(len(x1))
    for params in [
            # [0.6, 200, 40, 10],
            [0.6, 200, 30, 5],
            [1.0, 300, 30, 10],
            ]:
        y1 += egh(x1, *params)

    spike_demo_impl(x0, y0, x1, y1, num_components=2)

def spike_demo_impl(x0, y0, x1, y1, num_components=None):
    rb0 = RobustPeaks(x0, y0)
    rb1 = RobustPeaks(x1, y1)
    md = MappingDecomposer(rb0, rb1, num_components=num_components)
    # md.decompose_separatley()
    md.decompose()


def spike_demo_real(in_folder):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_sd()
    uv_curve = sd.get_uv_curve()
    xr_curve = sd.get_xray_curve()
    spike_demo_impl(uv_curve.x, uv_curve.y, xr_curve.x, xr_curve.y, num_components=6)
