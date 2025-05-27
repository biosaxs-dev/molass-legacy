"""
    SecTheory.SecEstimator.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from molass_legacy._MOLASS.SerialSettings import get_setting
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ElutionCurveModels import egh

NUM_SEC_PARAMS = 6      # used in other modules

def guess_initial_secparams(init_xr_params, rgs):
    Npc = get_setting('num_plates_pc')
    k = np.argmax(init_xr_params[:,0])
    tR = init_xr_params[k,1]
    s, t = init_xr_params[k,[2,3]]
    sigma = np.sqrt(s**2 + t**2)
    """
    tR - tI = sqrt(Np)*sigma
    tI = tR - sqrt(Np)*sigma
    tR = t0 + P*(1 - rho)**m
    t0 = tR - P*(1 - rho)**m
    """
    tI = tR - np.sqrt(Npc)*sigma
    rp = get_setting('poresize')
    print("----------------- guess_initial_secparams: rp=", rp)
    rg = rgs[k]
    rho = rg/rp
    P = 2000
    m = 2
    t0 = tR - P*(1 - rho)**m
    return Npc, rp, tI, t0, P, m


NUM_PARAMS = 4

class SecEstimator:
    def __init__(self, Npc, rp, tI, t0, P, m):
        m = min(2, m)   # force
        self.init_sec_params = Npc, rp, tI, t0, P, m
        print("init: Npc, rp, tI, t0, P, m =", self.init_sec_params)

    def fit_to_decomposition(self, x, y, init_xr_params, rgs, debug=False):
        """
            tR = t0 + P*(1 - rho)**m
            Tr = tR - tI
        """

        n = len(rgs)
        xr_params_shape = (n,4)
        zeros = np.zeros(n)
        Npc, rp = self.init_sec_params[0:2]
        rN = np.sqrt(Npc)
        rhos = rgs/rp
        rhos[rhos > 1] = 1

        def objective(p, debug=False, sub_title=""):
            tI, t0, P, m = p[:NUM_PARAMS]
            xr_params = p[NUM_PARAMS:].reshape(xr_params_shape)

            tR = t0 + P*(1 - rhos)**m
            sigma = (tR - tI)/rN
            s = np.sqrt(xr_params[:,2]**2 + xr_params[:,3]**2)
            scale = np.sum((xr_params[:,0] - init_xr_params[:,0])**2)
            # bound = min(0, tI + 1000)**2 + np.sum(np.min([zeros, sigma], axis=0)**2) + min(0, P - 1000)**2 + max(0, P - 10000)**2
            if debug:
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("objective: " + sub_title)
                    ax.plot(x, y, color="orange")

                    cy_list = []
                    for h_, mu, s_, t_ in xr_params:
                        cy = egh(x, h_, mu, s_, t_)
                        cy_list.append(cy)
                        ax.plot(x, cy, ":")

                    ty = np.sum(cy_list, axis=0)
                    ax.plot(x, ty, ":", color="red")
                    fig.tight_layout()
                    plt.show()

            return scale + np.sum((xr_params[:,1] - tR)**2 + (s - sigma)**2)

        start = 6 - NUM_PARAMS
        init_params = np.concatenate([self.init_sec_params[start:], init_xr_params.flatten()])

        if debug:
            objective(init_params, debug=True, sub_title="before")

        Npc, rp, tI, t0, P, m = self.init_sec_params

        bounds = [
                    (tI - 500, tI + 200),
                    (t0 - 500, t0 + 200),
                    (1000, 10000),
                    (1, 3),
                ]

        for h_, mu, s_, t_ in init_xr_params:
            bounds.append((h_*0.8, h_*1.2))
            ts_ = np.sqrt(s_**2 + t_**2)
            dx = ts_*0.1
            bounds.append((mu - dx, mu + dx))
            bounds.append((s_ - dx, s_ + dx))
            bounds.append((t_ - dx, t_ + dx))
        # ret = minimize(objective, init_params, method='Nelder-Mead', bounds=bounds)
        ret = basinhopping(objective, init_params, minimizer_kwargs=dict(method='Nelder-Mead', bounds=bounds))

        if debug:
            objective(ret.x, debug=True, sub_title="after")

        self.ret_sec_params = np.concatenate([[Npc, rp], ret.x[:NUM_PARAMS]])

        fit_xr_params = ret.x[NUM_PARAMS:].reshape(xr_params_shape)
        # fit_xr_params[:,0] = init_xr_params[:,0]

        if debug:
            print("ret_sec_params=", self.ret_sec_params)
            print("fit_xr_params=", fit_xr_params)

            init_cy_list = []
            for h, tr, s, t in init_xr_params:
                cy = egh(x, h,tr, s, t)
                init_cy_list.append(cy)
            init_ty = np.sum(init_cy_list, axis=0)

            fit_cy_list = []
            for h, tr, s, t in fit_xr_params:
                cy = egh(x, h, tr, s, t)
                fit_cy_list.append(cy)
            fit_ty = np.sum(fit_cy_list, axis=0)

            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("fit_to_decomposition")
                ax.plot(x, y, color="orange")
                for cy in fit_cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, init_ty, color="yellow")
                ax.plot(x, fit_ty, ":", color="red")
                fig.tight_layout()
                plt.show()

    def get_params(self):
        return self.ret_sec_params
