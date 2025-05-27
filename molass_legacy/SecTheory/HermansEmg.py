"""
    SecTheory.HermansEmg.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, root
from scipy.stats import linregress
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Peaks.ElutionModels import emg, egh
from LPM import get_corrected
from .FoleyDorseyEmg import fd_estimate_initial_params, convert_to_xr_params_fd

ESTIMATE_INJECTION_TIME = True
PENALTY_SCALE = 1e8
NEGTIVE_PENALTY_WEIGHT = 1000   # should be dynalically determined?

def compute_momemts_from_params(mu, sigma, tau):
    X = mu + tau
    Y = sigma**2 + tau**2
    Z = 2*tau**3
    return X, Y, Z

def compute_params_from_momemts(X, Y, Z):
    if np.isscalar(X):
        t = (Z/2)**(1/3)
        m = X - t
        s2 = Y - t**2
        s2_ = max(1, s2)
        s = np.sqrt(s2_)
        return m, s, t
    else:
        # s2_ = np.max([np.ones(len(s2)), s2], axis=0)
        params = np.zeros((len(X),3))
        for k, (x, y, z) in enumerate(zip(X, Y, Z)):
            params[k,:] = compute_params_from_momemts(x, y, z)
        return params

def compute_moments_from_ksec_p_r(t0, Ksec, P, R):
    K = Ksec*P
    X = t0 + K
    Y = 1/30 * K * R
    Z = 1/420 * K * R**2
    return X, Y, Z

def estimate_secparams_and_R(x, y, init_params, hs, trs, rgs, fdX, debug=False):
    """
    optimize secparams and R so that the model curve fits to the data
    """

    def objective(p, return_detail=False, debug=False, title=""):
        sec_params = p[0:-1]
        R = p[-1]
        X = convert_to_xr_params_hermans(hs, rgs, sec_params, R)
        ty = np.zeros(len(x))

        for i, (h, m, s, t) in enumerate(X):
            cy = emg(x, h, m, s, t)
            ty += cy

        if debug:
            ty_ = np.zeros(len(x))
            cy_list = []
            for i, (h, m, s, t) in enumerate(X):
                cy = emg(x, h, m, s, t)
                ty_ += cy
                if debug:
                    cy_list.append(cy)
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(x, y)
                for k, cy in enumerate(cy_list):
                    ax.plot(x, cy, ":", label="component-%d" % k)
                ax.plot(x, ty_, ":", color="red", label="total")
                ax.legend()
                fig.tight_layout()
                plt.show()

        fv1 = np.log10(np.sum((ty - y)**2))
        fv2 = np.log10(np.linalg.norm(X - fdX)**2)
        if return_detail:
            return fv1, fv2
        else:
            return  fv1 + fv2

    minP = trs[-1] - trs[0]
    R = init_params[-1]
    poresize_bounds = get_setting("poresize_bounds")
    bounds = [(-100, trs[0] - 10), (minP, minP*5), poresize_bounds, (1, 3), (R/2, R*2)]

    if debug:
        print("init_params=", init_params)
        print("bounds=", bounds)
        print("fv1, fv2=", objective(init_params, return_detail=True))
        objective(init_params, debug=True, title="estimate_secparams_and_R: init")

    ret = minimize(objective, init_params, bounds=bounds)

    if debug:
        print("ret.x=", ret.x)
        print("fv1, fv2=", objective(ret.x, return_detail=True))
        objective(ret.x, debug=True, title="estimate_secparams_and_R: optimized")

    return ret.x

def estimate_initial_params(ecurve, init_xr_params, init_rgs, seccol_params, D, debug=False):

    temp_xr_params, fd_seccol_params = fd_estimate_initial_params(ecurve, init_xr_params, seccol_params, debug=debug)
    Ti, Np = fd_seccol_params[-2:]
    fdX = convert_to_xr_params_fd(temp_xr_params, Ti, Np)

    x = ecurve.x
    y_ = ecurve.y

    # use corrected y
    y = get_corrected(y_)

    mu_ = np.average(init_xr_params[:,1])
    hs = init_xr_params[:,0]
    trs = init_xr_params[:,1]       # egh tr's

    t0, P, rp, m = seccol_params

    rhos =init_rgs/rp
    rhos[rhos > 1] = 1          # rhos[0] > 1 as in 20191114_4
    Ksec = (1 - rhos)**m

    slope, intercept = linregress(Ksec, trs)[0:2]

    t0 = intercept
    P = slope

    if debug:
        print("init_rgs=", init_rgs)
        print("rhos=", rhos)
        print("trs=", trs)
        print("t0, P=", t0, P)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("linregress")
            ax.plot(Ksec, trs, "o")
            ax.plot(Ksec, t0 + Ksec*P, "o")
            fig.tight_layout()
            plt.show()

    """
        M2 = 1/30*Ksec*P*R
        R = 30*M2/(Ksec*P)
    """
    sigma_ = np.average(init_xr_params[:,2])
    Ksec_ = np.average(Ksec)
    M2 = sigma_**2
    R = 30*M2/(Ksec_*P)

    init_params = [t0, P, rp, m, R]

    ret_params = estimate_secparams_and_R(x, y, init_params, hs, trs, init_rgs, fdX, debug=debug)
    secparams_ = ret_params[0:-1]
    R_ = ret_params[-1]

    if debug:
        t0_, P_, rp_, m = secparams_
        print("t0_, P_, rp_, m=", t0_, P_, rp_, m)
        rho_ = init_rgs/rp_
        ksec_ = (1 - rho_)**m
        tr = t0_ + ksec_*P_
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("SEC Parameters")
            ax.plot(x, y)
            for params  in init_xr_params:
                ax.plot(x, emg(x, *params), ":")
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)

            ax.plot([t0_, t0_], [ymin, ymax], ":", color="red")
            tp = t0_ + P_
            ax.plot([tp, tp], [ymin, ymax], ":", color="green")

            axt = ax.twinx()
            axt.grid(False)

            sy = np.linspace(0, rp_ - 0.1, 100)
            rho_ = sy/rp_
            sx = t0_ + P_*(1 - rho_)**m

            axt.plot(sx, sy, ":", color="blue")
            axt.plot(tr, init_rgs, "o", color="yellow")
            fig.tight_layout()
            plt.show()

    n = len(init_xr_params)
    p_zeros = np.zeros(D.shape[0])

    def objective(p, return_detail=False, debug=False, title=None):
        xr_params = convert_to_xr_params_hermans(p[0:n], init_rgs, p[n:-3], p[-1])
        ty = np.zeros(len(x))
        cy_list = []

        try:
            for h, m, s, t in xr_params:
                cy = emg(x, h, m, s, t)
                ty += cy
                cy_list.append(cy)

            C = np.array(cy_list)
            P = D @ np.linalg.pinv(C)
        except:
            """
            at ty += cy
            ValueError: operands could not be broadcast together with shapes (901,) (0,) (901,) 
            or numpy.linalg.LinAlgError: SVD did not converge
                where xr_params = 
                    [[nan nan  1. nan]
                     [nan nan  1. nan]
                     [nan nan  1. nan]]
            """
            if False:
                print("xr_params=", xr_params)
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("except")
                    ax.plot(x, y_, label="data")
                    ax.plot(x, y, label="corrected")
                    for k, cy in enumerate(cy_list):
                        ax.plot(x, cy, ":", label="component-%d" % k)
                    ax.plot(x, ty, ":", color="red", label="total")
                    fig.tight_layout()
                    plt.show()
            return PENALTY_SCALE

        negative_penalty = 0
        for p in P.T:
            negative_penalty += np.mean(np.min([p_zeros, p], axis=0)**2)

        if debug:
            with plt.Dp():
                fig, ax = plt.subplots()
                if title is None:
                    ax.set_title("objective")
                else:
                    ax.set_title(title)
                ax.plot(x, y)
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                ax.plot(x, ty, ":", color="red")
                fig.tight_layout()
                plt.show()

        fv1 = np.log10(np.sum((ty - y)**2))
        fv2 = np.log10(np.linalg.norm(xr_params - fdX)**2)
        if return_detail:
            return fv1, fv2
        else:
            return fv1 + fv2 + NEGTIVE_PENALTY_WEIGHT*negative_penalty

    Np = get_setting("num_plates_pc")
    if ESTIMATE_INJECTION_TIME:
        """
            N = (m - Ti)**2/s**2
            sqrt(N) = (m - Ti)/s
            sqrt(N)*s = m - Ti
            Ti = m - sqrt(N)*s
        """
        i = np.argmax(init_xr_params[:,0])
        m_, s_ = init_xr_params[i,1:3]
        Ti = m_ - np.sqrt(Np)*s_
    else:
        from .ColumnConstants import INJECTION_TIME
        Ti = INJECTION_TIME

    init_values = np.concatenate([init_xr_params[:,0], secparams_, [Ti, Np, R_]])

    if debug:
        print("fv1, fv2=", objective(init_values, return_detail=True))
        objective(init_values, debug=True, title="init_values")

    bounds = [(0, None)]*len(init_values)
    ret = minimize(objective, init_values, bounds=bounds)
    temp_xr_params, seccol_params, R = ret.x[0:n], ret.x[n:n+4], ret.x[-1]

    if debug:
        print("ret.fun=", ret.fun)
        print("temp_xr_params=", temp_xr_params)
        print("seccol_params=", seccol_params)
        print("fv1, fv2=", objective(ret.x, return_detail=True))
        objective(ret.x, debug=True, title="ret.x")

    return temp_xr_params, seccol_params, R

def convert_to_xr_params_hermans(xrh_params, rgs, sec_params, R):
    t0, P, rp, m = sec_params
    rho = rgs/rp
    rho[rho > 1] = 1
    Ksec = (1 - rho)**m
    M = compute_moments_from_ksec_p_r(t0, Ksec, P, R)
    X = compute_params_from_momemts(*M)
    ret_params = np.zeros((len(xrh_params), 4))
    ret_params[:,0] = xrh_params
    ret_params[:,1:] = X
    return ret_params

def spike_demo():

    x = np.linspace(0, 500, 501)

    t0 = 50
    mu = 200
    sigma = 30
    tau = 10
    y = emg(x, 1, mu, sigma, tau)

    M = compute_momemts_from_params(mu, sigma, tau)
    m, s, t = compute_params_from_momemts(*M)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, emg(x, 1, m, s, t), ":")
        fig.tight_layout()
        plt.show()

    P = 300
    hs = np.array([1])
    trs = np.array([mu + tau])
    Ksec = 0.3
    M2 = sigma**2 + tau**2
    R = np.sqrt(30*M2/(Ksec*P))
    print("t0, P, R=", t0, P, R)

    t0_, P_, R_ = estimate_P_R(x, y, t0, P, R, hs, trs)
    print("t0_, P_, R=", t0_, P_, R_)
    K_ = (trs - t0_)/P_
    M = compute_moments_from_ksec_p_r(t0_, K_, P_, R_)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.plot(x, y)
        for i, (m_, s_, t_) in enumerate(compute_params_from_momemts(*M)):
            ax.plot(x, emg(x, hs[i], m_, s_, t_), ":")
        fig.tight_layout()
        plt.show()
