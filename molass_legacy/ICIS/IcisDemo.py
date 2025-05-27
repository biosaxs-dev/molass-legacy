"""
    IcisDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.DENSS.DenssUtils import fit_data
from SvdDenoise import get_denoised_data

def get_primary_scurve(qv, D, E, ecurve, width=11):
    j = ecurve.peak_info[ecurve.primary_peak_no][1]
    hw = width//2
    eslice = slice(j-hw,j+hw+1)
    return np.array([qv] + [np.average(M[:,eslice], axis=1) for M in (D, E)]).T

def get_primary_eslice(ecurve):
    lower, middle, upper = ecurve.peak_info[ecurve.primary_peak_no]
    eslice = slice(lower,upper+1), middle - lower
    return eslice

def solve_Bq(iq, aq, bqinit, cinit):

    def objective(p):
        bq = p[:-1]
        c = p[-1]
        return np.sum((aq*c + bq*c**2 - iq)**2)

    pinit = np.concatenate([bqinit, [cinit]])
    ret = minimize(objective, pinit)
    print("fun init=", objective(pinit), "ret.nit=", ret.nit, "ret.fun=", ret.fun)
    return ret.x[:-1], ret.x[-1]

def get_theretical_C(C_):
    cinit = C_[0,:]

    def objective(c):
        C = np.array([c, c**2])
        return np.linalg.norm(C - C_)

    ret = minimize(objective, cinit)
    c = ret.x
    return np.array([c, c**2])

def demo(in_folder, sd):
    print(in_folder)

    D, E, qv, ecurve = sd.get_xr_data_separate_ly()

    Iq = get_primary_scurve(qv, D, E, ecurve)
    qc, ac, ec, dmax = fit_data(*Iq.T)

    eslice, j = get_primary_eslice(ecurve)
    D_ = get_denoised_data(D[:,eslice], rank=2)
    E_ = E[:,eslice]

    i_smp = sd.xray_index
    c = D_[i_smp,:]
    C = np.array([c, c**2])
    Cinv = np.linalg.pinv(C)
    P = D_ @ Cinv
    Dinv = np.linalg.pinv(D_)
    W = Dinv @ P
    Pe = np.sqrt(np.dot(E_**2, W**2))

    Aq = np.array([qv, P[:,0], Pe[:,0]]).T
    qc_, ac_, ec_, dmax_ = fit_data(*Aq.T)

    n = len(qc) - len(qv)
    diff = np.min(qc[n:] - qv)

    print(len(qv), len(qc), len(qc_), diff)

    Aq_ = np.array([qv, ac_[n:], ec_[n:]]).T
    Bq = np.array([qv, P[:,1], Pe[:,1]]).T

    cinit = c[j]
    bq_, cs = solve_Bq(Iq[:,1], Aq_[:,1], Bq[:,1], cinit)

    P_ = np.array([Aq_[:,1], bq_]).T
    Pinv = np.linalg.pinv(P_)
    C_ = Pinv @ D_
    Ct = get_theretical_C(C_)

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
        ax1.set_yscale('log')
        ax1.plot(qv, Iq[:,1])
        ax1.plot(qc, ac)
        ax1.plot(qv, Aq[:,1])
        ax1.plot(qv, Aq_[:,1])

        ax2.plot(qv, Bq[:,1])
        ax2.plot(qv, bq_)

        ax3.plot(c)
        # ax3.plot(C_[0,:])
        ax3.plot(Ct[0,:])

        fig.tight_layout()
        plt.show()
