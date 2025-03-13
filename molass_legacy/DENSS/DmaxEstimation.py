# coding: utf-8
"""
    DmaxEstimation.py

    Copyright (c) 2020-2022, SAXS Team, KEK-PF
"""
import numpy as np
import DebugPlot as plt
from .saxstats.saxstats import Sasrec, clean_up_data, calc_rg_I0_by_guinier, calc_rg_by_guinier_peak, filter_P
from .DenssUtils import fit_data_impl

### DENSS.saxstats.saxstats.py copy & modify BEGIN ###
def estimate_dmax(Iq,dmax=None,clean_up=True):
    """Attempt to roughly estimate Dmax directly from data."""
    #first, clean up the data
    if clean_up:
        Iq = clean_up_data(Iq)
    q = Iq[:,0]
    I = Iq[:,1]
    if dmax is None:
        #first, estimate a very rough rg from the first 20 data points
        nmax = 20
        try:
            rg, I0 = calc_rg_I0_by_guinier(Iq,ne=nmax)
        except:
            rg = calc_rg_by_guinier_peak(Iq,exp=1,ne=100)
        #next, dmax is roughly 3.5*rg for most particles
        #so calculate P(r) using a larger dmax, say twice as large, so 7*rg
        D = 7*rg
    else:
        #allow user to give an initial estimate of Dmax
        #multiply by 2 to allow for enough large r values
        D = 2*dmax
    #create a calculated q range for Sasrec for low q out to q=0
    qmin = np.min(q)
    dq = (q.max()-q.min())/(q.size-1)
    nq = int(qmin/dq)
    qc = np.concatenate(([0.0],np.arange(nq)*dq+(qmin-nq*dq),q))
    #run Sasrec to perform IFT
    sasrec = Sasrec(Iq, D, qc=None, alpha=0.0, extrapolate=False)
    #now filter the P(r) curve for estimating Dmax better
    r, Pfilt, sigrfilt = filter_P(sasrec.r, sasrec.P, sasrec.Perr, qmax=Iq[:,0].max())
    #estimate D as the first position where P becomes less than 0.01*P.max(), after P.max()
    Pargmax = Pfilt.argmax()
    #catch cases where the P(r) plot goes largely negative at large r values,
    #as this indicates repulsion. Set the new Pargmax, which is really just an
    #identifier for where to begin searching for Dmax, to be any P value whose
    #absolute value is greater than at least 10% of Pfilt.max. The large 10% is to 
    #avoid issues with oscillations in P(r).
    above_idx = np.where((np.abs(Pfilt)>0.1*Pfilt.max())&(r>r[Pargmax]))
    Pargmax = np.max(above_idx)
    near_zero_idx = np.where((np.abs(Pfilt[Pargmax:])<(0.001*Pfilt.max())))[0]
    near_zero_idx += Pargmax
    D_idx = near_zero_idx[0]
    D = r[D_idx]
    sasrec.D = D
    sasrec.update()
    return D, sasrec, [r, Pfilt, Pargmax, D_idx]
### DENSS.saxstats.saxstats.py copy & modify END ###

def plot_input(ax, data, in_file):
    q = data[:,0]
    a = data[:,1]
    e = data[:,2]

    sasrec, work_info = fit_data_impl(q, a, e, in_file)
    qc = sasrec.qc
    ac = sasrec.Ic
    ec = work_info.Icerr

    ax.set_yscale('log')
    ax.set_xlabel('q', fontsize=16)
    ax.set_ylabel('log(I)', fontsize=16)

    ax.plot(q, a, color='C1', label="input data")
    # ax1.plot(qc, ac, color='C2', label="fitted data")

    ax.legend(fontsize=16)

def illustrate_dmax(ax, data):
    D_, sasrec_, info = estimate_dmax(data)
    r = sasrec_.r
    P = sasrec_.P
    r_, Pfilt, Pargmax, D_idx = info

    ax.set_xlabel('r', fontsize=16)
    ax.set_ylabel('P', fontsize=16)

    ax.plot(r, P, color='C1', label="P(r) from input data")
    ax.plot(r_, Pfilt, color='C2', label="filtered P(r)")
    ax.plot(r_[D_idx], Pfilt[D_idx], 'o', color='red', label='estimated Dmax')

    ax.legend(fontsize=16)

def demo(in_file):
    print(in_file)
    data = np.loadtxt(in_file)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16,7))
    fig.suptitle("Denss Dmax Estimation Illustrated", fontsize=30)

    plot_input(ax1, data, in_file)
    illustrate_dmax(ax2, data)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.show()
