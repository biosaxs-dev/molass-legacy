import numpy as np
# import matplotlib.pyplot as plt
import seaborn
seaborn.set()

# see https://github.com/sartorius-research/STLC for installation
from stlc import grm

import os
import sys
lib_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(lib_dir)
kek_lib = os.path.join(lib_dir, "KekLib")
sys.path.append(kek_lib)
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.ElutionModels import emg
from SecTheory.BasicModels import fit_single_pore, robust_single_pore_pdf

def main():
    cm = 1e-2
    minute = 60
    zl = 1.7 * cm  #[cm]
    rp = 0.004 * cm  #[cm]
    ec = 0.4
    ep = 0.333
    kf = 0.01 * cm  #[cm]
    Dax = 0.002 * cm**2 / minute  #[cm^2 min^-1]
    Dp = 3.003 * 1e-6 * cm**2 / minute  #[cm^2 min^-1]
    Ds = 0.  #[cm^2 min^-1]
    u = 0.6 * cm / minute  #[cm min^-1]
    ka = 2.5  #[s^-1]
    kd = 1  #[s^-1]
    tinj = 20 * minute  #[min]
    cinj = 1  #[mol /m^-3]
    tmax = 100 * minute  #[min]
    qm = 1.
    nr = 10
    nz = 20
    dt = 1

    def step(t: float, tinj: float) -> float:
        return float(t <= tinj)

    parameters0 = grm.ModelParameters(c0=cinj,
                                          Dax=Dax,
                                          Dp=Dp,
                                          Ds=Ds,
                                          ka=ka,
                                          kd=kd,
                                          kf=kf,
                                          qm=qm,
                                          ip=lambda t: step(t, tinj))

    model = grm.GeneralRateModel(u=u,
                                         ep=ep,
                                         ec=ec,
                                         zl=zl,
                                         rp=rp,
                                         nz=nz,
                                         nr=nr,
                                         component_parameters=[parameters0])
    sol = grm.solve(model, tmax, dt)

    x = sol.t / minute
    y = sol.c[0, -1, :]

    print(len(sol.t), sol.t[0], sol.t[-1])

    m1 = np.sum(x*y)/np.sum(y)
    print("m1=", m1)
    m2 = np.sum((x - m1)**2 * y)/np.sum(y)
    print("m2=", m2)
    m3 = np.sum((x - m1)**3 * y)/np.sum(y)
    print("m3=", m3)

    tau = np.power(m3/2, 1/3)
    sigma = np.sqrt(m2 - tau**2)
    mu = m1 - tau

    y_ = emg(x, 1, mu, sigma, tau)
    y_ *= np.max(y)/np.max(y_)

    h, t0, np_, tp_ = fit_single_pore(x, y, moments=[m1, m2, m3])
    y_mp = h * robust_single_pore_pdf(x - t0, np_, tp_)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Comparison among GRM, EMG and Monopore")
        ax.plot(x, y, label='GRM - General Rate Model')
        ax.plot(x, y_, label='EMG')
        ax.plot(x, y_mp, label='Monopore')

        ax.legend()
        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
