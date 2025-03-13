"""
    Models/RateTheory/stlcGRM.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from stlc import grm

cm = 1e-2
minute = 60
dt = 1
qm = 1.
nr = 10
nz = 20

def grm_impl(x, zl, rp, ec, ep, kf, Dax, Dp, Ds, u, ka, kd, tinj, cinj):

    # tmax = len(x) * minute  #[min]
    tmax = len(x)

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

    y = sol.c[0, -1, :]
    return y

def guess(x, y):

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
    tinj = 50 * minute  #[min]
    cinj = 1  #[mol /m^-3]

    init_params = (zl, rp, ec, ep, kf, Dax, Dp, Ds, u, ka, kd, tinj, cinj)

    def objective(p):
        y_ = grm_impl(x, *p)
        return np.sum((y_ - y)**2)

    ret = minimize(objective, init_params)

    print("ret.x=", ret.x)

    """
    method="Nelder-Mead"
    ret.x= [ 4.09867334e-07  5.17007792e-05  1.69859416e-06  3.39363091e-01
  8.00288751e-05  2.96235961e-09  5.08620741e-12 -8.87774587e-05
  2.14400226e-04  1.07395544e+01  1.09575930e+00  2.03315646e+02
  1.27581541e+00]
    """

    return ret.x
