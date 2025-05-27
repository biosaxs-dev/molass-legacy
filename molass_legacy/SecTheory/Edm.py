"""
    SecTheory.Edm.py

    EDM - Equilibrium Dispersive Model

    based on the paper 2022, Jamil Ur Rehman,
        ANALYSIS OF EQUILIBRIUM DISPERSIVE MODEL
        OF LIQUID CHROMATOGRAPHY CONSIDERING
        A QUADRATIC-TYPE ADSORPTION ISOTHERM

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.special import erfc
from scipy.optimize import minimize, basinhopping

class Edm:
    def __init__(self, z=30, L=30, a=1.5, b=0.5, u=1.2, e=0.4, Dz=0.02, cinit=0, cinj=1.0, c0=0.0001, tinj=2.0):
        self.a = a
        self.b = b
        self.e = e
        self.Dz = Dz
        self.cinj = cinj
        self.F = F = (1 - e)/e      # phase ratio
        self.L = L
        self.x = z/L
        self.u = u
        self.Pe = Pe = L*u/Dz   # 

        denominator = (1 + b*c0)**3
        gam1 = (a * b**2 * c0**3) / denominator
        gam2 = (a + 3*a*b*c0) / denominator
        gam3 = (-a*b) / denominator
        self.R = R = 1 + gam2*F
        self.lam = lam = 2*F*gam3/R
        tau_inj = u*tinj/L
        self.Beta = lam*Pe/2 * cinj * tau_inj
        self.RPe = R*Pe

    def get_comp_params(self):
        return self.a, self.b, self.e, self.Dz, self.cinj

    def __call__(self, t):
        x = self.x
        L = self.L
        u = self.u
        tau = u*t/L
        xi = x - tau/self.R
        Pe = self.Pe

        U = 2/(self.lam*Pe)
        RPe = self.RPe
        V = xi**2/(4*tau/RPe)
        W = np.sqrt(np.pi*tau/RPe)
        Y = xi/(2*np.sqrt(tau/RPe))

        numerator = ( -U*np.exp(-V) + U*np.exp(self.Beta - V) )
        denominator = ( W*np.exp(self.Beta)* erfc(-Y) + W*erfc(Y) )

        ret_y = numerator / denominator
        ret_y[np.isnan(ret_y)] = 0
        return ret_y

NEGATIVE_PENALTY_SCALE = 1000

def guess_single_edm(x, y, init_params=None, debug=False):

    z = 30
    L = 30
    u = 0.5

    def objective(p):
        a, b, e, Dz, cinj = p
        edm = Edm(z=z, L=L, a=a, b=b, u=u, e=e, Dz=Dz, cinj=cinj)
        y_ = edm(x)
        negative = NEGATIVE_PENALTY_SCALE*min(0, Dz)**2
        fv = np.sum((y_ - y)**2) + negative
        return fv

    if init_params is None:
        init_params = (1.5, -3.0, 0.4, 0.06, 0.5)

    ret = minimize(objective, init_params)
    a, b, e, Dz,cinj = ret.x
    print("params=", (a, b, e, Dz, cinj))
    # [1]  (1.5047442427375226, -2.627270376342513,  0.3861655349460303,  0.05006483813658697,  0.584771571520628)
    # [2]  (1.4479395094650214, -3.0295291424530553, 0.25816194967491296, 0.011396843248255373, 0.052114693543924236)
    edm = Edm(z=z, L=L, a=a, b=b, u=u, e=e, Dz=Dz, cinj=cinj)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(x, y, label="data")
            ax.plot(x, edm(x), label="model")
            ax.legend()
            fig.tight_layout()
            plt.show()

    return edm

def guess_multiple_edms(x, y, num_components, debug=False):
    models = []
    y_ = y.copy()
    for k in range(num_components):
        edm = guess_single_edm(x, y_, debug=debug)
        models.append(edm)
        if len(models) == num_components:
            break

        cy = edm(x)
        y_ -= cy
    return models

def edm_func(t, u, a, b, e, Dz, cinj, cinit=0, c0=0.0001, tinj=2.0, L=30, z=30):
    F = (1 - e)/e      # phase ratio
    x = z/L
    Pe = L*u/Dz   # 

    denominator = (1 + b*c0)**3
    gam2 = (a + 3*a*b*c0) / denominator
    gam3 = (-a*b) / denominator
    R = 1 + gam2*F
    lam = 2*F*gam3/R
    tau_inj = u*tinj/L
    Beta = lam*Pe/2 * cinj * tau_inj
    RPe = R*Pe

    tau = u*t/L
    xi = x - tau/R

    U = 2/(lam*Pe)
    V = xi**2/(4*tau/RPe)
    W = np.sqrt(np.pi*tau/RPe)
    Y = xi/(2*np.sqrt(tau/RPe))

    # numer = U*( -np.exp(-V) + np.exp(Beta - V) )
    # denom = W*( np.exp(Beta)* erfc(-Y) + erfc(Y) )
    expB = np.exp(Beta)
    numer = U*(expB - 1)*np.exp(-V)
    denom = W*( expB * erfc(-Y) + erfc(Y) )

    ret_y = numer / denom
    ret_y[np.isnan(ret_y)] = 0
    return ret_y
