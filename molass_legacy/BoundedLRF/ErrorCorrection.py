"""
    ErrorCorrection.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from molass_legacy.Trimming.Sigmoid import sigmoid
from molass_legacy.DENSS.DenssUtils import fit_data

def compute_corrected_error(qv, i, P, c1, c2, P_, Pe, bq_bounds_, coerced_bq_, debug=False):

    aq, bq = P.T[0:2]
    aq_, bq_ = P_.T[0:2]

    ae = Pe[:,0]
    p0 = np.zeros(Pe.shape[0])
    modified = np.max([p0, (bq - coerced_bq_)*c2/c1], axis=0)
    modified_ratio = modified/ae

    p1 = np.ones(len(aq))
    corrected_ratio = np.min([p1, modified_ratio], axis=0)

    def sigmoid_fit_0(p):
        sy = sigmoid(qv, *p)
        return np.sum((sy - modified_ratio)**2) + np.sum(sy[0:i]**2)*1e5

    ret0 = minimize(sigmoid_fit_0, (1, 0.1, 1, 0))

    def sigmoid_fit_1(p):
        sy = sigmoid(qv, *p)
        return np.sum((sy - corrected_ratio)**2) + np.sum(sy[0:i]**2)*1e5

    ret1 = minimize(sigmoid_fit_1, ret0.x)

    def compute_delta(qv, aq, ae):
        qc, Ic, Icerr, Dc = fit_data(qv, aq, ae)
        spline = UnivariateSpline(qc, Ic, s=0, ext=3)
        delta = (aq - spline(qv))/ae
        return delta

    ae = Pe[:,0]
    delta = compute_delta(qv, aq, ae)

    correct_scale = 2.14        # this scale should be stable for various data

    ae_ = ae * (1 - sigmoid(qv, *ret1.x)*correct_scale)
    # how abount be?

    delta_ = compute_delta(qv, aq_, ae_)

    adjust_scale = np.std(delta) / np.std(delta_)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        print("adjust_scale=", adjust_scale)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("Correction Sigmoid Model Estimation", fontsize=20)
            ax.plot(qv, bq_, color="pink")

            for k, bound in enumerate(bq_bounds_):
                label = "B(q) Bounds" if k == 0 else None
                ax.plot(qv, bound, ":", color="red", label=label)

            ax.set_ylim(-3, 3)

            axt = ax.twinx()
            axt.grid(False)

            axt.plot(qv, modified_ratio, label="A(q) error modified ratio")
            axt.plot(qv, corrected_ratio, label="A(q) error corrected ratio")
            cy = sigmoid(qv, *ret1.x)
            axt.plot(qv, cy, lw=5, label="Correction ratio sigmoid")
            axt.plot(qv, cy*correct_scale, ":", lw=5, label="Effective correction ratio sigmoid (scale=%g)" % correct_scale)

            axt.legend(fontsize=16)

            fig.tight_layout()
            plt.show()

    Pe_ = Pe.copy()
    Pe_[:,0] = ae_ / adjust_scale
    return Pe_, adjust_scale
