"""
    IterativeLrfSolver.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from SvdDenoise import get_denoised_data
from SimpleGuinier import SimpleGuinier
from molass_legacy.Trimming.Sigmoid import sigmoid
from Theory.SolidSphere import get_boundary_params

class IterativeLrfSolver:
    def __init__(self, qv, D, E, i=None, denoise=True):
        if i is None:
            from bisect import bisect_right
            i = bisect_right(qv, 0.02)

        self.qv = qv
        self.i = i
        if denoise:
            D_ = get_denoised_data(D, rank=2)
        else:
            D_ = D
        self.D_ = D_
        self.E = E
        cy = np.average(D_[i-5:i+6,:], axis=0)
        self.cy_init = cy
        self.j = np.argmax(cy)
        self.Cinit = np.array([cy, cy**2])

    def solve(self, maxiter=10000, L=1, rg_refresh_cycle=20000, limit_ratio=0.003, debug=False):
        qv = self.qv
        i = self.i
        j = self.j
        cy_init = self.cy_init
        C = self.Cinit
        D_ = self.D_
        pjv = D_[:,j]/cy_init[j]
        E = self.E
        changes = []
        aq_history = []
        last_ratio = None

        for k in range(maxiter+1):
            Cinv = np.linalg.pinv(C)
            P = D_ @ Cinv
            aq, bq = P.T
            c1, c2 = C[:,j]

            if k % rg_refresh_cycle == 0:
                def compute_bounds(rg_only=False):
                    D_pinv = np.linalg.pinv(D_)
                    W = np.dot(D_pinv, P)
                    Pe = np.sqrt(np.dot(E**2, W**2))

                    data = np.array([qv, aq, Pe[:,0]]).T
                    sg = SimpleGuinier(data)
                    Rg = sg.Rg
                    print([k], "Rg=", Rg)
                    if rg_only:
                        return Rg
                    R = np.sqrt(5/3)*Rg
                    bound = 1/(qv*L*R)**2

                    bq_bound = bound * aq/c1
                    bq_bounds = (-bq_bound, bq_bound)
                    return Rg, R, bq_bounds
                Rg, R, bq_bounds = compute_bounds()
                aq_history.append(aq)
                last_ratio = None       # to avoid too early finish

            if k == 0:
                aq_init = aq
                c1_init = c1

            coerced_bq = P[:,1].copy()

            where = bq < bq_bounds[0]
            coerced_bq[where] = bq_bounds[0][where]
            where = bq > bq_bounds[1]
            coerced_bq[where] = bq_bounds[1][where]

            changed_norm_ratio = np.linalg.norm(coerced_bq - P[:,1])/np.linalg.norm(P[:,1])
            if k % 1000 == 0:
                print([k], "changed_norm_ratio=", changed_norm_ratio)
                changes.append(changed_norm_ratio)

            if (k >= maxiter
                or changed_norm_ratio < limit_ratio 
                or last_ratio is not None and changed_norm_ratio > last_ratio
                ):
            # if k >= maxiter:
                print([k], changed_norm_ratio, last_ratio)
                # recompute the filnal Rg
                Rg = compute_bounds(rg_only=True)
                k_final = k
                aq_history.append(aq)
                break


            last_ratio = changed_norm_ratio
            P = np.array([aq + (bq - coerced_bq)*c2/c1, coerced_bq]).T
            # P[:,1] = coerced_bq
            # P = np.array([aq, coerced_bq]).T
            Pinv = np.linalg.pinv(P)
            C = Pinv @ D_

        b1, b2, k_sig = get_boundary_params(Rg, qv=qv)
        w = sigmoid(qv, 1, b1, k_sig, 0)
        aq_init_ = aq_init*c1/c1_init
        modified_aq = aq_init_*(1 - w) + aq*w
        print("c1_init, c1=", c1_init, c1)

        if debug:
            glim_i = bisect_right(qv, 2.0/Rg)
            gslice = slice(0, glim_i)
            x_ = np.arange(len(changes))*1000
            qv2 = qv[gslice]**2

            with plt.Dp():
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
                ax1.set_title("Norm Difference Ratio Changes")
                ax1.plot(x_, changes)
                ax2.set_title("A(q) Changes")
                ax3.set_title("A(q) Changes (Guinier Plot)")
                ax2.set_yscale("log")
                ax2.plot(qv, pjv, label="data")
                ax3.plot(qv2, np.log(pjv[gslice]), label="data")
                for k_, aq_ in enumerate(aq_history):
                    label_k = k_*rg_refresh_cycle if k_ < len(aq_history)-1 else k_final
                    label = "[%d] A(q)" % label_k
                    ax2.plot(qv, aq_, label=label)
                    ax3.plot(qv2, np.log(aq_[gslice]), label=label)
                ax2.plot(qv, modified_aq, ":", color="red", lw=2, label="modifeid A(q)")
                ax3.plot(qv2, np.log(modified_aq[gslice]), ":", color="red", lw=2, label="modifeid A(q)")
                ax2.legend()
                ax3.legend()
                axt = ax2.twinx()
                axt.grid(False)
                axt.plot(qv, w, ":", color="cyan")
                fig.tight_layout()
                plt.show()

        P[:,0] = modified_aq

        return P, C, Rg, R, L, bq_bounds, coerced_bq, w, np.array(changes)
