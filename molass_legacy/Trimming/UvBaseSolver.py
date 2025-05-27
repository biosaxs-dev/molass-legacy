"""
    UvBaseSolver.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
from bisect import bisect_right
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d
from SvdDenoise import get_denoised_data
from molass_legacy.Peaks.EghSupples import egh, d_egh
from .Sigmoid import ex_sigmoid

class UvBaseSolver:
    def __init__(self, wv, A, info, debug=False):
        self.info = info
        self.wv = wv
        self.A = A
        self.indeces = [bisect_right(wv, w) for w in [280, 340]]

        if debug:
            egh_param_list, sigmoid_param_list, d_egh_param_list = info
            i1, i2 = self.indeces
            with plt.Dp():
                fig = plt.figure(figsize=(18,5))
                fig.suptitle("UvBaseSolver.__init__ debug")
                ax1 = fig.add_subplot(131, projection="3d")
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133)

                simple_plot_3d(ax1, A, x=wv)

                x = np.arange(A.shape[1])

                for ax, i in [(ax2, i1), (ax3, i2)]:
                    ax.plot(x, A[i,:])

                for params in egh_param_list:
                    ax2.plot(x, egh(x, *params))

                for params in sigmoid_param_list:
                    ys = ex_sigmoid(x, *params)
                    ax3.plot(x, ys)

                for k, params in enumerate(egh_param_list):
                    params[0] = d_egh_param_list[k]
                    yd = d_egh(x, *params)
                    ax3.plot(x, ys + yd)

                fig.tight_layout()
                plt.show()

    def solve(self, optimize=False, debug=False):
        U = get_denoised_data(self.A, rank=2)

        egh_param_list, sigmoid_param_list, d_egh_param_list = self.info

        egh_param_array = np.array(egh_param_list)
        sigmoid_param_array = np.array(sigmoid_param_list)
        d_egh_param_array = np.array(d_egh_param_list)

        np1 = np.product(egh_param_array.shape)
        np2 = np1 + np.product(sigmoid_param_array.shape)

        x = np.arange(U.shape[1])
        egh_param_list_ = [prm.copy()for prm in egh_param_list]

        def compute_PQ(p):
            egh_params = p[0:np1].reshape(egh_param_array.shape)
            sigmoid_params = p[np1:np2].reshape(sigmoid_param_array.shape)
            d_egh_params = p[np2:]

            c = np.zeros(len(x))
            for prms in egh_params:
                c += egh(x, *prms)

            g = np.zeros(len(x))
            for prms in sigmoid_params:
                g += ex_sigmoid(x, *prms)

            for k, prms in enumerate(egh_param_list_):
                prms[0] = d_egh_params[k]
                g += d_egh(x, *prms)

            Q = np.array([c, g])
            P = U @ np.linalg.pinv(Q)
            return P, Q

        p0 = np.concatenate([array.flatten() for array in [egh_param_array, sigmoid_param_array, d_egh_param_array]])

        P0, Q0 = compute_PQ(p0)
        U0 = P0 @ Q0
        U00 = P0[:,[0]] @ Q0[[0],:]
        U01 = P0[:,[1]] @ Q0[[1],:]

        if debug:
            from matplotlib.gridspec import GridSpec
            from DataUtils import get_in_folder
            wv = self.wv
            A = self.A
            gs = GridSpec(9,3)
            # fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(24,5), subplot_kw=dict(projection="3d"))
            with plt.Dp():
                fig = plt.figure(figsize=(15,12))
                ax1, ax2, ax3 = [fig.add_subplot(gs[0:3,k], projection="3d")  for k in range(3)]
                ax3u = fig.add_subplot(gs[3:6,2], projection="3d")
                ax3d = fig.add_subplot(gs[6:9,2], projection="3d")
                fig.suptitle("Trial Result of the New Correction Method based on Rank2 LRF for %s" % get_in_folder(), fontsize=20)
                ax1.set_title("Input Data", fontsize=16)
                ax2.set_title("Rank2 denoised", fontsize=16)
                ax3.set_title("Reconstructed with Rank2 LRF", fontsize=16)
                ax3u.set_title("Result (First Component only)", fontsize=16)

                simple_plot_3d(ax1, self.A)
                simple_plot_3d(ax2, U)
                simple_plot_3d(ax3, U0)
                simple_plot_3d(ax3u, U00)
                zmin, zmax = ax3u.get_zlim()
                ax3d.set_zlim( zmin, zmax)
                simple_plot_3d(ax3d, U01)
                for ax in [ax1, ax2, ax3, ax3u, ax3d]:
                    ax.view_init(15, 30)
                fig.tight_layout()
                plt.show()

        self.U00 = U00
        self.U01 = U01
        if not optimize:
            self.U00_ = None
            self.U01_ = None
            return

        i1, i2 = self.indeces

        def objective_func(p):
            P, Q = compute_PQ(p)
            return np.linalg.norm(P@Q - U)

        result = minimize(objective_func, p0)
        # result = basinhopping(objective_func, p0)

        p0_ = result.x
        P0_, Q0_ = compute_PQ(p0_)
        U0_ = P0_ @ Q0_
        U00_ = P0_[:,[0]] @ Q0_[[0],:]
        U01_ = P0_[:,[1]] @ Q0_[[1],:]

        print("objective values=", [objective_func(p) for p in [p0, p0_]])

        self.U00_ = U00_
        self.U01_ = U01_

    def get_baseline(self, debug=False):
        A = self.A
        i1, i2 = self.indeces

        if debug:
            from matplotlib.gridspec import GridSpec
            from DataUtils import get_in_folder
            wv = self.wv
            x = np.arange(A.shape[1])

            def plot_resuls(ax0, ax1, ax2, ax3, ax4, U0, U1):
                simple_plot_3d(ax0, A, x=wv)
                zmin, zmax = ax0.get_zlim()
                for ax in [ax1, ax2]:
                    ax.set_zlim(zmin, zmax)
                simple_plot_3d(ax1, U0, x=wv)
                simple_plot_3d(ax2, U1, x=wv)

                a1 = A[i1,:]
                b1 = U1[i1,:]
                ax3.plot(x, a1, label="input data")
                ax3.plot(x, a1 - b1, label="corrected")
                # ax3.plot(x, U0[i1,:], label="egh model")
                ax3.plot(x, U1[i1,:], color="red", label="LRF baseline")
                ax3.legend()

                a2 = A[i2,:]
                b2 = U1[i2,:]
                ax4.plot(x, a2, label="input data")
                ax4.plot(x, a2 - b2, label="corrected")
                ax4.plot(x, U1[i2,:], color="red", label="LRF baseline")
                ax4.legend()

            with plt.Dp():
                # fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10,8))
                U_list = [(self.U00, self.U01)]
                if self.U00_ is not None:
                    U_list.append((self.U00_, self.U01_))
                nrows = len(U_list)
                gs = GridSpec(nrows,5)
                fig = plt.figure(figsize=(23,4*nrows))
                for k, upair in enumerate(U_list):
                    ax0 = fig.add_subplot(gs[k,0], projection="3d")
                    ax1 = fig.add_subplot(gs[k,1], projection="3d")
                    ax2 = fig.add_subplot(gs[k,2], projection="3d")
                    ax3 = fig.add_subplot(gs[k,3])
                    ax4 = fig.add_subplot(gs[k,4])
                    plot_resuls(ax0, ax1, ax2, ax3, ax4, *upair)
                fig.tight_layout()
                plt.show()

        return self.U01[i1,:].copy()

def get_lrf_elution_base(sd, pre_recog, debug=False):
    from molass_legacy._MOLASS.SerialSettings import get_setting

    U, _, wv, ecurve = sd.get_uv_data_separate_ly()
    info = pre_recog.flowchange.get_solver_info(debug=debug)
    ubs = UvBaseSolver(wv, U, info)
    ubs.solve(debug=debug)
    base = ubs.get_baseline(debug=False)
    uv_restrict_list = get_setting('uv_restrict_list')
    if uv_restrict_list is None:
        eslice = slice(None, None)
    else:
        elution_restrict = uv_restrict_list[0]
        if elution_restrict is None:
            eslice = slice(None, None)
        else:
            eslice = slice(elution_restrict.start, elution_restrict.stop)
    return base[eslice].copy()
