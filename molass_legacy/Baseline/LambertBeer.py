"""

    LambertBeer.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF

"""
import logging
import copy
import numpy as np
from scipy.optimize import minimize
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy.KekLib.ThreeDimUtils import compute_plane
from molass_legacy._MOLASS.SerialSettings import get_setting
USE_PEAK_TOP_RIDGE = True
NEGATIVE_PENALTY = 1
POSITIVE_AREA_RATIO = 0.1

class BasePlane:
    def __init__(self, data, index, ecurve, denoise=False, rank=None, j0=0, debug=True):
        self.logger = logging.getLogger(__name__)

        if denoise:
            if rank is None:
                rank = len(ecurve.peak_info) + 1
            data = get_denoised_data(data, rank=rank)

        self.data = data
        self.index = index
        self.x = np.arange(data.shape[0])
        self.y = np.arange(data.shape[1])
        self.divided = data.shape[0] < 100
        self.NW = np.ones(data.shape)
        safe_level = ecurve.max_y*POSITIVE_AREA_RATIO
        self.NW[data < safe_level] = 0
        self.cd = get_setting('xr_bpa_option')
        self.consider_cd = self.cd == 2

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.DataStructure.MatrixData import simple_plot_3d, contour_plot

            print('data.shape=', data.shape, 'index=', index)
            print('type(ecurve)=', type(ecurve))

            with plt.Dp():
                fig = plt.figure(figsize=(21,6))
                ax1 = fig.add_subplot(131, projection='3d')
                ax2 = fig.add_subplot(132)
                ax3 = fig.add_subplot(133, projection='3d')

                ax1.set_title("Surface Plot", fontsize=16, y=1.08)
                simple_plot_3d(ax1, data)

                ax2.set_title("Contour Plot", fontsize=16)
                cp = contour_plot(ax2, data, levels=np.linspace(0.5, 10, 5)*safe_level)
                # fig.colorbar(cp, ax=ax2)  # TODO

                ax3.set_title("Weights for Negative Penalty", fontsize=16, y=1.08)
                simple_plot_3d(ax3, self.NW)

                fig.tight_layout()
                plt.show()

        self.ecurve = ecurve
        self.j0 = int(ecurve.x[0])
        # self.make_C_matrix()
        # self.make_P_matrix()

    def make_C_matrix(self):
        self.C = np.array([self.ecurve.sy])

    def make_P_matrix(self):
        ridge_list = []
        for info in self.ecurve.peak_info:
            top_j = info[1]
            if self.divided:
                sy = self.data[:,top_j]
            else:
                sy = smooth(self.data[:,top_j])
            ridge_list.append(sy)

        self.P = np.array(ridge_list).T

    def objective(self, params):
        if len(params) == 2:
            a = 0
            b, c = params
        else:
            a, b, c = params
        BP  = compute_plane(a, b, c, self.x, self.y)
        M_  = self.data - BP
        # ey_ = - (self.y*b + c)

        # print('peak_info=', self.ecurve.peak_info)

        P_list = []
        C_list = []
        start = 0
        for k, info in enumerate( self.ecurve.peak_info ):
            if k < len(self.ecurve.boundaries):
                stop = self.ecurve.boundaries[k] - self.j0
            else:
                stop = len(self.y)

            # print([k], start, stop)

            peak_j  = info[1] - self.j0
            P1 = M_[:,peak_j]
            P_list.append(P1)
            C_ = np.zeros(len(self.y))
            V_ = M_[self.index,start:stop]       # extract the corrected concentration
            """
            subtract baseline from V_ with a, b, c.
            must consider the scale between M_ and the SMP - standard mapping plane.
            investigate the objective function at this position.
            """
            # V_ = self.ecurve.sy[start:stop]
            C_[start:stop] = V_/np.max(V_)  # normalize to the max
            C_list.append(C_)
            if self.consider_cd:
                C_list.append(C_**2)

            start = stop

        C_ = np.array(C_list)
        if USE_PEAK_TOP_RIDGE and self.cd == 1:
            P_ = np.array(P_list).T
        else:
            P_ = M_ @ np.linalg.pinv(C_)
        PC = np.dot(P_, C_)
        N_ = copy.deepcopy(M_)
        N_[N_ > 0] = 0

        self.P = P_
        self.C = C_

        ret_val = np.linalg.norm((PC - M_)) + np.linalg.norm((self.NW*N_))*NEGATIVE_PENALTY

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.DataStructure.ModeledData import simple_plot_3d
            print('params=', (a, b, c))
            print('ret_val=', ret_val)
            if self.divided:
                index = 0
                i_slice = slice(None, None)
                j_slice = slice(None, None)
                j_width = self.data.shape[1]
            else:
                index = self.index
                i_slice = slice(0,30)
                j_width = 50
                j_start = 180
                j_slice = slice(j_start,j_start+j_width)
            with plt.Dp():
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(str(self.debug_info))

                x_ = np.ones(j_width)*index
                y_ = np.arange(j_width)
                for k in range(self.C.shape[0]):
                    z_ = C_[k,j_slice]
                    scale = self.P[0,k]
                    zm = z_*scale + (a*index + b*y_ + c)
                    ax.plot(x_, y_, zm, color='orange', linewidth=5)

                simple_plot_3d(ax, PC[i_slice,j_slice], alpha=0.2)
                simple_plot_3d(ax, M_[i_slice,j_slice], alpha=0.2)
                simple_plot_3d(ax, self.data[i_slice,j_slice], color='yellow', alpha=0.2)
                simple_plot_3d(ax, BP[i_slice,j_slice], color='red', alpha=0.2)

                fig.tight_layout()
                plt.show()

        return ret_val

    def solve(self, debug=False, debug_info=None):
        self.debug_info = debug_info
        data = self.data
        self.allow_nonzero_a = get_setting('allow_angular_slope_in_mf')

        bounds = []
        if self.allow_nonzero_a:
            bounds.append((-1e-2, +1e-2))
        bounds += [
            (-1e-2, +1e-2),
            (-1.0, +1.0),
        ]
        init_params = np.zeros(len(bounds))
        
        result  = minimize(self.objective, init_params, bounds=bounds)
        if len(init_params) == 2:
            a = 0
            b, c = result.x
        else:
            a, b, c = result.x
        self.params = (a, b, c)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            print('self.params=', self.params)
            with plt.Dp():
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(str(debug_info))
                self.debug_plot(ax)
                fig.tight_layout()
                plt.show()

        self.logger.info('MF-baseplane solved with params: (%g, %g, %g)' % self.params)
        return self.params

    def debug_plot(self, ax, plot_plane=True, plane_color='red', view_init=None):
        data = self.data
        a, b, c = self.params

        i = self.x
        j = self.y

        print('params=', self.params)
        # print('len(i)=', len(i), 'P.shape=', self.P.shape, 'C.shape=', self.C.shape)

        xx, yy = np.meshgrid(i, j)
        zz = data[xx, yy]

        ax.plot_surface(xx, yy, zz, alpha=0.2)

        x_ = np.ones(data.shape[1])*self.index
        y_ = j
        for k in range(self.C.shape[0]):
            z_ = self.C[k,:]
            scale = self.P[0,k]
            zm = z_*scale + (a*self.index + b*y_ + c)
            ax.plot(x_, y_, zm, color='orange')

        for k, info in enumerate(self.ecurve.peak_info):
            top_j = info[1]
            y_ = np.ones(len(i))*top_j
            z_ = self.P[:,k]
            ax.plot(i, y_, z_, color='green')

        if plot_plane:
            x = i[[0,-1]]
            y = j[[0,-1]]
            xx, yy = np.meshgrid(x, y)
            zz = xx*a + yy*b + c
            ax.plot_surface(xx, yy, zz, color=plane_color, alpha=0.2)

            y_ = j
            z_ = self.index*a + y_*b + c
            ax.plot(x_, y_, z_, color=plane_color, linewidth=3)

        if view_init is not None:
            ax.view_init(*view_init)

    def get_params(self):
        return self.params

    def get_baseline(self, x):
        a, b, c = self.params
        return x*a + self.y*b + c

    def get_baseplane(self):
        a, b, c = self.params
        plane = compute_plane(a, b, c, self.x, self.y)
        return plane

def get_args(sd, pre_recog, debug=False):
    """
    it would be better if we could do without the following
    if branch.
    """
    if pre_recog is None:
        start = 0
        data = sd.intensity_array[:,:,1].T
    else:
        sd = pre_recog.get_pre_recog_copy()
        start, stop = pre_recog.get_default_angle_range(sd)

        data = sd.intensity_array[:,start:stop,1].T

    xray_slice = sd.xray_slice
    index = (xray_slice.start + xray_slice.stop)//2
    ecurve = sd.xray_curve

    index_ = index - start
    return data, index_, ecurve

def get_standard_baseline(sd, pre_recog, debug=False, logger=None):
    data, index_, ecurve = get_args(sd, pre_recog, debug)
    bp = BasePlane(data, index_, ecurve, debug=debug)
    bp.solve(debug=debug)

    if logger is not None:
        a, b, c = bp.get_params()
        c_ratio = c/ecurve.max_y
        logger.info('baseplane parames: a=%g, b=%g, c=%g, c_ratio=%g', a, b, c, c_ratio)

    return bp.get_baseline(index_)

def get_base_plane(data, index, debug=False):
    from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
    ey = data[index,:]
    ecurve = ElutionCurve(ey)

    bp = BasePlane(data, index, ecurve, debug=debug)
    bp.solve(debug)
    return bp

def compute_base_plane(data, index, ecurve, denoise=False):
    bp = BasePlane(data, index, ecurve, denoise=denoise)
    bp.solve()
    BP = bp.get_baseplane()
    return BP