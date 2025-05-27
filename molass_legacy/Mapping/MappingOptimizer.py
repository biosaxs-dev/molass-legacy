"""
    MappingOptimizer.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from scipy import optimize
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt

MAJOR_PEAK_THRESHOLD_RATIO  = 0.3
USE_MODEL_PEAKS = True
USE_SLICE = True

def draw_zoomed_peaks_for_proof(a_curve, x_curve):
    from molass_legacy.Elution.CurveUtils import simple_plot
    from DataUtils import get_in_folder
    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("Proof Plot of the Poor Mapping on " + get_in_folder(), fontsize=20)
        ax1.set_title("UV Elution Curve", fontsize=16)
        ax2.set_title("XR Elution Curve", fontsize=16)

        for ax, ecurve, color in [(ax1, a_curve, "blue"), (ax2, x_curve, "orange")]:
            simple_plot(ax, ecurve, color=color, legend=False)

            # inset axes....
            x = ecurve.x
            y = ecurve.y
            j = np.argmax(y)
            px = x[j]
            py = y[j]
            width = ecurve.get_peak_region_width()
            dx = width/8
            dy = py*0.05
            axins = ax.inset_axes(
                [0.58, 0.3, 0.4, 0.4],
                xlim=(px-dx, px+dx), ylim=(py-2*dy, py+dy), xticklabels=[], yticklabels=[])
            axins.grid(False)
            # axins.set_facecolor('gray')
            for name in ['top', 'bottom', 'left', 'right']:
                axins.spines[name].set_color('black')

            for epeak in ecurve.get_emg_peaks():
                y_ = epeak.get_model_y(x)
                axins.plot(x, y_, color="cyan", lw=3)
                ymin, ymax = axins.get_ylim()
                cx = epeak.opt_params[1]
                axins.plot([cx, cx], [ymin, ymax], lw=3, color="yellow", label=r"EGH $\mu$")
                axins.legend(loc="upper left")

            simple_plot(axins, ecurve, color=color, legend=False)

            ax.indicate_inset_zoom(axins, edgecolor="black")
            ax.legend()

        fig.tight_layout()
        plt.show()

def make_slice_for_optimization(ecurve, debug=True):
    f, t = ecurve.get_peak_region_sigma()
    i, j = [bisect_right(ecurve.x, p) for p in [f, t]]
    return slice(i,j+1)

def get_reliable_peak_top_x(ecurve):
    # consider doing this earlier
    ret_x = []
    for epeak in ecurve.get_emg_peaks():
        ret_x.append(epeak.opt_params[1])
    return ret_x

def optimize_mapping_impl(self, debug=False):

    self.determine_mapping_ranges()

    self.inv_mapped_boundaries = []
    tgt_y = self.x_curve.y
    xray_x = self.x_curve.x
    uv_x = self.a_curve.x

    assert len(self.a_curve.peak_top_x) == len(self.x_curve.peak_top_x)
    threshold = self.x_curve.max_y * MAJOR_PEAK_THRESHOLD_RATIO

    if debug:
        draw_zoomed_peaks_for_proof(self.a_curve, self.x_curve)

    if USE_MODEL_PEAKS:
        xr_candidates = get_reliable_peak_top_x(self.x_curve)
        uv_candidates = get_reliable_peak_top_x(self.a_curve)
    else:
        xr_candidates = self.x_curve.peak_top_x
        uv_candidates = self.a_curve.peak_top_x

    xr_top_x = []
    xr_top_y = []
    uv_top_x = []
    for k, px in enumerate(xr_candidates):
        py = self.x_curve.spline(px)
        if py > threshold:
            xr_top_x.append(px)
            xr_top_y.append(py)
            uv_top_x.append(uv_candidates[k])
    xr_top_x = np.array(xr_top_x)
    xr_top_y = np.array(xr_top_y)
    peak_weights = xr_top_y/np.sum(xr_top_y)
    uv_top_x = np.array(uv_top_x)
    num_peaks = len(xr_top_x)

    if USE_SLICE:
        slice_for_opt = make_slice_for_optimization(self.x_curve)
        if debug:
            from matplotlib.patches import Rectangle
            from molass_legacy.Elution.CurveUtils import simple_plot
            from DataUtils import get_in_folder
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("Range for Mapping Optmization on " + get_in_folder())
                simple_plot(ax, self.x_curve, legend=False)
                ymin, ymax = ax.get_ylim()
                f, t = slice_for_opt.start, slice_for_opt.stop
                p_ = Rectangle(
                        (f, ymin),      # (x,y)
                        t - f,          # width
                        ymax - ymin,    # height
                        facecolor='cyan',
                        alpha=0.2,
                        label="Optimization Range",
                    )
                ax.add_patch(p_)
                ax.legend()
                fig.tight_layout()
                plt.show()
    else:
        slice_for_opt = slice(None, None)

    tgt_y_ = tgt_y[slice_for_opt]

    def objective(p, debug_info=None):
        A, B = p[0:2]
        mapped_y, _ = self.make_whole_mapped_vector(A, B, p[2:], make_inv_mapped_boundaries=True, with_original_scale=True)
        msd = np.average((mapped_y[slice_for_opt] - tgt_y_)**2)
        if num_peaks == 1:
            psd = 1
        else:
            psd = np.average((peak_weights*(A*xr_top_x + B - uv_top_x))**2)

        if debug_info is not None:
            title, = debug_info
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.plot(xray_x, tgt_y, label="tgt_y")
                ax.plot(xray_x, mapped_y, label="mapped_y")
                ymin, ymax = ax.get_ylim()
                if USE_SLICE:
                    from matplotlib.patches import Rectangle
                    f, t = slice_for_opt.start, slice_for_opt.stop
                    p_ = Rectangle(
                            (f, ymin),      # (x,y)
                            t - f,          # width
                            ymax - ymin,    # height
                            facecolor   = 'cyan',
                            alpha       = 0.2,
                        )
                    ax.add_patch(p_)
                ax.set_ylim(ymin, ymax)
                for xrx, uvx in zip(xr_top_x, (uv_top_x - B)/A):
                    ax.plot([xrx, xrx], [ymin, ymax], color="orange", label="XR peak top")
                    ax.plot([uvx, uvx], [ymin, ymax], color="blue", label="UV peak top")
                ax.legend()
                fig.tight_layout()
                plt.show()

        return msd*psd

    if debug:
        self.logger.info("feature_mapped=%s, manual_sync=%s", str(self.feature_mapped), str(self.manual_sync))

    if self.feature_mapped or self.manual_sync:
        if self.manual_sync:
            from molass_legacy._MOLASS.SerialSettings import get_setting
            A_ = get_setting('manual_time_scale')
            B_ = get_setting('manual_time_shift')
            A_init = 1/A_
            B_init = -B_/A_
        else:
            A_init = self.A_init
            B_init = self.B_init

        init_params = (A_init, B_init, *self.init_scales)
        delta = 1e-10   # trick to make the code simple for these options
        ab_bounds = ((A_init-delta, A_init+delta), (B_init-delta, B_init+delta))
    else:
        b_delta = uv_x[-1]*0.1
        init_params = (self.A_init, self.B_init, *self.init_scales)
        ab_bounds = ((self.A_init*0.9, self.A_init*1.1), (self.B_init-b_delta, self.B_init+b_delta))

    scale_bounds = [(0.1*s, 10*s) for s in self.init_scales]
    bounds = (*ab_bounds, *scale_bounds)

    if debug:
        objective(init_params, debug_info=("before minimize",))

    ret = optimize.minimize(objective, init_params, bounds=bounds)

    if debug:
        objective(ret.x, debug_info=("after minimize",))

    A, B = ret.x[0:2]

    if self.manual_sync:
        new_A_ = 1/A
        new_B_ = -B/A
        self.logger.info("manual params unchanged proof: (%g, %g) ⇔ (%g, %g)", A_, B_, new_A_, new_B_)

    self.mapped_vector, self.mapped_vector_orig = self.make_whole_mapped_vector(A, B, ret.x[2:], make_inv_mapped_boundaries=True, with_original_scale=True )
    self.map_params = A, B
    self.scale_params = ret.x[2:]

    # this is temporary for backward compatibility and should be removed in the near future
    temp_results = []
    A, B = self.map_params
    for S in self.scale_params:
        temp_results.append((A, B, S))
    self.opt_results = temp_results

    self.uv_ranges = []
    for i, range_ in enumerate( self.mapping_ranges ):
        self.uv_ranges.append( [ A*x + B for x in range_ ] )

    assert len(self.mapped_vector) == len(self.x_x)

    self.mapped_spline  = UnivariateSpline( self.x_x, self.mapped_vector, s=0, ext=3 )

    if False:
        from CallStack import CallStack
        cs = CallStack()
        print(cs)
        print( 'len(self.x_x)=', len(self.x_x) )
        print( 'len(self.mapped_vector)=', len(self.mapped_vector) )
        ax = plt.gca()
        ax.cla()
        ax.plot( self.x_x, self.x_curve.y, color='orange', label='x_curve.y' )
        ax.plot( self.mapped_vector, color='blue', label='mapped_vector' )
        ax.plot( self.x_x, self.mapped_spline(self.x_x), color='green', label='mapped_spline' )
        ax.legend()
        plt.tight_layout()
        plt.show()

    assert len(self.x_x) == len(self.mapped_vector)
