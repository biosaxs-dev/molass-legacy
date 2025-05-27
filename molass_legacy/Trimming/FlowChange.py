"""
    FlowChange.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.stats import linregress
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from .PeakRegion import PeakRegion
from .UvCorrectorEgh import UvCorrectorEgh
from .FlowChangeCandidates import get_largest_gradients
from .FlowChangePrep import get_easier_curve_y
from .FlowChangeUtils import is_ignorable_L_ratio, fit_sigmoid_impl
from .Sigmoid import ex_sigmoid, ex_sigmoid_inv

K_LIMIT_SAFE = 0.2
K_LIMIT_RISKY = 0.11            # < 0.117 for 20190305_1, > 0.108 for 20190227
K_LIMIT_GUARD_RATIO = 5         # < 5.92 for 20190305_1, > 3.32 for 20160227
CUSTOM_K_LIMIT = 0.95           # > 0.729 for OA, < 0.987 for HIF, 0.902 for Factin
K_LIMIT_SPECIAL = 0.05
L_RATIO_CUSTOM_LIMIT = 0.9      # < 0.996 for 20170309
R_VALUE_LIMIT = 0.9             # < 0.945 for SUB_TRN1, 
SAFE_RATIO = 0.99
SLICE_STOP_ALLOW = 10
MIN_DIST_ALLOW = 10
OUTSTANDING_RATIO_LIMIT = 5
SUPERSEDE_RATIO_ALLOW = 0.1
NARROW_MARGIN_RATIO = 0.05
SHOW_SPECIAL_CASE_MESSAGE = False

class FlowChange:
    def __init__(self, a_curve, a_curve2, x_curve, debug=False, fig_file=None):
        self.logger = logging.getLogger(__name__)
        self.a_curve = a_curve
        self.a_curve2 = a_curve2
        self.peak_region = PeakRegion(x_curve, a_curve, a_curve2)
        self.special_case_warned = False
        self.maybe_special = False

        def try_fc_model_fit(x, y, height, peak_region, std_p, pp3, slice_, ratio, outstanding_ratio, use_custom_sigmoid=True):
            print("============================= try_fc_model_fit: slice=", slice_)
            special = False
            failed = False
            try:
                x_ = x[slice_]
                y_ = y[slice_]
                assert len(x_) > 0
                ppk = [p for p in pp3 if x_[0] <= p and p <= x_[-1]]
                assert len(ppk) > 0
                popt, error_ratio = fit_sigmoid_impl(x_, y_, height, peak_region, std_p, ppk, outstanding_ratio, self.logger, debug=debug)
                if popt is None:
                    failed = True
                else:
                    if False:
                        debug_info = x_, y_
                    else:
                        debug_info = None
                    x1, y1, popt, special, L_ratio = self.get_safer_point(x, y, height, popt, ratio, debug_info, use_custom_sigmoid=use_custom_sigmoid)
            except:
                log_exception(self.logger, "fit_sigmoid error: ", n=10)
                failed = True

            if failed:
                x1, y1, popt, error_ratio, L_ratio = None, None, None, np.inf, 0

            return x1, y1, popt, error_ratio, L_ratio, special

        x = a_curve2.x
        y = a_curve2.y
        y_modified, y_for_gy, first_peak, last_peak = get_easier_curve_y(a_curve, a_curve2, self.peak_region, self.logger)

        uv_corrector = UvCorrectorEgh(a_curve, a_curve2)
        gy, pp3, outstanding_ratio, smoothed, corrected_y = get_largest_gradients(y_for_gy, 3, self.peak_region,
                                                return_full_info=True, uv_corrector=uv_corrector, logger=self.logger, debug=debug)

        self.gy = gy
        self.pp3 = pp3
        self.outstanding_ratio = outstanding_ratio
        if corrected_y is None:
            y_for_height = y_for_gy
        else:
            y_for_height = corrected_y

        height = np.max(y_for_height) - np.min(y_for_height)
        std_p = np.std(pp3)

        fc_xy = []
        popts = []
        error_ratios = []
        special_flags = []
        x0 = pp3[0]
        stop1 = first_peak[0]
        if abs(stop1 - x0) < SLICE_STOP_ALLOW:
            # as in 20170301
            stop1 = x0 + SLICE_STOP_ALLOW
        slices = [slice(0, stop1), slice(last_peak[-1], None), slice(0, None)]
        num_tests = 0
        num_candidates = 0
        for slice_, ratio in zip(slices, [SAFE_RATIO, 1 - SAFE_RATIO, SAFE_RATIO]):
            x1, y1, popt, error_ratio, L_ratio, special = try_fc_model_fit(x, y_for_gy, height, self.peak_region, std_p, pp3, slice_, ratio, outstanding_ratio)
            if special:
                self.maybe_special = True
            # print([num_tests], "-------------- x1, y1, popt, error_ratio=", x1, y1, popt, error_ratio)

            num_tests += 1
            if x1 is not None:
                num_candidates += 1

            if num_tests == 3:
                if x1 is not None:
                    if x1 > last_peak[1]:
                        # as in 20200630_3 which should be investigated.
                        self.logger.info("third trial at %.3g in the descending side of the last peak is discarded.", x1)
                    else:
                        # as in pH6
                        fc_xy[0] = (x1, y1, L_ratio, special)    # although it will finally be dismissed
                        popts[0] = popt
            else:
                fc_xy.append((x1, y1, L_ratio, special))
                popts.append(popt)
                error_ratios.append(error_ratio)
                special_flags.append(special)

            if num_tests == 2 and num_candidates > 0:
                break

        draw_debug_plot = debug or fig_file is not None
        if draw_debug_plot:
            import molass_legacy.KekLib.DebugPlot as plt
            def fc_result_plot(gy, pp3, title_trailer=""):
                from matplotlib.patches import Rectangle
                from molass_legacy._MOLASS.SerialSettings import get_setting
                from DataUtils import get_in_folder
                from molass_legacy.Elution.CurveUtils import simple_plot
                from molass_legacy.UV.PlainCurveUtils import get_flat_wavelength

                plt.push()
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,6))

                in_folder = get_in_folder()
                fig.suptitle("Flowchange Recognition on %s%s" % (in_folder, title_trailer), fontsize=20)
                lambda_list = [get_setting("absorbance_picking"), get_flat_wavelength()]
                ax1.set_title("Elution Curve (1) at λ=%.3g" % (lambda_list[0]), fontsize=16)
                ax2.set_title("Elution Curve (2) at λ=%.3g" % (lambda_list[1]), fontsize=16)
                ax3.set_title("Gradient of the Curve (2)", fontsize=16)

                simple_plot(ax1, a_curve, legend=False)
                ymin1, ymax1 = ax1.get_ylim()
                ax1.set_ylim(ymin1, ymax1)

                f, t = self.peak_region.get_wider_ends()
                p = Rectangle(
                        (f, ymin1),     # (x,y)
                        t - f,          # width
                        ymax1 - ymin1,  # height
                        facecolor   = 'cyan',
                        alpha       = 0.1,
                        label = "wider peak region",
                    )
                ax1.add_patch(p)
                ax1.legend()

                ax2.plot(x, y)
                if y_for_gy is not None:
                    ax2.plot(x, y_for_gy, ':', color="gray")
                if corrected_y is not None:
                    ax2.plot(x, corrected_y, color="green")
                ymin2, ymax2 = ax2.get_ylim()
                ax2.set_ylim(ymin2, ymax2)

                color = "green" if smoothed else None
                ax3.plot(x, gy, color=color)
                for i, p in enumerate(pp3):
                    ax3.plot(p, gy[p], "o", label="%d-th significant" % i)

                xmin3, xmax3 = ax3.get_xlim()
                ymin3, ymax3 = ax3.get_ylim()
                tx = (xmin3 + xmax3)/2
                if abs(ymin3) < abs(ymax3):
                    ty = ymax3/2
                else:
                    ty = ymin3/2
                smoothed_str = "\n(of the smoothed)" if smoothed else ""
                ax3.text(tx, ty, "outstanding_ratio=%.3g%s" % (outstanding_ratio, smoothed_str), fontsize=20, alpha=0.5, ha="center")
                ax3.legend()

                for popt, (x1, y1, L_ratio, special), slice_ in zip(popts, fc_xy, slices):
                    if x1 is None:
                        pass
                    else:
                        if self.peak_region.is_in_the_region(x1):
                            color0 = "gray"
                            color1 = "gray"
                            color1p = "gray"
                        elif special:
                            color0 = "cyan"
                            color1 = "cyan"
                            color1p = "cyan"
                        else:
                            color0 = "red"
                            color1 = "yellow"
                            color1p = "orange"

                        x_ = x[slice_]
                        y_ = y[slice_]
                        L, x0, k, b, s1, s2 = popt
                        ax2.plot(x_, ex_sigmoid(x_, *popt))
                        ax2.plot([x0, x0], [ymin2, ymax2], ":", color=color0)
                        ax2.plot([x1, x1], [ymin2, ymax2], color=color1)
                        ax2.plot(x1, y1, "o", color=color1p)
                        xmin2, xmax2 = ax2.get_xlim()
                        tx = x1
                        w = 0.5
                        ty = ymin2 * (1-w) + ymax2 * w
                        ax2.text(tx, ty, "L_ratio=%.3g, k=%.3g" % (L_ratio, k), fontsize=20, alpha=0.5, ha="center")
                        ax1.plot([x1, x1], [ymin1, ymax1], color=color1)
                        ax1.plot(x1, a_curve.spline(x1), "o", color=color1p)

                fig.tight_layout()
                fig.subplots_adjust(top=0.85)

                if fig_file is None:
                    plt.show()
                else:
                    from time import sleep
                    plt.show(block=False)
                    fig = plt.gcf()
                    fig.savefig(fig_file)
                    sleep(1)
                plt.pop()

            fc_result_plot(gy, pp3)

        has_special_ = False
        for n in range(2):
            x1, y1, L_ratio, special = fc_xy[n]
            if x1 is None:
                continue

            if special:
                has_special_ = True

            if self.peak_region.is_in_the_region(x1):
                # as in pH6
                # although similar case as in 20161202 does not fall herein
                fc_xy[n] = None, None, 0, False
                self.logger.info("this flow change at %.4g included in the peak %s has been dismissed.", x1, str(self.peak_region.get_ends()))
            else:
                popt = popts[0]
                if popt is None:
                    # as in 20191006_proteins5
                    continue

                x0 = popt[1]
                pp3d = np.abs(pp3 - x0)
                min_dist = np.min(pp3d)
                max_dist = np.max(pp3d)
                dist_ratio = self.peak_region.get_distance_ratio(x0)
                print("dist_ratio=%.3g" % dist_ratio)
                print("outstanding_ratio=%.3g, (min_dist, max_dist)=(%.3g, %.3g)" % (outstanding_ratio, min_dist, max_dist))
                outstanding = outstanding_ratio > OUTSTANDING_RATIO_LIMIT
                if (max_dist > MIN_DIST_ALLOW
                    and min(*pp3[1:3]) < first_peak[0]      # pp3[1] < first_peak[0] for OAGIwyatt_02
                    ):
                    error_ratio = error_ratios[0]
                    j = int(round(x1))
                    if j < first_peak[1]:
                        slice_ = slice(j, first_peak[1])
                        if min_dist < MIN_DIST_ALLOW:
                            # as in OAGIwyatt_02, 20161202
                            x1_, y1_, popt_, error_ratio_, L_ratio_, special_ = try_fc_model_fit(x, y_for_gy, height, self.peak_region, std_p, pp3, slice_,
                                                                                                    SAFE_RATIO, outstanding_ratio)
                            fix_type = "1"
                            gy_for_debug = gy
                            pp3_for_debug = pp3
                        else:
                            # as in 20170304
                            sy = smooth(y_for_gy)
                            sgy, pp3_ = get_largest_gradients(sy, 3, self.peak_region)
                            std_p_ = np.std(pp3_)
                            x1_, y1_, popt_, error_ratio_, L_ratio_, special_ = try_fc_model_fit(x, sy, height, self.peak_region, std_p_, pp3_, slice_,
                                                                                                    SAFE_RATIO, outstanding_ratio, use_custom_sigmoid=False)
                            fix_type = "2"
                            gy_for_debug = sgy
                            pp3_for_debug = pp3_
                        print("(error_ratio_, error_ratio+allow)=(%.3g, %.3g), special=%s" % (error_ratio_, error_ratio + SUPERSEDE_RATIO_ALLOW, str(special_)) )
                        if (x1_ is not None
                            and not self.peak_region.is_in_the_region(x1_)
                            and error_ratio_ < error_ratio + SUPERSEDE_RATIO_ALLOW
                            ):
                            # as in OAGIwyatt_02 (but not in 20161202)
                            # how abount special cases?
                            fc_xy[n] = x1_, y1_, L_ratio_, special_
                            popts[n] = popt_
                            if special_:
                                self.maybe_special = True

                            self.logger.info("next flow change[%d] at %.4g has been superseded with fix_type=%s and error_ratio %.3g < %.3g + %.3g", n, x1_, fix_type, error_ratio_, error_ratio, SUPERSEDE_RATIO_ALLOW)
                            if draw_debug_plot:
                                fc_result_plot(gy_for_debug, pp3_for_debug, title_trailer=" (after fix%s)" % fix_type)
                    else:
                        # 
                        pass

        self.has_special_ = has_special_
        self.fcj = [None if x1 is None or special else int(round(x1)) for x1, y1, L_ratio, special in fc_xy]
        self.popts = popts

    def has_special(self):
        return self.has_special_

    def get_safer_point(self, x, y, height, popt, ratio, debug_info, use_custom_sigmoid=True):

        special = False
        L, x0, k, b, s1, s2 = popt
        L_ratio = abs(L)/height

        if debug_info is not None:
            import molass_legacy.KekLib.DebugPlot as plt
            print("get_safer_point: L=%.3g, L_ratio=%.3g, k=%.3g" % (L, L_ratio, k))
            x_, y_ = debug_info
            plt.push()
            fig, ax = plt.subplots()
            ax.set_title("get_safer_point debug (1)")
            ax.plot(x, y)
            ax.plot(x_, y_, ":")
            ax.plot(x, ex_sigmoid(x, *popt))
            plt.show()
            plt.pop()

        if is_ignorable_L_ratio(L_ratio):
            self.logger.info("a flowchange candidate at %.3g ignored with L_ratio=%.3g", x0, L_ratio)
            return None, None, popt, special, L_ratio

        if (k > K_LIMIT_SAFE
            or k > K_LIMIT_RISKY and self.outstanding_ratio > K_LIMIT_GUARD_RATIO
            ):
            print("k=%.3g > K_LIMIT, L_ratio=%.3g" % (k, L_ratio))
            if use_custom_sigmoid and k < CUSTOM_K_LIMIT or L_ratio > L_RATIO_CUSTOM_LIMIT:
                from .CustomSigmoid import get_safer_point_impl
                try:
                    x1, y1 = get_safer_point_impl(x, y, ratio, popt, debug=False)
                    use_sigmoid = False
                except AssertionError:
                    # as in Matsumura
                    log_exception(self.logger, "get_safer_point_impl failure: ")
                    use_sigmoid = True
            else:
                use_sigmoid = True

            if use_sigmoid:
                y1 = b + L * ratio
                x1 = ex_sigmoid_inv(y1, L, x0, k, b, s1, s2)
                # x1 can exceed limits as in 20160227
                # print("--------------------- x[0], x1, x[-1]=", x[0], x1, x[-1])

        elif k > K_LIMIT_SPECIAL:
            r_value = linregress(x, y)[2]
            print("r_value=", r_value)
            if r_value > R_VALUE_LIMIT:
                # as in SUB_TRN1
                # getting a safer point is not appropriate in this case
                self.logger.info("this flowchange candidate with k=%.3g is a special case due to r_value=%.3g", k, r_value)
                x1, y1 = x0, ex_sigmoid(x0, *popt)
            else:
                x1, y1 = None, None
        else:
            x1, y1 = None, None
        if x1 is None:
            self.logger.info("a flowchange candidate ignored with k=%.3g", k)

        if x1 is not None:
            assert x[0] <= x1 and x1 <= x[-1]
            special, ratio = self.peak_region.is_special(x1, return_ratio=True)
            if special and SHOW_SPECIAL_CASE_MESSAGE:
                from molass_legacy._MOLASS.SerialSettings import get_setting
                test_pattern = get_setting("test_pattern")
                if test_pattern is None and not self.special_case_warned:
                    try:
                        import molass_legacy.KekLib.CustomMessageBox as MessageBox
                        from molass_legacy._MOLASS.Version import get_version_string
                        parent = plt.get_parent()       # note that enable_debug_plot==1 and this parent it the main dialog
                        if get_version_string().find("_MOLASS 1") >= 0:
                            tool_name = "_MOLASS ver. 2. or later"
                        else:
                            tool_name = '"Full Optimization"'
                        MessageBox.showwarning("Special Flowchange Warning",
                            "A special flow change(*) has been detected in this data set.\n"
                            "Analysis using %s is recommended.\n"
                            'Otherwise, trim manually at "Data Rage".\n\n'
                            "* this point will not appear in figures in this version." % tool_name,
                            parent=parent)
                    except:
                        # ctypes.ArgumentError: argument 1: <class 'OverflowError'>: int too long to convert
                        log_exception(self.logger, "Special Flowchange Warning failure: ")
                    self.special_case_warned = True
                self.logger.info("this flowchange candidate at %.4g is discarded as a special case with ratio=%.3g.", x1, ratio)
                # how about error_ratio in this case

        return x1, y1, popt, special, L_ratio

    def get_raw_flow_changes(self):
        return self.fcj

    def get_flow_changes(self, narrow=False):
        ret_list = []
        for k, fc in enumerate(self.fcj):
            if fc is None:
                fc = 0 if k == 0 else len(self.a_curve2.x) - 1
            else:
                if narrow:
                    margin = int(self.peak_region.get_size()*NARROW_MARGIN_RATIO)   # get_size returns size with UV elution units
                    sign = +1 if k == 0 else -1
                    fc += sign*margin
            ret_list.append(fc)
        return ret_list

    def get_safe_flow_changes(self):
        # used in 'narrow_trimming'
        return self.get_flow_changes(narrow=True)

    def get_real_flow_changes(self):
        # consider unifying with get_raw_flow_changes using 20211021
        flow_changes_src = self.get_flow_changes()
        flow_changes = []
        for fc in flow_changes_src:
            if fc == 0 or fc == len(self.a_curve2.x) - 1:
                fc_ = None
            else:
                fc_ = fc
            flow_changes.append(fc_)
        return flow_changes        # print( 'right_jump_j=', self.right_jump_j )

    def get_mapped_flow_changes(self):
        similarity = self.get_similarity()
        flow_changes = self.get_real_flow_changes()
        return [j if j is None else similarity.inverse_int_value(j) for j in flow_changes]

    def get_similarity(self):
        return self.peak_region.similarity

    def get_solver_info(self, debug=False):
        if debug:
            from importlib import reload
            import Trimming.UvBaseSolverInfo
            reload(Trimming.UvBaseSolverInfo)
        from .UvBaseSolverInfo import make_info_lists
        return make_info_lists(self, debug=debug)

    def remove_irregular_points(self):
        ret_pp = []
        err_pp = []
        size = len(self.a_curve2.y)
        for k in self.pp3:
            if k < size - 10:
                ret_pp.append(k)
            else:
                err_pp.append(k)
        if len(err_pp) > 0:
            ret_slice = slice(None, np.min(err_pp))
        else:
            ret_slice = slice(None, None)
        return np.array(sorted(ret_pp, key=lambda j: -abs(self.gy[j]))), np.array(err_pp), ret_slice
