"""
    AutoRestrictor.py

    Copyright (c) 2020-2024, SAXS Team, KEK-PF
"""
import logging
from bisect import bisect_right
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import LSQUnivariateSpline
from molass_legacy.Elution.CurveUtils import simple_plot
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy.Models.ElutionCurveModels import EGH
from molass_legacy.QuickAnalysis.ParammedModel import ParammedModel
from .MatchingPeaks import MatchingPeaks
from .TrimmingInfo import TrimmingInfo

SIGMA_POINT_RATIO = 10
MINOR_SIGMA_RATIO = 5
RECON_SIGMA_RATIO = 7
RECON_MINOR_SIGMA_RATIO = 3
EXTEND_LIMIT = 0.05
ACCEPTABLE_Y_RATIO = 0.05
N_SIGMAS_UNRELIABLE = 2.5   # chosen to modify so that 20191006_proteins5 be ok
MAX_SIGMA_WIDTH_RATIO = 0.7

def get_exact_emg_peaks_list(uv_curve, xr_curve, debug=False):
    # set_setting("local_debug", True)
    ret_uv_peaks = uv_curve.get_emg_peaks(debug=debug)
    # set_setting("local_debug", False)
    ret_xr_peaks = xr_curve.get_emg_peaks(debug=debug)

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("get_exact_emg_peaks_list")

            simple_plot(ax1, uv_curve, color="blue", legend=False)
            x = uv_curve.x
            for epeak in ret_uv_peaks:
                y = epeak.get_model_y(x)
                ax1.plot(x, y, ":")

            simple_plot(ax2, xr_curve, color="orange", legend=False)
            x = xr_curve.x
            for epeak in ret_xr_peaks:
                y = epeak.get_model_y(x)
                ax2.plot(x, y, ":")

            for ax in ax1, ax2:
                ax.legend()
            fig.tight_layout()
            plt.show()

    return ret_uv_peaks, ret_xr_peaks

class PeakMathcingError(Exception):
    pass


class AutoRestrictor:
    def __init__(self, sd, cs):
        self.logger = logging.getLogger()
        self.sd = sd
        self.cs = cs
        self.extended_flags = np.zeros((2, 2), dtype=int)
        self.mpeaks = None
        """
                        0 : left,  1 : right
            0 : UV
            1 : Xray
        """

    def get_better_restriction(self, uv_restrict_list=None, xr_restrict_list=None, debug=False, figfile=None):
        try:
            return self.get_better_restriction_impl(uv_restrict_list, xr_restrict_list, debug=debug, figfile=figfile)
        except:
            # as in 20181203 with PEAK_INFO_FIND_RATIO=0.3
            log_exception(self.logger, "get_better_restriction failed: ")

            if debug:
                from .TrimmingDebugUtils import trimming_debug_plot
                trimming_debug_plot("get_better_restriction_impl except debug", self.sd, uv_restrict_list, xr_restrict_list)

            return uv_restrict_list, xr_restrict_list

    def get_better_restriction_impl(self, uv_restrict_list=None, xr_restrict_list=None, debug=False, figfile=None):
        """
        TODO:
            unify *_a_restrict setting in this method
            from molass_legacy.Trimming.PreliminaryRecognition.set_restrict_info
        """

        if uv_restrict_list is None:
            uv_e_restrict, uv_a_restrict = None, None
        else:
            uv_e_restrict, uv_a_restrict = uv_restrict_list
        if xr_restrict_list is None:
            xr_e_restrict, xr_a_restrict = None, None
        else:
            xr_e_restrict, xr_a_restrict = xr_restrict_list

        uv_wl_lower_bound = get_setting("uv_wl_lower_bound")
        assert uv_wl_lower_bound is not None
        wlvector = self.sd.lvector
        start = bisect_right(wlvector, uv_wl_lower_bound)
        if uv_a_restrict is None:
                size = len(wlvector)
                uv_a_restrict = TrimmingInfo(1, start, size, size)
        else:
            self.logger.info("taling max(%d, %d)", uv_a_restrict.start, start)
            uv_a_restrict.start = max(uv_a_restrict.start, start)

        self.logger.info("uv_wl_lower_bound=%g, uv_a_restrict=%s", uv_wl_lower_bound, str(uv_a_restrict))
        sd = self.sd
        cs = self.cs
        # uv_curve = sd.get_uv_curve()
        # xr_curve = sd.get_xray_curve()
        uv_curve = cs.a_curve
        xr_curve = cs.x_curve

        try:
            # uv_peaks, xr_peaks = self.get_major_emg_peaks_list(uv_curve, xr_curve, debug=debug)
            uv_peaks, xr_peaks = get_exact_emg_peaks_list(uv_curve, xr_curve, debug=False)

            new_xr_restrict = self.get_sigma_restriction(xr_curve, xr_peaks, xr_e_restrict, debug=debug)
            mapped_points, maping_info = self.get_mapped_points(new_xr_restrict, xr_curve, xr_peaks, uv_curve, uv_peaks, debug=debug)
            new_uv_restrict = self.get_sigma_restriction(uv_curve, uv_peaks, uv_e_restrict, mapped_points=mapped_points, op_restrict=new_xr_restrict, debug=debug)
            new_xr_restrict_save = new_xr_restrict
            new_xr_restrict = self.map_back_to_xray(maping_info, new_uv_restrict, new_xr_restrict)

            if debug:
                import molass_legacy.KekLib.DebugPlot as plt
                def plot_range(ax, restrict, color):
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(ymin, ymax)
                    for j in (restrict.start, restrict.stop):
                        ax.plot([j, j], [ymin, ymax], color=color)

                with plt.Dp():
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                    fig.suptitle("get_better_restriction_impl: debug")
                    simple_plot(ax1, uv_curve, color="blue")
                    simple_plot(ax2, xr_curve, color="orange")
                    plot_range(ax1, new_uv_restrict, "yellow")
                    plot_range(ax2, new_xr_restrict_save, "cyan")
                    plot_range(ax2, new_xr_restrict, "yellow")
                    fig.tight_layout()
                    plt.show()

            new_e_restricts = [new_uv_restrict, new_xr_restrict]
        except:
            log_exception(self.logger, "get_better_restriction_impl error: ")
            new_e_restricts = self.get_them_roughly((uv_curve, uv_e_restrict), (xr_curve, xr_e_restrict), debug=debug)

        ret_info = [[new_e_restricts[0], uv_a_restrict], [new_e_restricts[1], xr_a_restrict]]

        curves = uv_curve, xr_curve
        emg_peaks_list = uv_peaks, xr_peaks
        old_e_restricts = uv_e_restrict, xr_e_restrict
        self.result_info = sd, curves, emg_peaks_list, old_e_restricts, ret_info, SIGMA_POINT_RATIO, self.mpeaks

        if debug or figfile is not None:
            from molass_legacy.Trimming.TrimmingDebugUtils import trimming_result_plot
            trimming_result_plot(*self.result_info, figfile=figfile)

        if (self.extended_flags[0,:] != self.extended_flags[1,:]).any():
            ret_info = self.correct_inconsistent_extensions(ret_info)

        return ret_info

    def get_them_roughly(self, uv_info, xr_info, debug=False):
        from molass_legacy.Peaks.RobustPeaks import RobustPeaks
        from molass_legacy.Trimming import get_mapped_info, get_wider_info

        ret_e_restricts = []
        for curve, e_restrict in [uv_info, xr_info]:
            x = curve.x
            y = curve.y
            rp = RobustPeaks(x, y)
            sigma_min, sigma_max = rp.get_limits_roughly(SIGMA_POINT_RATIO)

            size = len(x)
            if e_restrict is None or not e_restrict.flag:
                xmin, xmax = 0, size    # or size-1 ?
            else:
                xmin, xmax = e_restrict.start, e_restrict.stop

            ret_e_restricts.append(TrimmingInfo(1, max(xmin, sigma_min), min(xmax, sigma_max), size))

        A = self.cs.slope
        B = self.cs.intercept
        mapped_xr_info = get_mapped_info((A, B), ret_e_restricts[0].size, ret_e_restricts[1])
        ret_uv_info = get_wider_info(ret_e_restricts[0], mapped_xr_info)
        mapped_uv_info = get_mapped_info((1/A, -B/A), ret_e_restricts[1].size, ret_uv_info)
        ret_xr_info = get_wider_info(ret_e_restricts[1], mapped_uv_info)
        ret_e_restricts = [ret_uv_info, ret_xr_info]

        self.logger.info("elution trimming infos have been determined roughly due to a peak matching trouble.")

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, axes = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("get_them_roughly debug")

                def draw_trimming(ax, ylim, tinfo, color):
                    if tinfo is None:
                        return
                    for j in tinfo.start, tinfo.stop:
                        ax.plot([j, j], ylim, color=color)

                for ax, in_info, ret_info in zip(axes, [uv_info, xr_info], ret_e_restricts):
                    curve, e_restrict = in_info
                    simple_plot(ax, curve)
                    ylim = ax.get_ylim()
                    ax.set_ylim(ylim)
                    draw_trimming(ax, ylim, e_restrict, "cyan")
                    draw_trimming(ax, ylim, ret_info, "yellow")
                fig.tight_layout()
                plt.show()

        return ret_e_restricts

    def get_uv_sigma_points(self):
        return self.uv_sigma_info.start, self.uv_sigma_info.stop

    def get_major_emg_peaks_list(self, uv_curve, xr_curve, debug=False):
        self.mpeaks = MatchingPeaks(self.cs, uv_curve, xr_curve, debug=debug)
        uv_emg_peaks, xr_emg_peaks = self.mpeaks.get_matching_peak_lists()

        uv_acc_y = uv_curve.max_y * ACCEPTABLE_Y_RATIO
        xr_acc_y = xr_curve.max_y * ACCEPTABLE_Y_RATIO

        ret_uv_peaks = []
        ret_xr_peaks = []
        for uv_peak, xr_peak in zip(uv_emg_peaks, xr_emg_peaks):
            if uv_peak.top_y >= uv_acc_y or xr_peak.top_y >= xr_acc_y:
                ret_uv_peaks.append(uv_peak)
                ret_xr_peaks.append(xr_peak)

        return ret_uv_peaks, ret_xr_peaks

    def get_mapped_points(self, restrict_info, xr_curve, xr_peaks, uv_curve, uv_peaks, debug=False):
        slope = self.cs.slope
        intercept = self.cs.intercept

        def mapped_pos(k, p):
            if p is None:
                i = 0 if k == 0 else -1
                p = xr_curve.x[i]
            return int(round(p*slope + intercept))

        mapped_points = [mapped_pos(k, p) for k, p in enumerate([restrict_info.start, restrict_info.stop])]
        return mapped_points, (slope, intercept)

    def get_sigma_restriction(self, curve, emg_peaks, e_restrict,
            mapped_points=None,
            op_restrict=None,       # not used currently
            debug=False):

        x = curve.x
        size = len(x)

        num_peaks = len(emg_peaks)
        if num_peaks == 1:
            sigma_min, sigma_max = self.get_simple_sigma_points(curve, emg_peaks[0], debug=debug)
            info_L = (emg_peaks[0], sigma_min)
            info_R = (emg_peaks[0], sigma_max)
        else:
            sigma_min = self.get_simple_sigma_points(curve, emg_peaks[0], debug=debug)[0]
            sigma_max = self.get_simple_sigma_points(curve, emg_peaks[-1], debug=debug)[-1]
            info_L = (emg_peaks[0], sigma_min)
            info_R = (emg_peaks[-1], sigma_max)

        if mapped_points is not None:
            s_min, s_max = mapped_points
            sigma_min = min(sigma_min, s_min)
            sigma_max = max(sigma_max, s_max)

            if not self.cs.reliable:
                # consider to do this always
                sigma_min, sigma_max = self.modify_limits_using_moments(curve, sigma_min, sigma_max)

        sigma_info = Struct(jmin=sigma_min, jmax=sigma_max)

        if e_restrict is None or not e_restrict.flag:
            xmin, xmax = 0, size    # or size-1 ?
        else:
            xmin, xmax = e_restrict.start, e_restrict.stop

        trim_min = max(xmin, sigma_min)
        trim_max = min(xmax, sigma_max)

        try:
            recon_info = self.reconsider_using_lower_peak_model(curve, emg_peaks, trim_min, trim_max, debug=debug)
        except:
            log_exception(self.logger, "reconsider_using_lower_peak_model failed: ")
            recon_info = None

        if recon_info is None:
            final_min = trim_min
            final_max = trim_max
        else:
            final_min = max(xmin, min(sigma_min, recon_info.jmin))
            final_max = min(xmax, max(sigma_max, recon_info.jmax))

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_sigma_restriction debug")

                simple_plot(ax, curve, legend=False)
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)

                for x in [final_min, final_max]:
                    ax.plot([x, x], [ymin, ymax], color="yellow", lw=2, label="final")

                if mapped_points is not None:
                    for x in mapped_points:
                        ax.plot([x, x], [ymin, ymax], ":", color="cyan", lw=2, label="mapped")

                ax.legend()
                fig.tight_layout()
                plt.show()

        extra = Struct(info_L=info_L, info_R=info_R, mapped_points=mapped_points, num_peaks=num_peaks,
                        sigma_info=sigma_info, recon_info=recon_info)

        return TrimmingInfo(1, final_min, final_max, size, extra=extra)

    def get_simple_sigma_points(self, curve, emgpeak, debug=False):
        return self.get_simple_sigma_points_impl(curve, emgpeak.opt_params, emgpeak.top_x, debug=debug)

    def get_simple_sigma_points_impl(self, curve, params, peak_top_x, force_ratio=None, debug=False):
        if debug:
            from molass_legacy.KekLib.DebugUtils import show_call_stack
            show_call_stack("----: ", indented_only=True)

        sigma = params[2]
        tau = params[3]

        if force_ratio is None:
            y = curve.spline(peak_top_x)
            if y > curve.max_y*0.1:
                ratio = SIGMA_POINT_RATIO
            else:
                ratio = MINOR_SIGMA_RATIO
        else:
            ratio = force_ratio

        # width = ratio*max(sigma, -tau)
        width = ratio*sigma     # tau < 0 is not valid according to stochastic theory
        f = max(curve.x[0], peak_top_x - width)
        width = ratio*max(sigma, tau)
        t = min(curve.x[-1], peak_top_x + width)

        width_ratio = width/len(curve.x)
        if width_ratio < MAX_SIGMA_WIDTH_RATIO:
            pass
        else:
            # as in SUB_TRN1
            raise AssertionError("width_ratio=%.3g >= MAX_SIGMA_WIDTH_RATIO" % width_ratio)

        ret_points = [int(round(v)) for v in (f, t)]
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Peaks.ElutionModels import egh
            print("sigma=", sigma, "tau=", tau, "ratio=", ratio)
            print("width=", width)
            x = curve.x
            h = params[0]
            mu = params[1]
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("get_simple_sigma_points_impl")
                simple_plot(ax, curve, legend=False)
                ax.plot(x, egh(x, h, mu, sigma, tau), ":", label="model")
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                for j in ret_points:
                    ax.plot([j, j], [ymin, ymax], color="yellow")
                ax.legend()
                fig.tight_layout()
                plt.show()

        return ret_points

    def proof_plot(self, better_ret, pause=True, savefig=False, figfile=None):
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
        plt.push()
        # uv_curve = self.sd.get_uv_curve()
        # xr_curve = self.sd.get_xray_curve()
        uv_curve = self.cs.a_curve
        xr_curve = self.cs.x_curve
        uv_ret, xr_ret = better_ret
        uv_restrict = None if uv_ret is None else uv_ret[0]
        xr_restrict = None if xr_ret is None else xr_ret[0]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
        ax1, ax2 = axes
        fig.suptitle("AutoRestrictor Proof for %s" % get_in_folder(), fontsize=16)

        ax1.plot(uv_curve.y)
        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax)
        if uv_restrict is not None:
            for x in [uv_restrict.start, uv_restrict.stop]:
                ax1.plot([x,x], [ymin, ymax], color='yellow')

        for rec in uv_curve.peak_info:
            x = rec[1]
            ax1.plot(x, uv_curve.spline(x), 'o', color='pink')

        for peak in uv_curve.get_emg_peaks():
            ax1.plot(peak.top_x, peak.top_y, 'o', color='red')

        ax2.plot(xr_curve.y)
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax)
        if xr_restrict is not None:
            for x in [xr_restrict.start, xr_restrict.stop]:
                ax2.plot([x,x], [ymin, ymax], color='yellow')

        for rec in xr_curve.peak_info:
            x = rec[1]
            ax2.plot(x, xr_curve.spline(x), 'o', color='pink')

        for peak in xr_curve.get_emg_peaks():
            ax2.plot(peak.top_x, peak.top_y, 'o', color='red')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        if pause:
            plt.show()
        else:
            from time import sleep
            plt.show(block=False)
            sleep(0.5)
        if savefig:
            import os
            from molass_legacy._MOLASS.SerialSettings import get_setting
            temp_folder = get_setting('temp_folder')
            path = os.path.join(temp_folder, figfile)
            fig.savefig(path)
        plt.pop()

    def correct_inconsistent_extensions(self, ret_info):
        flags = self.extended_flags
        self.logger.info("correcting the inconsistent extensions detected: %s.", str(list(flags)))
        lr_id = 0
        for uv_flag, xr_flag in zip(flags[0,:], flags[1,:]):
            if uv_flag == xr_flag:
                pass
            else:
                uv_xr_id =  uv_flag
                self.logger.info("extend also %s %s.", ('Xray' if uv_xr_id else 'UV'), ('right' if lr_id else 'left'))
                ret_info[uv_xr_id] = self.set_correponding_point(uv_xr_id, lr_id, ret_info[uv_xr_id], ret_info[1 - uv_xr_id])
            lr_id += 1
        return ret_info

    def set_correponding_point(self, uv_xr_id, lr_id, info_to, info_from):
        if uv_xr_id:
            # correcting Xray from molass_legacy.UV
            # UV => Xray
            A = 1/self.cs.slope
            B = - self.cs.intercept/self.cs.slope
        else:
            # correcting UV from Xray
            # Xray => UV
            A = self.cs.slope
            B = self.cs.intercept

        """
        print('info_to=', info_to)
        print('info_from=', info_from)
        info_to= [Info(1, 0, 804, 805), None]
        info_from= [Info(1, 6, 401, 402), None]
        """

        k = lr_id + 1
        from_value = info_from[0][k]
        to_value = info_to[0][k]
        to_max = info_to[0][-1]
        set_value = max(0, min(to_max-1, int(A*from_value + B)))
        self.logger.info("correcting %d to %d by %d.", to_value, set_value, from_value)
        set_list_ = list(info_to[0])
        set_list_[k] = set_value
        info_to[0] = TrimmingInfo(*set_list_)
        return info_to

    def modify_limits_using_moments(self, curve, xmin, xmax):
        try:
            mean, variance = curve.compute_moments()
            assert variance > 0
            sigma = np.sqrt(variance)
            print('modify_limits_using_moments: mean=', mean, 'sigma=', sigma)
            ret_xmin = int(min(mean - N_SIGMAS_UNRELIABLE*sigma, xmin))
            ret_xmax = int(max(mean + N_SIGMAS_UNRELIABLE*sigma, xmax))
            if ret_xmin < xmin or ret_xmax > xmax:
                self.logger.info("modified sigma limits from %s to %s due to unreliable mapping.", str((xmin, xmax)), str((ret_xmin, ret_xmax)))
        except:
            ret_xmin = xmin
            ret_xmax = xmax
        return ret_xmin, ret_xmax

    def reconsider_using_lower_peak_model(self, curve, emg_peaks, trim_min, trim_max, debug=False):
        num_peaks = len(emg_peaks)
        if num_peaks > 1:
            # currently support only when num_peaks == 1
            return None

        main_emg_peak = emg_peaks[0]
        eslice = slice(trim_min, trim_max)
        x = curve.x[eslice]
        y = curve.y[eslice]
        low_part = y < curve.max_y * 0.25
        x_ = x[low_part]
        y_ = y[low_part]
        knots = np.linspace(x[1], x[-2], 10)
        spline = LSQUnivariateSpline(x_, y_, knots)
        sy = spline(x)

        model = EGH()
        params = model.guess(sy, x=x)
        out = model.fit(sy, params, x=x)
        fy = model.eval(out.params, x=x)

        peak_top_x = main_emg_peak.top_x

        orig_x = curve.x
        orig_y = curve.y

        curve_proxy = Struct(x=orig_x, max_y=curve.max_y, spline=spline)

        try:
            recon_min, recon_max = self.get_simple_sigma_points_impl(curve_proxy, out.params, peak_top_x,
                                                                        force_ratio=RECON_SIGMA_RATIO)
        except:
            log_exception(self.logger, "get_simple_sigma_points_impl: ")
            raise RuntimeError("get_simple_sigma_points_impl failure")

        prm_model = ParammedModel([(model, out.params)])
        recon_info = Struct(jmin=recon_min, jmax=recon_max, prm_model=prm_model)
        recon_info = self.try_adding_minor_peak(curve, main_emg_peak, recon_info)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            plt.push()
            fig, ax = plt.subplots()
            ax.set_title("reconsider_using_lower_peak_model debug")
            ax.plot(orig_x, orig_y)
            ax.plot(x_, y_)
            ax.plot(x, sy)
            ax.plot(x, fy)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for px in [recon_min, recon_max]:
                ax.plot([px, px], [ymin, ymax], ":", color="red")
            plt.show()
            plt.pop()

        return recon_info

    def try_adding_minor_peak(self, curve, main_emg_peak, recon_info):
        peak_top_x = main_emg_peak.top_x
        candidates = []
        for k, info in enumerate(curve.peak_info):
            ptx = info[1]
            if ptx < peak_top_x:
                candidates.append(info)

        if len(candidates) > 0:
            # select the tallest minor peak
            candidate = sorted(candidates, key=lambda info: -curve.y[info[1]])[0]
            eslice = slice(candidate[0], candidate[2])
            x_ = curve.x[eslice]
            y_ = curve.y[eslice]
            model = EGH()
            params = model.guess(y_, x=x_)
            out = model.fit(y_, params, x=x_)
            fy = model.eval(out.params, x=x_)

            jmin = max(0, int(round(candidate[1] - abs(out.params["tau"]) * RECON_MINOR_SIGMA_RATIO)))

            if False:
                import molass_legacy.KekLib.DebugPlot as plt
                from molass_legacy.Elution.CurveUtils import simple_plot
                plt.push()
                fig, ax = plt.subplots()
                ax.set_title("try_adding_minor_peak debug")
                simple_plot(ax, curve)
                ax.plot(x_, fy)
                ax.plot([jmin], model.eval(out.params, x=[jmin]), "o", color="yellow")
                plt.show()
                plt.pop()

            prm_model = recon_info.prm_model
            prm_model.append((model, out.params))
            ret_info = Struct(jmin=jmin, jmax=recon_info.jmax, prm_model=prm_model)
        else:
            ret_info = recon_info

        return ret_info

    def map_back_to_xray(self, maping_info, uv_restrict, xr_restrict):
        a, b = maping_info
        # j = a*i + b
        # i = (j - b)/a
        jmin, jmax = [ int(round((j - b)/a)) for j in [uv_restrict.start, uv_restrict.stop]]

        size = xr_restrict.size
        extra = xr_restrict.extra
        return TrimmingInfo(1, max(0, jmin), min(size, jmax), size, extra=extra)
