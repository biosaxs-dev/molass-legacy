"""
    Trimming.FlangeLimit.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import numpy as np
import logging
from bisect import bisect_right
from molass_legacy.KekLib.GeometryUtils import rotated_argmin

MIN_Q_LIMIT             = 0.2       #
MIN_Q_LIMIT_RATIO       = 0.67      # to cover 20161205
ALLOWANCE_WIDTH_RATIO   = 0.01
Q_LIMIT_RATIOS          = [0.99, 0.97, 0.95, 0.93]
OK_RATIO_LIMIT          = 0.15      # > 0.12 for 20190309_1
RETRY_RATIO_LIMIT       = 0.3
MIN_PEAK_NEG_ALLOW      = -0.03     # < -0.01 for SUB_TRN1
MIN_PEAK_POS_LIMIT      = 0.01
MIN_PEAK_WIDTH_Q        = 0.003     # < 0.006 for AhRR
LIMIT_TYPE_FLANGE = 0
LIMIT_TYPE_WA_CORNER = 1
END_FIND_ROTATION       = -np.pi/4

class FlangeLimit:
    def __init__(self, data, error, e_curve, q_vector):
        self.logger = logging.getLogger( __name__ )
        self.data = data
        self.error = error
        self.e_curve = e_curve
        self.q_vector = q_vector
        self.limit = None
        self.limit_type = None

    def get_limit(self, debug=False):
        if self.limit is None:
            try:
                self.get_limit_impl(self.e_curve, self.data, self.error, self.logger, debug=debug)
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                self.logger.warning("get_limit_impl failed." + etb.last_lines(5))
        return self.limit

    def get_limit_impl(self, e_curve, data, error, logger, debug=False):
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            
        # self.q_vector[-1] == 0.194 in 20171203
        i = min(int(len(self.q_vector)*MIN_Q_LIMIT_RATIO), bisect_right(self.q_vector, MIN_Q_LIMIT))
        j = e_curve.get_primarypeak_i()
        w = int(data.shape[0]*ALLOWANCE_WIDTH_RATIO)
        hw = w//2
        jslice = slice(j-hw, j+hw)
        avg_yd = np.average(data[:,jslice], axis=1)
        avg_yd[avg_yd <= 0] = np.nan        # to avoid np.log10() invalid value warning 
        log_yd = np.log10(avg_yd)
        error_ = error[:,jslice].copy()

        zeros_fix = False
        if error_[-1,0] == 0:
            # error data bug fix
            error_[-1,:] = error_[-2,:]
            zeros_fix = True
        elif error_[0,0] == 0:
            error_[0,:] = error_[1,:]
            zeros_fix = True

        if zeros_fix:
            self.logger.warning("zeros at the angular ends of error data have been replaced with the adjacent values.")

        log_ye = np.log10(np.average(error_, axis=1))
        log_yeg = np.gradient(log_ye)
        gy = log_yeg[i:]
        agy = np.abs(gy)

        hw_ = None
        def find_limit_cadidate(limit_ratio):
            nonlocal hw_
            k = int(len(agy)*limit_ratio)
            pp = np.argpartition(agy, k)[k:]
            f = np.min(pp)
            if gy[f] > 0:
                hw_ = hw
            else:
                # search in wider range as in 20170301
                hw_ = min(f//2, hw*10)      # so that f-hw_ > 0 for 20190317
            neighbor = gy[f-hw_:f+hw_]
            assert len(neighbor) > 0
            m = np.argmax(neighbor)
            f = f - hw_ + m  # change f to the neighbor max
            safe_f = f - w
            return pp, f, safe_f

        def has_a_peak_of_some_width(sf):
            y = log_ye[sf:] - log_ye[sf]
            wneg = np.where(y < MIN_PEAK_NEG_ALLOW)[0]
            if debug:
                print('y=', y[:10])
                plt.push()
                fig, ax = plt.subplots()
                ax.plot(y)
                ax.plot(wneg, y[wneg], 'o', color='yellow')
                plt.show()
                plt.pop()
            if len(wneg) > 0:
                y_ = y[0:wneg[0]]
                wpos = np.where(y_ > MIN_PEAK_POS_LIMIT)[0]
                if len(wpos) > 0:
                    qw = self.q_vector[sf+wpos[-1]] - self.q_vector[sf+wpos[0]]
                    # print('y=', y[0:10], 'wpos=', wpos[0:10], 'qw=', qw)
                    ret_judge = qw > MIN_PEAK_WIDTH_Q
                else:
                    qw = 0
                    ret_judge = False
            else:
                ret_judge = True
            if not ret_judge:
                self.logger.info("flange limit candidate has been discarded due to peak width of %.3g at Q=%.3g", qw, self.q_vector[sf])
            return ret_judge

        for k, q_limit_ratio in enumerate(Q_LIMIT_RATIOS):
            try:
                pp, f, safe_f = find_limit_cadidate(q_limit_ratio)
            except AssertionError:
                # as in 20200121_1
                continue

            check_ratio = np.average(agy[safe_f-w:safe_f])/agy[f]
            is_limit = np.isfinite(agy[f]) and check_ratio < OK_RATIO_LIMIT and has_a_peak_of_some_width(i+safe_f)
            if debug:
                # agy[f] can be inf as in 20170226/Sugiyama
                print([k], (f, self.q_vector[f]), is_limit, check_ratio, agy[safe_f-hw:safe_f+hw], agy[f-2:f+3])
            if is_limit or check_ratio > RETRY_RATIO_LIMIT:
                break
            else:
                # as in HIF
                continue

        fmin = np.argmin(log_ye[i:])
        if not is_limit or fmin < safe_f:
            # fmin < safe_f as in 0171203
            wy = log_ye[i+fmin:]
            if len(wy) > 1:
                m = rotated_argmin(END_FIND_ROTATION, wy, debug)
                safe_f = fmin + m
            else:
                safe_f = fmin
            is_limit = True
            self.limit_type = LIMIT_TYPE_WA_CORNER
            self.logger.info("wide angle limit has been determined as the corner of rightmost error-upturn.")
        else:
            self.limit_type = LIMIT_TYPE_FLANGE

        f_ = i + f
        safe_f_ = i + safe_f

        if debug:
            print('pp=', pp)
            print('check_ratio=', check_ratio)
            from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
            limit_type = "Flange Limit" if self.limit_type == LIMIT_TYPE_FLANGE else "Wide Angle Limit"

            in_folder = get_in_folder()
            plt.push()
            fig = plt.figure(figsize=(21, 7))
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
            axes = [ax1, ax2, ax3]

            fig.suptitle(limit_type + " Debug for " + in_folder, fontsize=20)
            ax1.set_title("Log10( intensity )", fontsize=16)
            ax2.set_title("Log10( error )", fontsize=16)
            ax3.set_title("Gradient of Log10( error )", fontsize=16)

            ax1.plot(self.q_vector, log_yd)
            ax2.plot(self.q_vector, log_ye)
            ax3.plot(self.q_vector, log_yeg)
            ax3.plot(self.q_vector[i+pp], log_yeg[i+pp], 'o', color='yellow')
            fx, fy = self.q_vector[f_], log_yeg[f_]
            ax3.plot(fx, fy, 'o', color='pink')
            xmin, xmax = ax3.get_xlim()
            w = xmax - xmin
            ymin, ymax = ax3.get_ylim()
            h = ymax - ymin

            h_ = h*0.2
            for x_ in [f-hw_, f+hw_]:
                ix_ = min(i+x_, len(self.q_vector)-1)
                ax3.plot(self.q_vector[[ix_, ix_]], [-h_, +h_], ':', color='pink')

            if is_limit:
                sfx, sfy = self.q_vector[safe_f_], log_yeg[safe_f_]
                ax3.plot(sfx, sfy, 'o', color='red')
                ax3.annotate(limit_type, xy=(sfx, sfy), xytext=(sfx-0.1*w, sfy+0.2*h), ha="right", arrowprops=dict(arrowstyle="-", color='k'))
                for ax in axes[0:2]:
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim()
                    ax.plot([sfx, sfx], [ymin, ymax], ':', color='red', label=limit_type)
                    ax.legend()

            else:
                ax3.annotate("not a Flange Limit", xy=(fx, fy), xytext=(fx, fy+0.2*h), ha="center", arrowprops=dict(arrowstyle="-", color='k'))

            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()
            plt.pop()

        if is_limit:
            self.limit = safe_f_
