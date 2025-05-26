"""
    Trimming.CdLimit.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
import numpy as np
import logging
from bisect import bisect_right
from scipy.optimize import minimize
from molass_legacy.KekLib.GeometryUtils import rotated_argmin
from molass_legacy._MOLASS.SerialSettings import get_setting

NUM_Q_POINTS = 10
CD_LIMIT_NOFIT_RATIO    = 0.02
CD_SCORE_OK_LIMIT       = 1.0
CD_LIMIT_FIND_ROTATION  = -np.pi/6

class CdLimit:
    def __init__(self, data, error, e_curve, qvector):
        self.logger = logging.getLogger( __name__ )
        self.data = data
        self.e_curve = e_curve
        self.qidx = bisect_right(qvector, 0.02)
        self.qindex = bisect_right(qvector, get_setting('cd_eval_qmax'))
        print('qindex=', self.qindex)
        self.aslice = slice(0, self.qindex)
        self.x = qvector[self.aslice]

    def get_limit(self, smooth_bq=True, debug=False):
        limit, surely = self.get_limit_impl(NUM_Q_POINTS, smooth_bq, debug=debug)
        if limit == NUM_Q_POINTS - 1:
            self.logger.info("retry due to an end limit.")
            limit, surely = self.get_limit_impl(NUM_Q_POINTS*2, smooth_bq, retry_argmin=True, debug=debug)
        return limit, surely

    def get_limit_impl(self, num_points, smooth_bq, retry_argmin=False, debug=False):
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt

        start = max(1, int(len(self.x)*CD_LIMIT_NOFIT_RATIO))

        e_curve = self.e_curve
        pno = e_curve.primary_peak_no
        rec = e_curve.peak_info[pno]
        f, p, t = rec
        eslice = slice(p,t+1)
        c = e_curve.y[eslice]
        c = c/np.max(c)
        C = np.array([c, c**2])
        Cpinv = np.linalg.pinv(C)
        ok_info_list = []
        init_scales = []
        scores = []
        scales = []
        ab_pairs = []
        rank = 2
        for i in range(num_points):
            aslice = slice(i,self.qindex)
            M = self.data[aslice,eslice]
            U, s, VT = np.linalg.svd(M)
            M_ = U[:,0:rank] @ np.diag(s[0:rank]) @ VT[0:rank,:]
            P = M_ @ Cpinv
            a = P[:,0]
            b = P[:,1]
            ab_pairs.append((a,b))

            def min_norm_diff_bq_scaled_aq(pv):
                return np.sum((b - pv[0]*a)**2)

            init_scale = s[1]/s[0]
            ret = minimize(min_norm_diff_bq_scaled_aq, (init_scale,))
            scale = ret.x[0]
            score = np.sqrt(np.average((b - scale*a)**2))*100/a[self.qidx]
            init_scales.append(init_scale)
            scales.append(scale)
            scores.append(score)
            if score < CD_SCORE_OK_LIMIT:
                ok_info_list.append((i, score))

        if len(ok_info_list) > 0:
            limit = rotated_argmin(CD_LIMIT_FIND_ROTATION, scores, debug)
            if retry_argmin:
                if limit == num_points - 1:
                    m = np.argmin(scores[0:-1])     # excluding the end point
                    self.logger.info("the end limit %d has been replaced by no rotation argmin %d.", limit, m)
                    limit = m
            surely = False
        else:
            limit = 0
            ii = np.argmax(init_scales)
            il = np.argmin(scales)
            surely = ii == 0 and il == 0

        if debug:
            from matplotlib.gridspec import GridSpec
            from molass_legacy.KekLib.SciPyCookbook import smooth
            from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

            plt.push()
            rsize = 4
            ncols = 5
            nrows = num_points//ncols
            fig = plt.figure(figsize=(23,12))
            gs = GridSpec(nrows*rsize, ncols)
            main_axes = []
            diff_axes = []
            for m in range(nrows):
                main_row = []
                diff_row = []
                m_ = m*rsize
                for n in range(ncols):
                    main_ax = fig.add_subplot(gs[m_:m_+3,n])
                    diff_ax = fig.add_subplot(gs[m_+3,n])
                    main_row.append(main_ax)
                    diff_row.append(diff_ax)
                main_axes.append(main_row)
                diff_axes.append(diff_row)

            main_axes = np.array(main_axes)
            diff_axes = np.array(diff_axes)

            fig.suptitle("A(q), B(q) Debug Plots for " + get_in_folder(), fontsize=20)

            def draw_score_text(ax, score, color, dy=0):
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                tx = (xmin + xmax)/2
                ty = (ymin + ymax)/2 + (ymax-ymin)*dy
                score_text = 'SCD=%.3g' % score
                ax.text(tx, ty, score_text, color=color, alpha=0.3, fontsize=40, ha='center', va='center')

            smooth_scales = []
            smooth_scores = []

            smooth_bq=True

            for i in range(num_points):
                a, b = ab_pairs[i]
                scale = scales[i]
                score = scores[i]
                m, n = divmod(i,ncols)
                ax = main_axes[m,n]
                x = self.x[i:]
                ax.grid(False)
                ax.get_xaxis().set_visible(False)
                ax.plot(x, a, color='C1')
                axt = ax.twinx()
                axt.plot(x, b, color='pink', label='B(q)')
                if smooth_bq:
                    sb = smooth(b)
                    axt.plot(x, sb, color='cyan', label='smooth B(q)')
                axt.plot(x, scale*a, color='blue', label='scale*A(q)')
                axt.legend()
                dy = 0.1 if smooth_bq else 0
                draw_score_text(ax, score, 'pink', dy=0.1)
                if smooth_bq:
                    def min_norm_diff_sbq_scaled_aq(pv):
                        return np.sum((sb - pv[0]*a)**2)
                    init_scale = init_scales[i]
                    ret = minimize(min_norm_diff_sbq_scaled_aq, (init_scale,))
                    smooth_scale = ret.x[0]
                    smooth_scales.append(smooth_scale)
                    sy = sb - smooth_scale*a
                    smooth_score = np.sqrt(np.average((sb - smooth_scale*a)**2))*100/a[self.qidx]
                    smooth_scores.append(smooth_score)
                    draw_score_text(ax, smooth_score, 'cyan', dy=-0.1)

                axd = diff_axes[m,n]
                y = b - scale*a
                axd.plot(x, y, color='gray', label='B(q) - scale*A(q)')
                if smooth_bq:
                    axd.plot(x, sy, color='cyan', label='smooth B(q) - scale*A(q)')

                axd.legend(bbox_to_anchor=(1, 1), loc='lower right')

            fig.tight_layout()
            fig.subplots_adjust(top=0.92)
            plt.show()
            plt.pop()

            a, b = ab_pairs[limit]
            scale = scales[limit]
            print('limit=', limit)
            plt.push()
            gs = GridSpec(4, 3)
            fig = plt.figure(figsize=(23,8))
            ax1 = fig.add_subplot(gs[0:2,0])
            ax2 = fig.add_subplot(gs[2:4,0])
            ax3 = fig.add_subplot(gs[0:3,1])
            ax3.get_xaxis().set_visible(False)
            ax3.grid(False)
            ax3d = fig.add_subplot(gs[3,1])
            ax4 = fig.add_subplot(gs[0:3,2])
            ax4.grid(False)
            ax4.get_xaxis().set_visible(False)
            ax4d = fig.add_subplot(gs[3,2])

            fig.suptitle("B(q) Limit Debug Plot for " + get_in_folder(), fontsize=20)
            ax1.set_title("Scales", fontsize=16)
            ax1.plot(init_scales, label='init scales')
            ax1.plot(scales, label='min norm scales')
            ax2.set_title("Scores", fontsize=16)
            if len(scores) > NUM_Q_POINTS:
                xticks = np.arange(0,len(scores), 2)
                for ax in [ax1, ax2]:
                    ax.set_xticks(xticks)
            ax2.plot(scores, label='scores')
            if smooth_bq:
                ax2.plot(smooth_scores, color='cyan', label='smooth scores')

            for ax in [ax1, ax2]:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                ax.plot([limit, limit], [ymin, ymax], ':', color='red', label='angular range start')
                ax.legend()

            x = self.x[limit:]
            ax3.set_title("B(q) - scale*A(q)", fontsize=16)
            ax3.plot(x, a, color='C1', label='A(q)')
            ax3.legend(loc='center right')
            ax3t = ax3.twinx()
            ax3t.plot(x, b, color='pink', label='B(q)')
            ax3t.plot(x, scale*a, color='blue', label='scale*A(q)')
            ax3t.legend()
            ax3d.plot(x, b - scale*a, color='gray', label='B(q) - scale*A(q)')
            ax3d.legend(bbox_to_anchor=(1, 1), loc='lower right')

            sb = smooth(b)
            ax4.set_title("smooth B(q) - scale*A(q)", fontsize=16)
            ax4.plot(x, a, color='C1', label='A(q)')
            ax4.legend(loc='center right')
            ax4t = ax4.twinx()
            ax4t.plot(x, sb, color='cyan', label='smooth B(q)')

            if smooth_bq:
                smooth_scale = smooth_scales[limit]
            else:
                def min_norm_diff_sbq_scaled_aq(pv):
                    return np.sum((sb - pv[0]*a)**2)
                init_scale = init_scales[i]
                ret = minimize(min_norm_diff_sbq_scaled_aq, (init_scale,))
                smooth_scale = ret.x[0]

            sa = smooth_scale*a
            ax4t.plot(x, sa, color='blue', label='scale A(q)')
            ax4t.legend()
            ax4d.plot(x, sb - sa, color='gray', label='smooth B(q) - scale*A(q)')
            ax4d.legend(bbox_to_anchor=(1, 1), loc='lower right')

            fig.tight_layout()
            fig.subplots_adjust(top=0.88)
            plt.show()
            plt.pop()

        return limit, surely
