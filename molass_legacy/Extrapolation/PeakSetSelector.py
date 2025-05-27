"""
    PeakSetSelector.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import logging
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Mapping.SingleComponent import PEAK_EVAL_RANGE_RATIO
from molass_legacy.KekLib.ExceptionTracebacker import log_exception

class KnownRec:
    def __init__(self, rno, pno, k_info):
        self.rno = rno
        self.pno = pno
        self.k_info = k_info

    def __repr__(self):
        return '<KnownRec(rno=%d, pno=%d, k_info=%s)>' % (self.rno, self.pno, str(self.k_info))

def get_range_from_elm_recs(p, num, k, f, t, elm_recs, ecurve=None, debug=False):
    x = ecurve.x

    if debug:
        y = ecurve.y
        plt.push()
        fig, ax = plt.subplots()
        ax.set_title("get_range_from_elm_recs debug: [%d-%d] %d:%d" % (p, k, f, t))
        ax.plot(x, y, color='orange')
        axt = ax.twinx()

    hmax = None
    top = (f+t)//2
    for i, rec in enumerate(elm_recs):
        evaluator = rec[1]
        # h = evaluator.get_param_value(0)
        """
        h does not necessarily indicate the hight
        as observed in 20190529_4
        """
        h = evaluator(x[top])

        if debug:
            print([i], 'h=', h)
            c = rec[0]
            y = evaluator(x)
            axt.plot(x, y, color='C%d'%c, label=str(c))

        if hmax is None or h > hmax:
            try:
                f, t = evaluator.x_from_height_ratio(ecurve, PEAK_EVAL_RANGE_RATIO)
                hmax = h
                if debug:
                    print([i], "set: (f, t)=", (f, t), "h=", h)
            except:
                if False:
                    with plt.Dp():
                        print([i], "debug plot")
                        y_ = ecurve.y
                        fig, ax = plt.subplots()
                        ax.set_title("get_range_from_elm_recs failure debug")
                        ax.plot(x, y_)
                        ax.plot(x, evaluator(x))
                        fig.tight_layout()
                        plt.show()
                log_exception(None, "get_range_from_elm_recs failure: ")

    if debug:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        for p in [f, t]:
            p_ = int(round(p))
            ax.plot([p_,p_], [ymin, ymax], ':', color='red', label=str(p_))
        ax.legend(loc='upper left')
        axt.legend(loc='upper right')
        plt.show()
        plt.pop;()

    return int(round(f)), int(round(t))

class PeakSetSelector:
    def __init__(self, paired_ranges, ecurve=None, debug=False):
        self.logger = logging.getLogger(__name__)
        self.paired_ranges = paired_ranges
        x0 = ecurve.x[0]    # x0 == 0 for traditional v1 usage

        if debug:
            x = ecurve.x
            y = ecurve.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("PeakSetSelector entry")
                ax.plot(x, y)
                for p, paired_range in enumerate(paired_ranges):
                    for k, rec in enumerate(paired_range.elm_recs):
                        func = rec[1]
                        ax.plot(x, func(x), ":", label="%d-%d" % (p,k))
                ax.legend()
                fig.tight_layout()
                plt.show()

        range_indeces = []
        pfoot_indeces = []
        for p, paired_range in enumerate(paired_ranges):
            elm_recs = paired_range.elm_recs
            fromto_list = paired_range.get_fromto_list()
            num_fromto = len(fromto_list)
            for k, (f, t) in enumerate(fromto_list):
                range_indeces.append( [p, f + x0, t + x0])
                if elm_recs is None:
                    pfoot_indeces.append( [p, f + x0, t + x0])
                else:
                    f_, t_ = get_range_from_elm_recs(p, num_fromto, k, f, t, elm_recs, ecurve=ecurve, debug=debug)
                    pfoot_indeces.append( [p, f_, t_])

        self.logger.info("range_indeces have been created as follows")
        for n, rec in enumerate(range_indeces):
            self.logger.info("[%d] %s", n, str(rec))
        self.logger.info("pfoot_indeces have been created as follows")
        for n, rec in enumerate(pfoot_indeces):
            self.logger.info("[%d] %s", n, str(rec))

        self.required_peakset_infos = []
        self.known_peak_info_list = None
        self.row_info_dict = {}
        """
            e.g. required_peakset_infos is constructed as follows for OA_Ald

            [row] [peak0, n-th-in-the-selected, {peak0, peak1, ...} ]

            [0] [0, 0, {0}]         # row [0] represents peak0 which is 0-th in {peak0}
            [1] [0, 0, {0, 1}]      # row [1] represents peak0 which is 0-th in {peak0, peak1}
            [2] [1, 1, {0, 1}]      # row [2] represents peak1 which is 1-th in {peak0, peak1}
            [3] [2, 0, {2}]         # row [3] represents peak2 which is 0-th in {peak2}
            [4] [2, 0, {2}]         # row [4] represents peak2 which is 0-th in {peak2}

            note that numbers in the required_peakset_infos represent peaks not elements.
            (although in this case, "peaks" and "elements" are identical,
            there are cases where a single peak consists of more than one elements)

            for a microfluidic paried range, it should be

            [0] [0, 0, {0, 1}]
            [1] [1, 1, {0, 1}]
        """

        """
            f ---------------- t
                  f_ --------------------- t_
             max(f,f_)  <=  min(t,t_)
               means there is some overlap
        """
        self.logger.info("required_peakset_infos are created as follows")
        row = 0
        for p, f, t in range_indeces:
            peaks = set()
            for p_, f_, t_ in pfoot_indeces:
                max_f = max(f, f_)
                min_t = min(t, t_)
                if max_f <= min_t:
                    peaks.add( p_ )
            try:
                nth = list(peaks).index(p)
            except:
                # verify that this should never occur
                nth = 0
                self.logger.warning("to cope with this unexpected situation, nth is set to 0")

            self.required_peakset_infos.append( [p, nth, peaks] )
            self.logger.info("[%d] %s", row, str([p, nth, peaks]))
            row_info = self.row_info_dict.get(p)
            if row_info is None:
                row_info = self.row_info_dict[p] = []
            row_info.append(row)
            row += 1

        if debug:
            from matplotlib.patches import Rectangle

            print('PeakSetSelector ---- debug info ---')
            print('paired_ranges=', paired_ranges)
            print('range_indeces=', range_indeces)
            print('pfoot_indeces=', pfoot_indeces)
            print('required_peakset_infos=', self.required_peakset_infos)

            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("PeakSetSelector Debug for" + str(paired_ranges))
                x = ecurve.x
                y = ecurve.y
                ax.plot(x, y, color='orange')
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax)
                def plot_ranges(ymin_, ymax_, indeces, color, label_fmt):
                    for pk, f, t in indeces:
                        p = Rectangle(
                                (f, ymin_),     # (x,y)
                                t - f,          # width
                                ymax_ - ymin_,  # height
                                facecolor = color,
                                alpha = 0.2,
                                label = label_fmt % pk,
                            )
                        ax.add_patch(p)
                plot_ranges(ymin, ymax, range_indeces, 'cyan', "%dth range")
                plot_ranges(ymin, ymax/2, pfoot_indeces, 'pink', "%dth foot")
                ax.legend()
                plt.show()

    def update_known_peak_info_list(self, known_info_list):
        # print('update_known_peak_info_list: required_peakset_infos=', self.required_peakset_infos)
        # print('update_known_peak_info_list: known_info_list=', known_info_list)
        self.known_peak_info_list = []
        for row, info in enumerate(self.required_peakset_infos):
            pno, nth, peakset = info

            known_peak_info = []
            for ref_pno in peakset:
                known_pair_list = []
                for r in self.row_info_dict[ref_pno]:
                    known_pair_list.append( (abs(r - row), known_info_list[r]) )
                peak_info = sorted(known_pair_list)[0][1]
                known_peak_info.append(peak_info)
            self.known_peak_info_list.append(known_peak_info)

    def select_peakset(self, row):
        peakset_main_info = self.required_peakset_infos[row]
        if self.known_peak_info_list is None:
            known_peak_info = None
        else:
            known_peak_info = self.known_peak_info_list[row]
        self.logger.info("selected %d-th range has %s element(s).", row, len(peakset_main_info[2]))
        return  peakset_main_info + [known_peak_info]

    def select_demo_ranges_for_gd(self, uv_y, opt_recs):
        print('paired_ranges=', self.paired_ranges)
        print('required_peakset_infos=', self.required_peakset_infos)

        max_top_y = None
        max_top_x = None
        max_pno = None
        for p, paired_range in enumerate(self.paired_ranges):
            top_x = paired_range.top_x
            print([p], top_x)
            top_y = uv_y[top_x]
            if max_top_y is None or top_y > max_top_y:
                max_top_y = top_y
                max_top_x = top_x
                max_pno = p

        selected_pno = None
        for pno, peakset_info in enumerate(self.required_peakset_infos):
            if max_pno == peakset_info[0]:
                selected_pno = pno
                break

        print('max_pno=', max_pno, 'max_top_x=', max_top_x)
        selected_eno = None
        for eno, opt_rec in enumerate(opt_recs):
            top_x = opt_rec.peak.top_x
            if abs(top_x - max_top_x) < ALLOW_WIDER:
                selected_eno = eno
                break

        return self.required_peakset_infos[selected_pno]+[None], selected_eno
