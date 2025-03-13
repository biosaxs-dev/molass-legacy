"""
    RgProcess.RgCurve.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.SliceUtils import slice_consecutives
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Trimming import TrimmingInfo   # used in eval_file

def compute_rg_segment(qv, D, E, slice_, rg_buffer, qu_buffer, progress_cb=None):
    from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

    includes_nan = False
    for j in range(slice_.start, slice_.stop):
        data = np.array([qv, D[:,j], E[:,j]]).T
        sg = SimpleGuinier(data)
        if sg.Rg is None:
            includes_nan = True
        rg = sg.Rg
        rg_buffer[j] = rg
        qu_buffer[j] = sg.basic_quality
        if progress_cb is not None:
            progress_cb(rg_buffer, j)

    if includes_nan:
        # as in 20171226
        import logging
        logger = logging.getLogger(__name__)
        buffer = rg_buffer[slice_]
        pad_value = np.mean(buffer[np.logical_not(np.isnan(buffer))])
        where = slice_.start + np.where(np.isnan(buffer))[0]
        buffer[np.isnan(buffer)] = pad_value
        logger.warning("NaN Rg's have been replaced by mean value %g of non-NaN values at %s.", pad_value, str(where))

def compute_rg_segment_proxy(slice_, rg_buffer, qu_buffer, progress_cb=None):
    if progress_cb is None:
        return

    for j in range(slice_.start, slice_.stop):
        progress_cb(rg_buffer, j)

def compute_rg_segment_wrapper(arg):
    from molass_legacy.KekLib.SharedArrays import SharedArrays
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.DataStructure.MatrixData import simple_plot_3d
    name, tuples, slice_ = arg
    print("name=", name, "tuples=", tuples)
    sa = SharedArrays(name=name, tuples=tuples)
    qv, D, E = sa.get_arrays()
    rg_buffer = np.zeros(D.shape[1])
    compute_rg_segment(qv, D, E, slice_, rg_buffer)
    if False:
        print(slice_)
        plt.push()
        fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        simple_plot_3d(ax1, D)
        ax2.plot(rg_buffer)
        plt.show()
        plt.pop()
    return rg_buffer

def make_availability_slices(y, max_y=None, min_ratio=0.03, min_size=20):
    if max_y is None:
        max_y = np.max(y)

    # int(f), int(t) is meant to convert np.int64 to int in 64GB or larger RAM environment
    # to avoid str(slice_) to become like 'slice(np.int64(202), np.int64(258), None)'
    pairs = [(int(f), int(t)) for f, t in slice_consecutives(np.where(y/max_y >= min_ratio)[0])]

    slices = []
    states = []
    start = 0
    for f, t in pairs:
        size = t - f
        if size < min_size:
            continue

        if start < f:
            slices.append(slice(start, f))
            states.append(0)

        start = t+1
        slices.append(slice(f, start))
        states.append(1)
    if start < len(y):
        slices.append(slice(start, len(y)))
        states.append(0)
    return slices, states

def rg_folder_ok_stamp_path(rg_folder):
    return os.path.join(rg_folder, 'ok.stamp')

def check_rg_folder(rg_folder):
    ret_state = os.path.exists(rg_folder) and os.path.exists(rg_folder_ok_stamp_path(rg_folder))

    if ret_state:
        baseline_type = get_setting("unified_baseline_type")
        baseline_type_txt = os.path.join(rg_folder, "baseline_type.txt")
        if os.path.exists(baseline_type_txt):
            from molass_legacy.KekLib.EvalUtils import eval_file
            prev_baseline_type = eval_file(baseline_type_txt, locals_=globals())
        else:
            prev_baseline_type = None
        ret_state = prev_baseline_type == baseline_type

    return ret_state

class RgCurve:
    def __init__(self, qv, ecurve, D, E, xy_info=None, min_ratio=0.03, multi_process=False, progress_cb=None):
        from time import time
        import logging
        self.logger = logging.getLogger(__name__)
        self.progress_cb = progress_cb

        t0 = time()
        if xy_info is None:
            x = ecurve.x
            y = ecurve.y
            max_y = ecurve.max_y
        else:
            x, y, max_y = xy_info

        slices, states = make_availability_slices(y, max_y=max_y, min_ratio=min_ratio)

        if multi_process:
            # not tested
            segments, qualities = self.create_segments_multi_process(qv, x, y, D, E, slices, states)
        else:
            segments, qualities = self.create_segments_single_process(qv, x, y, D, E, slices, states)

        self.x = x
        self.y = y
        self.ecurve = ecurve
        self.slices = slices
        self.states = states
        self.segments = segments
        self.qualities = qualities
        xr_restrict_list = get_setting("xr_restrict_list")
        self.rg_trimming = None if xr_restrict_list is None else xr_restrict_list[0]
        self.baseline_type = get_setting("unified_baseline_type")
        self.X = None
        self.excl_info = None
        self.excl_spline = None
        self.logger.info("It took %.3g seconds for rg_curve construction.", time()-t0)

    def get_valid_slices(self):
        return self.slices

    def get_mask(self):
        mask = np.zeros(len(self.x), dtype=bool)
        for slice_, state in zip(self.slices, self.states):
            if state:
                mask[slice_] = True
        return mask

    def get_weights(self):
        weight_list = []
        k = 0
        for slice_, state in zip(self.slices, self.states):
            if state:
                if self.qualities is None:
                    qu = None
                else:
                    qu = self.qualities[k]
                if qu is None:
                    qu = np.ones(slice_.stop - slice_.start)
                weight_list.append(qu)
                k += 1
        return np.concatenate(weight_list)

    def create_segments_multi_process(self, qv, x, y, D, E, slices, states):
        from molass_legacy.KekLib.SharedArrays import SharedArrays
        from molass_legacy.KekLib.SciPyCookbook import smooth
        from multiprocessing import Pool

        sa = SharedArrays([qv, D, E])
        sa_info = (sa.name, sa.get_tuples())

        args_list = []
        arg_slices = []
        for slice_, state in zip(slices, states):
            if state == 0:
                continue
            args_list.append((*sa_info, slice_))
            arg_slices.append(slice_)

        with Pool(2) as p:
            ret_list = p.map(compute_rg_segment_wrapper, args_list)

        segments = []
        qualities = []
        for ret, slice_ in zip(ret_list, arg_slices):
            segments.append((x[slice_], y[slice_], smooth(ret[slice_])))
            # qu = 
            # qualities.append(qu)
        return segments, qualities

    def create_segments_single_process(self, qv, x, y, D, E, slices, states):
        from molass_legacy.KekLib.SciPyCookbook import smooth

        rg_buffer_file = get_setting("rg_buffer_file")
        if rg_buffer_file is None:
            rg_buffer = np.zeros(len(x))
            qu_buffer = np.zeros(len(x))
        else:
            rg_buffer = np.loadtxt(rg_buffer_file)
            qu_buffer = None

        segments = []
        qualities = []
        for slice_, state in zip(slices, states):
            if state == 0:
                continue
            if rg_buffer_file is None:
                compute_rg_segment(qv, D, E, slice_, rg_buffer, qu_buffer, progress_cb=self.progress_cb)
                rg = smooth(rg_buffer[slice_])
                qu = qu_buffer[slice_]
            else:
                compute_rg_segment_proxy(slice_, rg_buffer, qu_buffer, progress_cb=self.progress_cb)
                rg = rg_buffer[slice_]
                qu = None
            segments.append((x[slice_], y[slice_], rg))
            qualities.append(qu)
        return segments, qualities

    def get_curve_segments(self):
        return self.segments

    def get_valid_curves(self):
        x_segs = []
        y_segs = []
        rg_segs = []
        for x, y, rg in self.segments:
            x_segs.append(x)
            y_segs.append(y)
            rg_segs.append(rg)
        return [np.concatenate(segs) for segs in [x_segs, y_segs, rg_segs]]

    def add_exclspline(self, poresize=70, return_rhov=False, return_excl_xy=False):
        from scipy.optimize import minimize
        from scipy.interpolate import UnivariateSpline
        x = self.ecurve.x
        segments = self.get_curve_segments()
        k = 0
        tr_list = []
        rg_list = []
        qu_list = []
        for x_, y_, rg in segments:
            tr_list.append(x_)
            rg_list.append(rg)
            qu = self.qualities[k]
            qu_list.append(qu)
            k += 1
        trv = np.concatenate(tr_list)
        rgv = np.concatenate(rg_list)
        quv = np.concatenate(qu_list)
        rhov = rgv/poresize
        rhov[rhov > 1] = 1
        # print("rhov=", rhov)
        if return_rhov:
            return rhov

        def objective(p):
            t0, K = p
            trv_ = t0 + K*(1 - rhov)**3
            return np.sum(quv * (trv_ - trv)**2)

        res = minimize(objective, (0, 1000))
        t0, K = res.x
        excl_x = t0 + K*(1 - rhov)**3
        excl_y = rhov*poresize
        if return_excl_xy:
            return excl_x, excl_y

        x_ = np.flip(excl_x)
        y_ = np.flip(excl_y)
        ux_, uy_ = np.unique(np.array([x_, y_]), axis=1)        

        self.excl_info = poresize, t0, K
        self.excl_spline = UnivariateSpline(ux_, uy_)

    def get_probabilistic_data(self):
        if self.X is None:
            from Rgg.RggUtils import convert_to_probabilitic_data
            X_list = []
            for x, y, rg in self.get_curve_segments():
                X_list.append(convert_to_probabilitic_data(x, y, rg, max_y =self.ecurve.max_y))
            self.X = np.concatenate(X_list)
        return self.X

    def proof_plot_impl(self, fig, axes):
        import molass_legacy.KekLib.MatplotlibUtils      # required for annotate3D
        from Rgg.RggUtils import plot_histogram_2d

        ax1, ax2, ax3 = axes
        ecurve = self.ecurve
        segments = self.get_curve_segments()
        x = ecurve.x
        y = ecurve.y
        max_y = ecurve.max_y
        X = self.get_probabilistic_data()
        fig.suptitle("Generation of 2D Probabilistic Data", fontsize=20)
        ax1.set_title("Xray Elution and Smoothed Rg Curve", fontsize=16)
        ax2.set_title("Rg Qualities", fontsize=16)
        ax3.set_title("Histogram of the 2D Propbabilistic Data", fontsize=16)
        axt = ax1.twinx()
        axt.grid(False)

        ax1.plot(x, y, color='orange')
        ax2.set_xlim(ax1.get_xlim())

        k = 0
        for x_, y_, rg in segments:
            axt.plot(x_, rg, ':', color='C1')
            qu = self.qualities[k]
            ax2.plot(x_, qu)
            plot_histogram_2d(ax3, x_, y_, rg, max_y)
            k += 1

        ymin, ymax = axt.get_ylim()
        axt.set_ylim(min(10, ymin), max(50, ymax))

        ptx = ecurve.get_primarypeak_x()
        pti = ecurve.get_primarypeak_i()
        k = 0
        for slice_, state in zip(self.slices, self.states):
            if state == 0:
                continue
            if slice_.start <= pti and pti < slice_.stop:
                pty = segments[k][2][pti - slice_.start]
                break
            k += 1

        ptz = 100
        peaktop = (ptx, pty, ptz)
        ax3.annotate3D('(%.3g, %.3g, %.3g)' % peaktop, peaktop,
              xytext=(30,-30),
              textcoords='offset points',
              bbox=dict(boxstyle="round", fc="lightyellow"),
              arrowprops = dict(arrowstyle="-|>", ec='black', fc='white', lw=1))

        ax3.set_xlabel('Eno')
        ax3.set_ylabel('Rg')
        ax3.set_zlabel('Counts')

        ax3.set_xlim(ax1.get_xlim())
        ax3.set_ylim(10, 60)

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

    def proof_plot(self):
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            figsize=(21,7)
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133, projection='3d')
            self.proof_plot_impl(fig, (ax1, ax2, ax3))
            plt.show()

    def plot(self, fig=None, ax=None):
        if fig is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot()

        axt = ax.twinx()
        axt.grid(False)

        ax.set_title("$R_g$ Curve Object Plot")

        ecurve = self.ecurve
        segments = self.get_curve_segments()
        x = ecurve.x
        y = ecurve.y
        ax.plot(x, y, label='data')
        k = 0
        for x_, y_, rg in segments:
            label = 'Guinier $R_g$ curve' if k == 0 else None
            axt.plot(x_, rg, ':', color='gray', label=label)
            k += 1

        if self.excl_spline is not None:
            poresize, t0 = self.excl_info[0:2]
            ex = x[x > t0]
            ey = self.excl_spline(ex)
            axt.plot(ex, ey, color='yellow', label='SEC $R_g$ curve')
            axt.axvline(x=t0, linestyle=':', color='red', label='$t_0$ (poresize=%.3g)' % poresize)

        axt.legend()
        fig.get_tight_layout()

    def save_buffer(self, path):
        buffer = np.zeros(len(self.x))
        k = 0
        for slice_, state in zip(self.slices, self.states):
            if state == 0:
                continue
            buffer[slice_] = self.segments[k][2]
            k += 1
        np.savetxt(path, buffer)

    def export(self, folder):
        import os

        precision_save = np.get_printoptions()["precision"]
        np.set_printoptions(precision=17)

        for filename, item in [
                ("slices.txt", self.slices),
                ("states.txt", self.states),
                ("segments.txt", self.segments),
                ("qualities.txt", self.qualities),
                ("rg_trimming.txt", self.rg_trimming),
                ("baseline_type.txt", self.baseline_type),
                ]:
            with open(os.path.join(folder, filename), "w") as fh:
                fh.write(str(item))

        with open(rg_folder_ok_stamp_path(folder), "w") as fh:
            fh.write("")

        np.set_printoptions(precision=precision_save)

    def get_rgs_from_trs(self, trs):
        if self.excl_spline is None:
            self.add_exclspline()
        return self.excl_spline(trs)

    def get_rgs_from_trs_depricated(self, trs):
        from bisect import bisect_right
        ret_rgs = np.ones(len(trs)) * np.nan
        for k, tr in enumerate(trs):
            for x, y, rg in self.segments:
                if x[0] <= tr and tr <= x[-1]:
                    j = bisect_right(x, tr)
                    ret_rgs[k] = rg[j]
                    break
        return ret_rgs

RG_CALLBACK_CYCLE = 10

class ProgressCallback:
    def __init__(self, queue, f, t):
        self.queue = queue
        self.f = f
        self.t = t

    def __call__(self, rg_buffer, j):
        if j % RG_CALLBACK_CYCLE == 0:
            rate = j/len(rg_buffer)
            progress = self.f*(1 - rate) + self.t*rate
            self.queue.put([progress, (rg_buffer, j)])

def draw_rg_bufer(ax, p_info, dialog, x):
    rg_buffer, j = p_info
    draw = j % RG_CALLBACK_CYCLE == 0
    if draw:
        # drawing each seems too heavy
        rg_positive = rg_buffer > 0
        x = x[rg_positive]
        y = rg_buffer[rg_positive]
        if dialog.rg_line is None:
            dialog.rg_line, = ax.plot(x, y, color='gray', alpha=0.5, label='Rg')
            ax.legend(loc='center left')
        else:
            dialog.rg_line.set_data(x, y)
        ax.set_ylim(10, np.max(y))
        print([j], "drawn")

    return draw

def compute_init_rgs(rg_curve, n_components):
    x_, y_, rg_ = rg_curve.get_valid_curves()
    mean = np.mean(rg_)
    std = np.std(rg_)
    init_rgs = np.linspace(mean+std, mean-std, n_components)
    return init_rgs
