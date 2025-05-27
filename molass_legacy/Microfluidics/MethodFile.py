# coding: utf-8
"""
    MethodFile.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""
from bisect import bisect_left
import numpy as np

START_DELAY_RATIO = 0.1
TOTAL_RATE_COLUMNS = [2, 4, 6, 10]

class MethodFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.rows = []
        self.rows_integ = []
        self.x = None
        self.y = None
        self.computed_size = None

        t = 0
        with open(filepath) as fh:
            is_in_data_section  = False
            for line in fh:
                # print(line)
                if line.find('[Data]') >= 0:
                    is_in_data_section = True
                    continue
                if line.find('[End]') >= 0:
                    is_in_data_section = False
                    continue
                if is_in_data_section:
                    if line[0] == ':':
                        self.colnames = line[1:].split('\t')
                        continue
                    values = [ eval(value) for value in line.split('\t') ]
                    self.rows.append(values)
                    m = values[1]/60
                    t += m
                    total_rate = np.sum([values[k] for k in TOTAL_RATE_COLUMNS])
                    self.rows_integ.append([m, t, total_rate])

    def get_elution_data(self, size, debug=False, auto=True):
        print('get_elution_data', size)
        if self.computed_size is None or size != self.computed_size:
            x, y = self.compute_elution_data(size, debug=debug, auto=auto)
            self.x, self.y = x, y
            self.computed_size = size

        return self.x, self.y

    def compute_elution_data(self, size, tv_only=False, debug=False, auto=True):
        print('compute_elution_data', size, tv_only)
        tlist = []
        t = 0
        values = []
        check_rownum = True
        self.row_nums = []
        for k, row in enumerate(self.rows):
            number = row[0]
            # assert number == k + 1
            if check_rownum:
                if number == k + 1:
                    pass
                else:
                    import OurMessageBox as MessageBox
                    MessageBox.showwarning( "Warning",
                        "Row numbers are not as expected in %s." % self.filepath,
                        parent=None )
                    check_rownum = False
            duration = row[1]
            tlist.append(t)
            t += duration
            tlist.append(t)
            fr_values = []
            to_values = []
            for j in range(5):
                j_ = 2+j*2
                fr_value = row[j_]
                fr_values.append(fr_value)
                to_value = row[j_+1]
                to_values.append(to_value)
            values.append(fr_values)
            values.append(to_values)

        self.t = t = np.array(tlist)
        matrix = np.array(values)
        print('matrix.shape=', matrix.shape, 'len(t)=', len(t))
        total = np.sum(matrix[:,[0,1,2,4]], axis=1)
        # print('t=', t)
        # print('total=', total)
        j = 1
        half_t = t[-1] * 0.7
        print('half_t=', half_t)
        self.v = v = matrix[:,j]/total
        self.start = None
        for ub in [0.2]:
            wh = np.where( np.logical_and( t < half_t, v < ub ) )[0]
            if len(wh) > 0:
                self.start = wh[-1]
                break

        assert self.start is not None

        try:
            self.start_t = t[self.start]
            self.end = np.where( np.logical_and(t > self.start_t, v > 0.5) )[0][0]
            print( 'start, end=', (self.start, self.end) )
            self.end_t = t[self.end]
            self.start_v = v[self.start]
            self.end_v = v[self.end]
        except:
            import molass_legacy.KekLib.DebugPlot as plt
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(t, v)
            plt.show()

        if tv_only:
            return t, v

        x = np.linspace(0, t[-1], size)
        y = []
        for x_ in x:
            i = bisect_left(t, x_)
            if self.start_t <= x_ and x_ <= self.end_t:
                w = (x_ - self.start_t)/(self.end_t - self.start_t)
                v_ = self.start_v * (1-w) + self.end_v * w
            else:
                v_ = v[i]
            y.append(v_)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            import time
            # from molass_legacy.KekLib.NumpyUtils import np_savetxt
            # np_savetxt('matrix.csv', matrix)
            colname = self.colnames[2+j*2].replace('St.', '')
            fig = plt.figure()
            ax = fig.gca()
            # for j in range(matrix.shape[1]):
            ax.plot(x, y, label=colname + ' generated')
            ax.plot(t, v, 'o', label=colname + ' data')
            for k in [self.start, self.end]:
                ax.plot(t[k], v[k], 'o', color='red')
            ax.legend()
            fig.tight_layout()
            if auto:
                plt.show(block=False)
                time.sleep(0.5)
            else:
                plt.show()

        self.cp = []
        for t_ in [self.start_t, self.end_t]:
            self.cp.append(bisect_left(x, t_))
        print('self.cp=', self.cp)
        return x, np.array(y)

    def get_linear_slope_ends(self):
        # print('cp=', self.cp)
        return [ self.cp[0], self.cp[1]-1 ]

    def guess_range(self, size):
        self.get_elution_data(size)
        start, stop = self.get_linear_slope_ends()
        length = stop - start
        delay = int(length*START_DELAY_RATIO)
        return slice(start+delay, stop)

    def guess_critical_points(self):
        last_row_integ = self.rows_integ[-1]
        print(last_row_integ)
        size = int(last_row_integ[1])+1
        t, v = self.compute_elution_data(size, tv_only=True)
        w = np.where(np.logical_and(t < self.start_t, v > self.start_v))
        i = w[0][-1]

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            n = np.arange(len(t))
            print(list(zip(n, t, v)))
            print([i, self.start, self.end])
            plt.plot(t, v, ':')
            plt.plot(t, v, 'o')
            for k in [i, self.start, self.end]:
                plt.plot(t[k], v[k], 'o', color='red')
            plt.show()

        i_ = i//2
        j_ = self.start//2
        k_ = j_ + 1
        return i_, j_, k_
