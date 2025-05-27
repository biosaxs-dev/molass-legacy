# coding: utf-8
"""
    DelayOptimizer.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from SerialDataUtils import get_mtd_filename
from FlowSimulator import FlowSimulator
import molass_legacy.KekLib.DebugPlot as plt

class DelayOptimizer:
    def __init__(self, in_folder, xdata):
        self.y = xdata.e_y
        self.size = xdata.data.shape[1]
        filepath = get_mtd_filename(in_folder)
        self.simulator = FlowSimulator(filepath)
        self.init_params = self.simulator.make_simulation_data(debug=False)
        self.slice_ = self.simulator.guess_range(self.size)
        self.recognize_y_channel_location()
        t, x = self.init_params[0:2]
        print('x[0:5]=', x[0:5])
        self.sm_info = self.simulator.get_linear_range_info(t)

    def get_slice(self):
        return self.slice_

    def get_initial_data(self):
        return self.init_params

    def recognize_y_channel_location(self):
        t, x = self.init_params[0:2]
        i, j = self.simulator.get_y_channel_interval(t)
        slice_ = slice(int(x[i]), int(x[j]))
        print('i=', i, j, slice_)
        y_ = self.y[slice_]
        k = np.argmax(y_)
        self.y_channel = slice_.start + k
        print('self.y_channel=', self.y_channel)

    def get_y_channel_point(self):
        return self.y_channel, self.y[self.y_channel]

    def get_optimized_data(self):
        self.optimize_with_two_steps()
        return self.opt_params

    def optimize_with_two_steps(self):
        simulator = self.simulator
        i = self.sm_info.yc_peak
        j = self.sm_info.lin_start
        k = self.sm_info.lin_end
        xi = self.init_params[1]
        k_ = int(xi[k])
        j_ = int(xi[j])
        xsize = k_ - j_
        ysize = self.y[k_] - self.y[j_]

        def delay_error(params):
            dvf = params[0]
            smt, smx, smy = simulator.make_simulation_data(delay_factor=dvf)
            i = simulator.get_y_channel_index(smt)
            c_error = ((smx[i] - self.y_channel)/xsize)**2
            return c_error

        init_params = np.array([simulator.delay_factor])
        result = minimize(delay_error, init_params, method=None, bounds=[(-10, 20)])
        opt_dvf = result.x[0]

        smt, smx, smy = simulator.make_simulation_data(delay_factor=opt_dvf)
        i, j, k = simulator.get_optimizer_indeces(smt)
        """
        to be precise, correct rounding error
        """
        xmin = smx[j]
        xmax = smx[k]
        ymin = smy[j]
        ymax = smy[k]
        j_, k_ = int(xmin), int(xmax)+1
        slope = (ymax - ymin)/(xmax - xmin)
        y0 = np.linspace(ymin+slope*(j_ - xmin), ymax+slope*(k_ - 1 - xmax), k_ - j_)
        debug = [0]
        def overlay_error(params):
            vscale = params[0]
            y1 = y0*vscale
            y2 = self.y[j_:k_]
            v_error = np.average(((y1-y2)/ysize)**2)
            if debug[0]:
                print('vscale=', vscale)
                print('v_error=', v_error)
                fig = plt.figure()
                ax = fig.gca()
                x = np.arange(j_, k_)
                ax.plot(x, y1)
                ax.plot(x, y2, color='orange')
                ax.plot(smx[i-1:], smy[i-1:]*vscale, color='blue')  # this line must overlap with y1 line
                ax.plot(smx[i], smy[i]*vscale, 'o', color='red')
                ax.plot(self.y_channel, self.y[self.y_channel], 'o', color='yellow')
                ret = plt.show()
                if not ret:
                    debug[0] = 0

            return v_error

        init_endy = self.init_params[2][k]
        init_vscale = ysize/init_endy
        print('ysize=', ysize, 'init_endy=', init_endy)
        init_params = np.array([init_vscale])
        print('init_params=', init_params)
        method = 'L-BFGS-B'
        # method = 'TNC'
        # method = 'SLSQP'
        # method = 'Nelder-Mead'
        result = minimize(overlay_error, init_params, method=method, bounds=[(init_vscale*0.5, init_vscale*1.5)])
        vscale = result.x[0]
        print('opt_dvf, vscale=', opt_dvf, vscale)

        sm_info = simulator.get_linear_range_info(smt, vscale)
        self.opt_params = smt, smx, smy, sm_info
