"""
    PreliminaryRg.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF
"""

import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.Trimming import FlangeLimit, GuinierLimit, CdLimit
import molass_legacy.KekLib.DebugPlot as plt

class PreliminaryRg:
    def __init__(self, D, E, e_curve, qv, flange_limit, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.qv = qv
        self.default_eno = e_curve.get_primarypeak_i()
        self.D = D
        self.E = E
        self.flange_limit = flange_limit
        self.current_eno = None
        self.compute_rg(**kwargs)
        self.logger.info("default_eno=%d, current_eno=%d", self.default_eno, self.current_eno)

    def compute_rg(self, selected=None, ip_effect_info=None, debug=False):
        if selected is None:
            selected = self.default_eno

        if self.current_eno is None or selected != self.current_eno:
            qslice = slice(0, self.flange_limit)
            scattering_y, selected = self.get_scattering_y(selected, qslice, return_selection=True)
            data = np.vstack( [self.qv[qslice], scattering_y, self.E[qslice, selected]] ).T
            self.sg = sg = SimpleGuinier( data )
            """
            self.sg is going to be used in AngularRange.GuinierLimit,
            which may not be safe for erroneous cases below
            """
            if sg.Rg is None or sg.Rg < 10.0:
                self.Rg = 10.0
                self.guinier_start = 0
                self.logger.warning("Rg=%s < 10.0 has been replaced by 10.0 to avoid exception.", str(sg.Rg))
            else:
                self.Rg = sg.Rg
                self.guinier_start = sg.guinier_start
            self.logger.info("preliminary Rg=%.3g at eno=%d", self.Rg, selected)
            self.current_eno = selected

            if debug:
                print("self.guinier_start=", self.guinier_start )
                qv = self.qv
                glim = sg.guinier_stop
                qv2 = qv[0:glim]**2
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("PreliminaryRg.compute_rg debug")
                    ax.plot(qv2, np.log(scattering_y[0:glim]))
                    ax.plot(sg.guinier_x, sg.guinier_y)
                    fig.tight_layout()
                    plt.show()

        return self.Rg

    def get_scattering_y(self, selected=None, qslice=None, return_selection=False):
        if selected is None:
            selected = self.default_eno
        if qslice is None:
            qslice = slice(0, self.flange_limit)
        start = max(0, selected - 2)
        stop  = min(self.D.shape[1], selected + 2)
        y = np.average(self.D[qslice, start:stop], axis=1)
        if return_selection:
            return y, selected
        else:
            return y

    def get_guinier_start_index(self, **kwargs):
        self.compute_rg(**kwargs)
        return self.guinier_start

def get_flange_limit(D, E, e_curve, qv):
    fl = FlangeLimit(D, E, e_curve, qv)
    return fl.get_limit()

def get_small_anlge_limit(pre_rg, D, E, e_curve, qv, qlimit, logger):

    cut_before_guinier = get_setting('cut_before_guinier')
    if cut_before_guinier == 0:
        angle_start = 0
    else:
        guinier_start = pre_rg.get_guinier_start_index()
        if cut_before_guinier == 1:
            angle_start = guinier_start
            if angle_start > 0:
                logger.info('angular range start Q[%d]=%g has been set from the Guinier interval.' % (angle_start, qv[angle_start]) )
        else:
            gl = GuinierLimit(D, e_curve, pre_rg, qlimit)
            rg_consistency = get_setting('acceptable_rg_consist')
            try:
                angle_start = gl.get_limit(rg_consistency, debug=False)
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(logger, "GuinierLimit.get_limit failed.", n=10)
                angle_start = 0

            if angle_start > 0:
                logger.info('angular range start Q[%d]=%g has been set from the extended Guinier limit.' % (angle_start, qv[angle_start]) )

            use_bqlimit = get_setting('use_bqlimit')
            if use_bqlimit:
                try:
                    bl = CdLimit(D, E, e_curve, qv)
                    b_limit, surely = bl.get_limit()
                    if b_limit > angle_start:
                        logger.info("Guinier limit %d has been superseded by B(q) limit %d.", angle_start, b_limit)
                        angle_start = b_limit
                    elif b_limit < angle_start:
                        force_bqlimit = get_setting('force_bqlimit')
                        if force_bqlimit and surely:
                            logger.info("Guinier limit %d has been surely replaced by B(q) limit %d.", angle_start, b_limit)
                            angle_start = b_limit
                except:
                    from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                    log_exception(logger, "getting B(q) limit failed")

        return angle_start

def get_default_angle_range_impl(D, E, e_curve, qv, qlimit, logger):
    flange_limit =get_flange_limit(D, E, e_curve, qv)

    pre_rg = PreliminaryRg(D, E, e_curve, qv, flange_limit)

    angle_start = get_small_anlge_limit(pre_rg, D, E, e_curve, qv, qlimit, logger)

    return angle_start, flange_limit, pre_rg
