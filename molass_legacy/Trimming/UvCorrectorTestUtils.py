# coding: utf-8
"""
    UvCorrectorTestUtils.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
from time import sleep
from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings
from SerialTestUtils import prepare_serialdata_env
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.UV.PlainCurve import make_secondary_e_curve_at
from molass_legacy.UV.Absorbance import Absorbance
from .PeakRegion import PeakRegion
from .UvCorrectorEgh import UvCorrectorEgh

class Manipulator:
    def __init__(self, parent, logger, canvas_id=1, restart_str=None):
        self.parent = parent
        self.logger = logger
        self.canvas_id = canvas_id
        self.counter = -1
        self.restart_str = restart_str
        self.restarting = restart_str is not None
        self.prepare_out_folder()

    def prepare_out_folder(self):
        self.out_folder = "temp/figs"
        if not os.path.exists(self.out_folder):
            from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
            mkdirs_with_retry(self.out_folder)

    def run_for_all(self):
        from DataUtils import get_pytools_folder, serial_folder_walk
        pytools = get_pytools_folder()
        root_folder = os.path.join(pytools, "Data")
        serial_folder_walk(root_folder, self.do_for_a_file)

    def do_for_a_file(self, in_folder, uv_folder, plot):
        # from TkTester import TestClient  can cause "RuntimeError: main thread is not in main loop"
        clear_temporary_settings()

        try:
            assert in_folder == uv_folder

            self.counter += 1
            if self.restarting:
                if in_folder.find(self.restart_str) >= 0:
                    self.restarting = False
                else:
                    print("skipping", in_folder)
                    return True, None

            sd = prepare_serialdata_env(in_folder, uv_folder)
            absorbance = Absorbance( sd.lvector, sd.conc_array, sd.xray_curve, col_header=sd.col_header )
            data = absorbance.data
            vector = absorbance.wl_vector
            a_curve = absorbance.a_curve
            a_curve2 = make_secondary_e_curve_at(data, vector, a_curve, self.logger)
            fig_file = os.path.join(self.out_folder, "fig-%03d" % self.counter)
            peak_region = PeakRegion(sd.xray_curve, a_curve, a_curve2)
            uvc = UvCorrectorEgh(a_curve, a_curve2)
            uvc.fit(peak_region, debug=True, fig_file=fig_file)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "do_for_a_file(%s) failure: " % in_folder)

        return True, None
