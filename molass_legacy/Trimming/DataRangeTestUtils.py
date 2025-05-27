# coding: utf-8
"""
    DataRangeTestUtils.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
from time import sleep
from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings
from SerialTestUtils import prepare_serialdata_env
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from .DataRange import DataRangeDialog

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
            pre_recog = PreliminaryRecognition(sd)

            print(in_folder)
            self.dialog = dialog = DataRangeDialog(self.parent, pre_recog)
            dialog.after(1000, self.manipulate)
            dialog.show()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(self.logger, "do_for_a_file(%s) failure: " % in_folder)

        return True, None

    def manipulate(self):
        dialog = self.dialog
        if self.canvas_id == 1:
            self.canvas = dialog.get_current_frame().canvases[1]
            dialog.after(1000, self.save_the_figure)
            self.canvas.show_extra_info()
        elif self.canvas_id == 2:
            self.canvas = dialog.get_current_frame().canvases[2]
            dialog.after(2000, self.save_the_figure)    # needs more time than the above
            self.canvas.show_guinier_inspector()

    def save_the_figure(self):
        inspector = self.canvas.inspector
        n = 0
        while inspector is None:
            print("waiting inspector to be ready")
            sleep(1)
            inspector = self.canvas.inspector
            n += 1
            self.dialog.cancel()
            assert n < 10

        file = os.path.join(self.out_folder, "fig-%03d" % self.counter)
        inspector.save_the_figure(file)
        inspector.cancel()
        self.dialog.cancel()

