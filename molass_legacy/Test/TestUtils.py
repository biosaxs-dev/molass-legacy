# coding: utf-8
"""

    Test.TestUtils.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF

"""
from molass_legacy.SerialAnalyzer.DataUtils import serial_folder_walk

def execute_for_all_data(root_folder, func):
    serial_folder_walk(root_folder, func)

def show_test_dialog(root, in_folder, prepare_func, tester_func, frame_func=None, v2_standard=True):
    from molass_legacy.KekLib.OurTkinter import Tk, Dialog
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from SecSaxs.DataTreatment import DataTreatment
    from molass_legacy.Optimizer.StateSequence import StateSequence
    from molass_legacy.Optimizer.OptimizerSettings import OptimizerSettings
    from molass_legacy.Optimizer.InitialInfo import InitialInfo
    from molass_legacy.Optimizer.OptDataSets import OptDataSets
    from molass_legacy.Optimizer.OptJobInfo import OptJobInfo
    from molass_legacy.KekLib.BasicUtils import Struct

    if root is None:
        from molass_legacy.KekLib.TkUtils import get_tk_root
        root = get_tk_root()

    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)

    if v2_standard:
        pre_recog = PreliminaryRecognition(sd)
        treat = DataTreatment(route="v2", trimming=2, correction=1)
        sd_copy = treat.get_treated_sd(sd, pre_recog)
    else:
        pre_recog = None
        treat = None
        sd_copy = sd

    pref_info = prepare_func(treat, sd_copy, pre_recog)

    class TestDialog(Dialog):
        def __init__(self, parent):
            Dialog.__init__(self, parent, "TestDialog", visible=False)

        def show(self):
            self._show()

        def body(self, body_frame):
            if frame_func is not None:
                frame_func(self, body_frame)

            run_btn = Tk.Button(body_frame, text="Run", command=self.run)
            run_btn.pack()

        def run(self):
            tester_func(self, pref_info)

    def show_pi():
        dialog = TestDialog(root)
        dialog.show()
        root.quit()

    root.after(100, show_pi)
    root.mainloop()
    root.destroy()
