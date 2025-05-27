# coding: utf-8
"""
    QmmMenu.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from MenuButton import MenuButton

class QmmMenu(Tk.Frame):
    def __init__(self, parent, dialog, in_folder):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.dialog = dialog
        self.in_folder = in_folder
        self.menu = MenuButton(self, "▶▶ QMM Menu", [
                            ("EGH-QMM", self.show_egh_qmm_dialog),
                            ("EMG-QMM", self.show_emg_qmm_dialog),
                            ("QMM Options", self.show_qmm_options_dialog),
                            ("QMM Window", self.show_qmm_window_setting_dialog),
                            ])
        self.menu.pack()

    def prepare_denoise_rank(self, dialog):
        """
        called from dialog(GuiMain) when ready
        """
        from .PairedDataSets import get_denoise_rank_impl
        from molass_legacy._MOLASS.SerialSettings import set_setting
        sd = dialog.serial_data
        denoise_rank = get_denoise_rank_impl(sd.get_xray_curve())
        set_setting('last_denoise_rank', denoise_rank)

    def config(self, **kwargs):
        self.menu.config(**kwargs)

    def show_egh_qmm_dialog(self):
        from Prob.EghMixture import EghMixture
        self.show_qmm_dialog(EghMixture)

    def show_emg_qmm_dialog(self):
        from Prob.EmgMixture import EmgMixture
        self.show_qmm_dialog(EmgMixture)

    def show_qmm_dialog(self, mm_type):
        from Prob.QmmController import get_qmm_controller
        from molass_legacy.SerialAnalyzer.AbnormalityCheck import update_abnormality_fix_state

        dialog = self.dialog

        if not dialog.get_it_ready_for_qmm():
            return

        self.qmm_controller = None
        try:
            update_abnormality_fix_state(dialog.serial_data, dialog.file_info_table, dialog.analyzer.app_logger)

            in_folder = self.in_folder.get()
            dialog.is_executing = True
            dialog.qmm_controller = get_qmm_controller(dialog, mm_type, in_folder, sd=dialog.serial_data)
            assert dialog.qmm_controller is not None
            dialog.is_executing = False

            ret = dialog.qmm_controller.show_dialog(dialog)
        except:
            try:
                import molass_legacy.KekLib.CustomMessageBox         as MessageBox
            except:
                import OurMessageBox            as MessageBox
            if dialog.testing:
                self.dialog.quit(immediately=True)
            else:
                import sys
                message = "error occured in QmmDialog\n" + str(sys.exc_info())
                MessageBox.showerror( "QmmDialog Error", message, parent=dialog.parent )
            ret = False

        if ret:
            print("to Serial Analysis")

    def show_qmm_options_dialog(self):
        from .QmmOptions import QmmOptionsDialog
        opts_dialog = QmmOptionsDialog(self.dialog)
        opts_dialog.show()

    def show_qmm_window_setting_dialog(self):
        from .QmmWindowSetting import QmmWindowSetting
        window_dialog = QmmWindowSetting(self.dialog, self.dialog.fig_frame)
        window_dialog.show()

class MockCurve:
    def get_major_peak_info(self):
        return [None]*3

class MockSd:
    def get_xray_curve(self):
        return MockCurve()

class MockAnalyzer:
    def __init__(self):
        import logging
        self.app_logger = logging.getLogger(__name__)

class App(Dialog):
    def __init__(self, parent, in_folder):
        self.in_folder = Tk.StringVar()
        self.in_folder.set(in_folder)
        self.testing = False
        self.serial_data = MockSd()
        self.fig_frame = None
        self.analyzer = MockAnalyzer()
        Dialog.__init__(self, parent, location="lower right")

    def body(self, body_frame):
        label = Tk.Label(body_frame, textvariable=self.in_folder)
        label.pack()
        menu = QmmMenu(body_frame, self, self.in_folder)
        menu.pack()

    def get_it_ready_for_qmm(self):
        from molass_legacy._MOLASS.SerialSettings import set_setting
        set_setting( 'analysis_name', 'analysis-000' )
        return True
