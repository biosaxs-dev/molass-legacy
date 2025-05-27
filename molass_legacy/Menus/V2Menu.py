"""
    V2Menu.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.MenuButton import MenuButton
from molass_legacy._MOLASS.Version import is_developing_version
from molass_legacy._MOLASS.SerialSettings import clear_v2_temporary_settings
from molass_legacy.Global.V2Init import update_sec_settings

class V2Menu(Tk.Frame):
    def __init__(self, parent, dialog, adjusted_sd=False):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.dialog = dialog
        self.adjusted_sd = adjusted_sd

        MENU_LIST = [
                    ("Optimization", lambda: self.v2_exec_wrapper(self.show_peak_editor, "V2 Optimization")),
                    ("Result Viewer", lambda: self.v2_exec_wrapper(self.show_result_viewer, "V2 Result Viewer")),
                    ("Result Animation", lambda: self.v2_exec_wrapper(self.show_result_animation, "V2 Result Animation")),
                    # ("Automated Execution", lambda: self.v2_exec_wrapper(self.show_automated_execution, "V2 Automated Execution")),
                    ]

        if is_developing_version():
            MENU_LIST.append(("Parameter Transition", lambda: self.v2_exec_wrapper(self.show_parameter_transition, "V2 parameter Transition")))

        self.menu = MenuButton(self, "▶▶ V2 Menu", MENU_LIST)
        # self.menu.entryconfig(2, state=Tk.DISABLED)
        self.menu.pack()

    def config(self, **kwargs):
        self.menu.config(**kwargs)

    def v2_exec_wrapper(self, proc, proc_name):
        self.dialog.set_state_guide_message("Executing " + proc_name)
        self.dialog.exec_wrapper(proc, proc_name)
        self.dialog.set_state_guide_message(proc_name + " done.")

    def get_ready_for_peak_editor(self):
        if not self.dialog.get_it_ready_for_qmm():
            return False

        dialog = self.dialog
        logger = dialog.analyzer.app_logger     # not yet used
        in_folder = dialog.in_folder.get()

        self.data_folder    = in_folder.replace( '\\', '/' )
        tester_info = self.dialog.tester_info
        if tester_info is None:
            tester_info_log = ''
        else:
            tester_info_log = ' with test pattern ' + str( tester_info.test_pattern )
        logger.info( "start analysis for " + self.data_folder + tester_info_log )
        return True

    def show_peak_editor(self, pe_proxy=None, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.OptStrategyDialog
            reload(molass_legacy.Optimizer.OptStrategyDialog)
        from molass_legacy.Optimizer.OptStrategyDialog import OptStrategyDialog
        from molass_legacy.Optimizer.OptimizerUtils import show_peak_editor_impl
        clear_v2_temporary_settings()
        update_sec_settings()

        self.strategy_dialog = None
        self.peak_editor = None
        self.fullopt_dialog = None

        dialog = self.dialog
        if pe_proxy is None:
            strategy_dialog = OptStrategyDialog(dialog, dialog.serial_data, pre_recog=dialog.pre_recog)
            self.strategy_dialog = strategy_dialog
            strategy_dialog.show()
            if not strategy_dialog.applied:
                return
        else:
            strategy_dialog = None

        if not self.get_ready_for_peak_editor():
            # what happend?
            # task: clarify this case
            return

        show_peak_editor_impl(strategy_dialog, dialog, pe_proxy=pe_proxy,
                                pe_ready_cb=self.pe_ready_cb, apply_cb=self.pe_apply_cb)

        self.dialog.reset_tmp_logger("fullopt_dialog")

    def get_strategy_dialog(self):
        return self.strategy_dialog

    def pe_ready_cb(self, peak_editor):
        self.peak_editor = peak_editor

    def get_peak_editor(self):
        return self.peak_editor

    def pe_apply_cb(self, fullopt_dialog):
        self.fullopt_dialog = fullopt_dialog

    def get_fullopt_dialog(self):
        return self.fullopt_dialog

    def show_result_folder_selector(self, proc_name, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.Optimizer.ResultFolderSelector
            reload(molass_legacy.Optimizer.ResultFolderSelector)
        from molass_legacy.Optimizer.ResultFolderSelector import show_result_folder_selector_impl

        clear_v2_temporary_settings()
        update_sec_settings()

        # logger = self.dialog.analyzer.app_logger
        self.trace_dialog = None
        show_result_folder_selector_impl(self.dialog, proc_name, ready_cb=self.td_ready_cb)
        # self.dialog.reset_tmp_logger("trace_dialog")  # this results in "final_error.log" ???

    def show_result_viewer(self, debug=False):
        self.show_result_folder_selector("Result Viewer", debug=debug)

    def show_result_animation(self, debug=False):
        self.show_result_folder_selector("Result Animation", debug=debug)

    def show_parameter_transition(self, debug=False):
        self.show_result_folder_selector("Parameter Transition", debug=debug)

    def td_ready_cb(self, trace_dialog):
        self.trace_dialog = trace_dialog

    def get_trace_dialog(self):
        return self.trace_dialog
    
    def show_automated_execution(self):
        clear_v2_temporary_settings()
        update_sec_settings()
