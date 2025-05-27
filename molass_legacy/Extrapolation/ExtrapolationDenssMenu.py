# coding: utf-8
"""
    ExtrapolationDenssMenu.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk
from MenuButton import MenuButton

class ExtrapolationDenssMenu(Tk.Frame):
    def __init__(self, parent, dialog, data_list):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.dialog = dialog
        self.data_list = data_list
        self.denss_menu = MenuButton(self, "DENSS Menu", [
                            ("Run all in background", self.submit_all),
                            ("Show DENSS Manager", self.show_denss_manager),
                            ("Show empty DENSS GUI", self.show_denss_gui),
                            ("Electron Density Viewer", self.show_ed_viewer),
                            ("SAXS Simulator", self.show_saxs_simulator),
                            ])
        self.denss_menu.pack()

    def submit_all(self):
        print('submit_all')
        from molass_legacy.DENSS.DenssUtils import fit_data
        from molass_legacy.DENSS.DenssManager import get_list
        import molass_legacy.KekLib.CustomMessageBox as MessageBox

        job_list = get_list()
        num_jobs = len(job_list)
        num_jobs_str = "no" if num_jobs == 0 else "already " + str(num_jobs)
        yn = MessageBox.askyesno( "Run All Comfirmation",
            "There are %s jobs submitted.\n" % num_jobs_str
            + "Are you sure to run these %d jobs?" % len(self.data_list), parent=self.dialog )
        if not yn:
            return

        from molass_legacy.DENSS.DenssManager import JobInfo
        from molass_legacy.DENSS.DenssManagerDialog import show_manager_dialog
        jobs = []
        for q, a, e, f in self.data_list:
            qc, ac, ec, dmax = fit_data(q, a, e)
            jobs.append(JobInfo('denss', q=qc, a=ac, e=ec, dmax=dmax, infile_name=f))
        show_manager_dialog(self.dialog, jobs)

    def show_denss_gui(self):
        print('show_denss_gui')
        from molass_legacy.DENSS.DenssGui import DenssGuiDialog
        dialog = DenssGuiDialog(self.dialog)
        dialog.show()

    def show_denss_manager(self):
        print('show_denss_manager')
        from molass_legacy.DENSS.DenssManagerDialog import show_manager_dialog
        show_manager_dialog(self.dialog)

    def show_ed_viewer(self):
        print('show_ed_viewer')
        from molass_legacy.Saxs.EdViewer import EdViewer
        from molass_legacy.KekLib.OurMatplotlib import reset_to_default_style
        viewer = EdViewer(self.dialog)
        viewer.show()
        reset_to_default_style()

    def show_saxs_simulator(self):
        print('show_saxs_simulator')
        from molass_legacy.Saxs.SaxsSimulator import SaxsSimulator
        from molass_legacy.KekLib.OurMatplotlib import reset_to_default_style
        simulator = SaxsSimulator(self.parent)
        simulator.show()
        reset_to_default_style()
