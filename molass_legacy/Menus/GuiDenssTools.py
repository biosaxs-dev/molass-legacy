"""

    GuiDenssTools.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy._MOLASS.SerialSettings import get_setting

class GuiDenssToolsMenu(Tk.Menu):
    def __init__(self, parent, menubar ):
        self.parent = parent

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade( label="DENSS Tools", menu=self )
        self.add_command( label="DENSS GUI", command=self.show_denss_dialog )
        self.add_command( label="DENSS Manager", command=self.show_denss_manager )
        self.add_command( label="Electron Density Viewer", command=self.show_ed_viewer )
        self.add_command( label="SAXS Simulator", command=self.show_saxs_simulator )
        self.update_states()

    def update_states(self):
        from molass_legacy.Env.EnvInfo import get_global_env_info
        analysis_folder = get_setting('analysis_folder')
        env_info = get_global_env_info()
        state = Tk.DISABLED if analysis_folder is None or env_info is None else Tk.NORMAL

    def show_denss_dialog( self ):
        from molass_legacy.DENSS.DenssGui import DenssGuiDialog
        analysis_folder = get_setting("analysis_folder")
        if analysis_folder is None:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            yn = MessageBox.askyesno("Output Folder Preference Query",
                    '"Analysis Result Folder" is not yet specified.\n'
                    "You can continue without it, but you have to specify the output folder each time.\n"
                    "Or, if you specify it in advance, successive output folders will be automatically created as shown below.\n"
                    "    .../analysis-nnn/DENSS/000\n"
                    "    .../analysis-nnn/DENSS/001\n"
                    "          ︙\n"
                    "Do you wish to continue simply (i.e., without auto-creating)?",
                    parent=self.parent)
            if not yn:
                return

        self.parent.update_analysis_folder()
        dialog = DenssGuiDialog(self.parent)
        dialog.show()

    def show_denss_manager(self):
        print('show_denss_manager')
        from molass_legacy.DENSS.DenssManagerDialog import show_manager_dialog
        show_manager_dialog(self.parent)

    def show_ed_viewer(self):
        print('show_ed_viewer')
        from molass_legacy.Saxs.EdViewer import EdViewer
        from molass_legacy.KekLib.OurMatplotlib import reset_to_default_style
        viewer = EdViewer(self.parent)
        viewer.show()
        reset_to_default_style()

    def show_saxs_simulator(self):
        print('show_saxs_simulator')
        from molass_legacy.Saxs.SaxsSimulator import SaxsSimulator
        from molass_legacy.KekLib.OurMatplotlib import reset_to_default_style
        simulator = SaxsSimulator(self.parent)
        simulator.show()
        reset_to_default_style()
