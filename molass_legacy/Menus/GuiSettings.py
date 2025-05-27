"""

    GuiSettings.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy._MOLASS.SerialSettings import get_setting

class GuiSettingsMenu(Tk.Menu):
    def __init__(self, parent, menubar ):
        self.parent = parent

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade( label="Settings", menu=self )
        self.add_command( label="Basic Settings", command=self.show_settings_dialog )
        self.add_command( label="Specialist Options", command=self.show_specialist_dialog )
        self.add_command( label="Check Environment", command=self.check_env)

    def show_settings_dialog(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.SerialAnalyzer.SettingsDialog
            reload(molass_legacy.SerialAnalyzer.SettingsDialog)
        from molass_legacy.SerialAnalyzer.SettingsDialog import SettingsDialog
        while True:
            dialog = SettingsDialog( self.parent, 'Settings' )
            if dialog.applied is None:
                # i.e., some errors exist
                # see SettingsDialog.py
                continue
            else:
                # i.e., dialog.applied is True (without any error) or False
                break

        if dialog.applied:
            self.parent.after(500, self.parent.re_prepare_serial_data)

    def show_specialist_dialog(self, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.SerialAnalyzer.SpecialistOptions
            reload(molass_legacy.SerialAnalyzer.SpecialistOptions)
        from molass_legacy.SerialAnalyzer.SpecialistOptions import SpecialistOptionsDialog
        prev_use_xray_conc = get_setting( 'use_xray_conc' )
        prev_apply_backsub = get_setting( 'apply_backsub' )
        dialog = SpecialistOptionsDialog( self.parent, 'Specialist Options' )
        dialog.show()
        if dialog.applied:
            self.parent.update_uv_folder_entry_state()
            self.parent.update_analysis_button_state()
            if ( get_setting( 'use_xray_conc' ) != prev_use_xray_conc
                or get_setting( 'apply_backsub' ) != prev_apply_backsub
                ):
                self.parent.loader.reset_current_status()

    def check_env(self):
        print('check_env')
        from molass_legacy.Env.CheckEnvDialog import CheckEnvDialog
        dialog = CheckEnvDialog(self.parent)
        dialog.show()
