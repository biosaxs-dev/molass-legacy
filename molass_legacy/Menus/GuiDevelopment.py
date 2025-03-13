"""
    GuiDevelopment.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk

class GuiDevelopmentMenu(Tk.Menu):
    def __init__(self, parent, menubar ):
        self.parent = parent

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade( label="Development", menu=self )
        self.add_command( label="Test for input data in all subfolders in a folder", command=self.show_tester_dialog )
        self.add_command( label="Check diffences of logfiles in report folders", command=self.log_diff_dialog )
        self.add_command( label="Developer Options", command=self.show_developer_dialog )
        self.add_command( label="Restore from Result", command=self.restore_settings )
        self.add_command( label="Run Automation Script", command=self.run_automation_script )

    def show_tester_dialog(self):
        from molass_legacy.Test.TesterDialog import TesterDialog
        dialog = TesterDialog( self.parent, 'Test' )
        dialog.show()

    def log_diff_dialog(self):
        from molass_legacy.Test.TesterDialog import LogDiffDialog
        dialog = LogDiffDialog( self.parent, 'Diff logfiles' )
        dialog.show()

    def show_developer_dialog(self):
        from molass_legacy.SerialAnalyzer.DeveloperOptions import DeveloperOptionsDialog
        self.dev_dialog = DeveloperOptionsDialog( self.parent, 'Developer Options' )
        self.dev_dialog.show()

    def restore_settings(self):
        from molass_legacy.SerialAnalyzer.SettingsDialog  import RestoreSettingDialog
        dialog = RestoreSettingDialog( self.parent, 'Restore Settings' )
        if dialog.applied:
            self.parent.refresh()

    def run_automation_script(self):
        from importlib import reload
        import Test.Automation
        reload(Test.Automation)
        from molass_legacy.Test.Automation import AutomationDialog
        dialog = AutomationDialog(self.parent, self.parent)
        dialog.show()
