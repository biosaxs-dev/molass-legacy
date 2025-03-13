"""

    GuiParts.UvFolderEntry.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF

"""
import os
from molass_legacy.KekLib.OurTkinter import is_empty_val
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from molass_legacy._MOLASS.SerialSettings import get_setting

class UvFolderEntry(FolderEntry):
    def check( self, *args ):
        if get_setting( 'use_xray_conc' ) == 1:
            return True

        f = self.variable.get()
        ret = False
        if is_empty_val( f ):
            self.set_error()
            message = (   'This folder input is required.\n'
                        + 'Enable "use xray-proportional concentration" in the "Specialist Options" '
                        + 'if you would like to analyze without UV-data.'
                        )
            MessageBox.showerror( "Folder Input Error", message, parent=self.parent )
        elif not os.path.exists( f ):
            self.set_error()
            MessageBox.showerror( "Folder Input Error", "'%s' does not exist." % f, parent=self.parent )
        elif not os.path.isdir( f ):
            self.set_error()
            MessageBox.showerror( "Folder Input Error", "'%s' is not a folder." % f, parent=self.parent )
        else:
            self.entry.config( fg='black' )
            ret = True

        if not ret:
            self.entry.focus_force()

        return ret
