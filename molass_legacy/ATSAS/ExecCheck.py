"""
    ATSAS.ExecCheck.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import os
from .AutoRg import autorg_exe_array

def atsas_exec_check(parent, debug=False):
    if len(autorg_exe_array) == 0:
        return False

    exe_path = autorg_exe_array[0]

    ok = os.path.exists(exe_path)

    if not ok or debug:
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        folder = os.path.dirname(os.path.dirname(exe_path))
        MessageBox.showerror("ATSAS Installation Info Error",
            'Somethig is wrong with the ATSAS installation under\n'
            '"%s"\n'
            'Check the installation status at "Settings/Basic Settings"\n'
            'and try "Update" button if it is not the latest.'
            % folder,
            parent=parent
            )

        if debug:
            ok = False

    return ok
