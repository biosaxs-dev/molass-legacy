"""
    _MOLASS.AppSingleInstance.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import sys
from molass_legacy.KekLib.SingleInstance import SingleInstance
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import molass_legacy.KekLib.OurMessageBox as MessageBox

def single_instance_check(lock_path):
    si = None
    try:
        si = SingleInstance(lock_path=lock_path, raise_=False)
        if si.is_running is None:
            print('unexpected state for', lock_path)
        elif si.is_running:
            from molass_legacy.SerialAnalyzer.SerialTestUtils import get_tk_root
            root = get_tk_root()
            ret = MessageBox.askyesno("Already Running Notification",
                '_MOLASS seems to be already running!\n'
                '\n'
                'Multiple instance execution is not yet well supported.\n'
                'However, you can continue at your own risk.\n'
                '\n'
                'Would you really like to continue?'
                ,
                parent=root)
            if ret:
                pass
            else:
                sys.exit("User canceled a multiple instance execution!")
    finally:
        pass

    return si
