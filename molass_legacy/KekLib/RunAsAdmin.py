# coding: utf-8
"""
    RunAsAdmin.py
    Copyright (c) 2019, Masatsuyo Takahashi, KEK-PF
"""

"""
    borrowed from
    https://superuser.com/questions/615654/run-exe-file-via-python-as-administrator
"""
import ctypes, sys, os

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def runas():
    if is_admin():
        import time
        print("Running as admin.")
        time.sleep(3)
    else:
        # Re-run the program with admin rights
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        MessageBox.askokcancel("RunAsAdmin",
            'Please trust this program and reply "Yes" to the next dialog.'
            )
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

if __name__ == '__main__':
    runas()
