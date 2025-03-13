# coding: utf-8
"""
    OurSubprocess.py

    Copyright (c) 2016-2018, Masatsuyo Takahashi, KEK-PF
"""
import sys
from subprocess     import Popen, PIPE

def set_mode_and_get_flags():
    # learned at
    # https://www.activestate.com/blog/2007/11/supressing-windows-error-report-messagebox-subprocess-and-ctypes
    if sys.platform.startswith("win"):
        # Don't display the Windows GPF dialog if the invoked program dies.
        # See comp.os.ms-windows.programmer.win32
        #  How to suppress crash notification dialog?, Jan 14,2004 -
        #     Raymond Chen's response [1]
        import ctypes
        SEM_NOGPFAULTERRORBOX   = 0x0002    # From MSDN
        ctypes.windll.kernel32.SetErrorMode(SEM_NOGPFAULTERRORBOX);
        CREATE_NO_WINDOW  =0x08000000       # From MSDN
        subprocess_flags = CREATE_NO_WINDOW
    else:
        subprocess_flags = 0

    return subprocess_flags

def exec_subprocess( cmd, shell=False ):
    subprocess_flags = set_mode_and_get_flags()

    p = Popen( cmd, shell=shell, stdout=PIPE, stderr=PIPE,
                universal_newlines=True,
                creationflags=subprocess_flags )
    out, err = p.communicate()

    return out, err
