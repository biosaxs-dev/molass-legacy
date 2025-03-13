# coding: utf-8
"""

    SubProcess.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF

"""

import subprocess

"""
    thanks to https://github.com/SublimeText/VintageEx/blob/master/plat/windows.py
"""
def get_startup_info():
    # Hide the child process window.
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startupinfo

class Popen(subprocess.Popen):
    def __init__(self, cmd, **kwargs):
        subprocess.Popen.__init__(self, cmd, startupinfo=get_startup_info(), **kwargs)
