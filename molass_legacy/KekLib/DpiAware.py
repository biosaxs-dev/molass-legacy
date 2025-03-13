# coding: utf-8
"""
    DpiAware.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""

def set_dpi_aware():
    """
    setting suitable layout for high-resolution monitors

    https://github.com/asweigart/pyautogui/issues/33

    see also:
    Win32api is not giving the correct coordinates with GetCursorPos() in python
    https://stackoverflow.com/questions/32541475/win32api-is-not-giving-the-correct-coordinates-with-getcursorpos-in-python
    """
    from ctypes import windll
    user32 = windll.user32
    user32.SetProcessDPIAware()
