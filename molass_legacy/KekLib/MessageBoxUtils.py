# coding: utf-8
"""

    MessageBoxUtils.py

    Copyright (c) 2019, Masatsuyo Takahashi, KEK-PF

"""

"""
    References:
        (1) Win32 Python: Getting all window titles
            https://sjohannes.wordpress.com/tag/win32/

        (2) MessageBoxをプログラムから閉じたい
            http://tarulab.blogspot.com/2014/03/messagebox.html

        (3) 第２３章 ダイアログボックスの研究 その１
            http://mrxray.on.coocan.jp/Halbow/Chap23.html
"""
import ctypes

EnumWindows = ctypes.windll.user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
GetWindowText = ctypes.windll.user32.GetWindowTextW
GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
IsWindowVisible = ctypes.windll.user32.IsWindowVisible

WM_COMMAND = 0x0111
ID_OK = 1
ID_CANCEL = 2
ID_ABORT = 3
ID_RETRY = 4
ID_IGNORE = 5
ID_YES = 6
ID_NO = 7
ID_CLOSE = 8
ID_HELP = 9

COMMAND_DICT = {'Y':ID_YES, 'N':ID_NO, 'O':ID_OK, 'C':ID_CANCEL}

class Window:
    def __init__(self, hwnd, title):
        self.hwnd = hwnd
        self.title = title

    def send_command(self, command_id):
        ctypes.windll.user32.SendMessageW(self.hwnd, WM_COMMAND, command_id, 0)

def find_windows(pattern):
    windows = []   
    def foreach_window(hwnd, lParam):
        if IsWindowVisible(hwnd):
            length = GetWindowTextLength(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            if buff.value.find(pattern) >=0 :
                windows.append(Window(hwnd, buff.value))
        return True

    EnumWindows(EnumWindowsProc(foreach_window), 0)

    return windows

def reply_messagebox(title, reply):
    windows = find_windows(title)
    assert len(windows) == 1
    command_id = COMMAND_DICT[reply]
    windows[0].send_command(command_id)

def window_exists(pattern):
    return len(find_windows(pattern)) > 0

if __name__ == '__main__':
    # show a messagebox titled "Test Title"
    # and try one of the following
    reply_messagebox('Test Title', 'Y')     # when showing Info
    reply_messagebox('Test Title', 'Y')     # when asking Yes/No
    reply_messagebox('Test Title', 'O')     # when asking Ok/Cancel
