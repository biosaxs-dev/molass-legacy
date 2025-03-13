#-*- coding: utf-8 -*-
#!python
"""
    borrowed from
    https://stackoverflow.com/questions/20995045/how-to-include-image-in-message-box-using-ctypes-in-python/21051586#21051586
    and slightly modified
"""

from ctypes import *
from ctypes.wintypes import *
from os import path
import platform

#################################################################

RelPath = lambda file : path.join(path.dirname(path.abspath(__file__)), file)

#################################################################

GetModuleHandle = windll.kernel32.GetModuleHandleW
GetModuleHandle.restype = HMODULE
GetModuleHandle.argtypes = [LPCWSTR]

#################################################################

IMAGE_ICON = 1
LR_LOADFROMFILE = 0x00000010
LR_CREATEDIBSECTION = 0x00002000

LoadImage = windll.user32.LoadImageW
LoadImage.restype = HANDLE
LoadImage.argtypes = [HINSTANCE, LPCWSTR, UINT, c_int, c_int, UINT]

#################################################################

LRESULT = c_int64 if platform.architecture()[0] == "64bit" else c_long

SendMessage = windll.user32.SendMessageW
SendMessage.restype = LRESULT
SendMessage.argtypes = [HWND, UINT, WPARAM, LPARAM]

#################################################################

MB_OK = 0x00000000

MessageBox = windll.user32.MessageBoxW
MessageBox.restype  = c_int
MessageBox.argtypes = [HWND, LPCWSTR, LPCWSTR, UINT]

#################################################################

WH_CBT = 5
HCBT_ACTIVATE = 5
HOOKPROC = WINFUNCTYPE(LRESULT, c_int, WPARAM, LPARAM)

SetWindowsHookEx = windll.user32.SetWindowsHookExW
SetWindowsHookEx.restype = HHOOK
SetWindowsHookEx.argtypes = [c_int, HOOKPROC, HINSTANCE, DWORD]

#################################################################

CallNextHookEx = windll.user32.CallNextHookEx
CallNextHookEx.restype = LRESULT
CallNextHookEx.argtypes = [HHOOK, c_int, WPARAM, LPARAM]

#################################################################

GetCurrentThreadId = windll.kernel32.GetCurrentThreadId
GetCurrentThreadId.restype = DWORD
GetCurrentThreadId.argtypes = None

#################################################################

UnhookWindowsHookEx = windll.user32.UnhookWindowsHookEx
UnhookWindowsHookEx.restype = BOOL
UnhookWindowsHookEx.argtypes = [HHOOK]

#################################################################
# code starts here

def AlterIcon(_hWnd, lpszIcon):

  WM_SETICON = 0x0080
  ICON_BIG = 1

  hModel = GetModuleHandle(None)
  hIcon = LoadImage(hModel,
              RelPath(lpszIcon),
              IMAGE_ICON,
              0, 0,
              LR_LOADFROMFILE | LR_CREATEDIBSECTION)

  SendMessage(_hWnd, WM_SETICON, ICON_BIG, hIcon)

def AlterButton(_hWnd):
  #**********************************************************#
  # center button code
  WNDENUMPROC = WINFUNCTYPE(BOOL, HWND, LPARAM)

  EnumChildWindows = windll.user32.EnumChildWindows
  EnumChildWindows.restype = BOOL
  EnumChildWindows.argtypes = [HWND, WNDENUMPROC, LPARAM]

  GetClassName = windll.user32.GetClassNameW
  GetClassName.restype = HWND
  GetClassName.argtypes = [HWND, LPCWSTR, c_int]

  GetClientRect = windll.user32.GetClientRect 
  GetClientRect.restype = BOOL
  GetClientRect.argtypes = [HWND, POINTER(RECT)]

  MoveWindow = windll.user32.MoveWindow
  MoveWindow.restype = BOOL
  MoveWindow.argtypes = [HWND, c_int, c_int, c_int, c_int, BOOL]

  MapWindowPoints = windll.user32.MapWindowPoints
  MapWindowPoints.restype = c_int
  MapWindowPoints.argtypes = [HWND, HWND, POINTER(POINT), UINT]

  def EnumChildProc(hwnd, lParam):
    ClassName = (c_wchar * 7)()
    if GetClassName(hwnd, ClassName, 7) > 0:
      if ClassName.value.lower() == "button":
        wrect = RECT()
        GetClientRect(lParam, byref(wrect))
        brect = RECT()
        GetClientRect(hwnd, byref(brect))
        bpoint = RECT()
        MapWindowPoints(hwnd, lParam, cast(byref(bpoint), POINTER(POINT)), 2)
        MoveWindow(hwnd,
                  ((wrect.right - wrect.left) - (brect.right - brect.left)) // 2,
                  bpoint.top,
                  brect.right - brect.left,
                  brect.bottom - brect.top,
                  True)
        return False
    return True

  #**********************************************************#
  pEnumChildProc = WNDENUMPROC(EnumChildProc)
  EnumChildWindows(_hWnd, pEnumChildProc, _hWnd.value)
  #**********************************************************#

def MyMessageBox(hWnd, lpText, lpCaption, uType, hook_callback=None):
  hHook = HHOOK(None)
  #**********************************************************#

  def CBTProc(nCode, wParam, lParam):
    if nCode == HCBT_ACTIVATE:
      _hWnd = cast(wParam, HWND)
      if hook_callback is not None:
        hook_callback(_hWnd)
    CallNextHookEx(hHook, nCode, wParam, lParam)
    return 0

  # WARNING: don't pass HOOKPROC(CBTProc) directly to SetWindowsHookEx
  pCBTProc = HOOKPROC(CBTProc)

  hHook = SetWindowsHookEx(WH_CBT, pCBTProc, None, GetCurrentThreadId())
  ret = MessageBox(hWnd, lpText, lpCaption, uType)
  UnhookWindowsHookEx(hHook)
  return ret

# example of usage
if __name__ == '__main__':
    def hook_callback( hWnd ):
        AlterIcon(hWnd, "favicon.ico")
        AlterButton(hWnd)
    MyMessageBox(None, "Hello world!", "Title", MB_OK, hook_callback=hook_callback)