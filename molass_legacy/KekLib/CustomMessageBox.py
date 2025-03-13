# coding: utf-8
"""

    CustomMessageBox.py

    Copyright (c) 2018-2020, Masatsuyo Takahashi, KEK-PF

    https://msdn.microsoft.com/ja-jp/library/windows/desktop/ms645505(v=vs.85).aspx
    https://stackoverflow.com/questions/27257018/python-messagebox-with-icons-using-ctypes-and-windll
    https://www.skyarch.net/blog/?p=4812
    http://www.jasinskionline.com/windowsapi/ref/b/bm_click.html
    https://stackoverflow.com/questions/20995045/how-to-include-image-in-message-box-using-ctypes-in-python/21051586#21051586
"""
import os
import ctypes

SetActiveWindow     = ctypes.windll.user32.SetActiveWindow
SendMessage         = ctypes.windll.user32.SendMessageW
PostMessage         = ctypes.windll.user32.PostMessageW
MessageBox          = ctypes.windll.user32.MessageBoxW
FindWindow          = ctypes.windll.user32.FindWindowW
FindWindowEx        = ctypes.windll.user32.FindWindowExW
GetWindowRect       = ctypes.windll.user32.GetWindowRect
SetWindowPos        = ctypes.windll.user32.SetWindowPos

# icons
MB_OK               = 0x0
MB_OKCANCEL         = 0x01
MB_YESNOCANCEL      = 0x03
MB_YESNO            = 0x04
MB_HELP             = 0x4000
ICON_EXLAIM         = 0x30
ICON_INFORMATION    = 0x40
ICON_ERROR          = 0x10
ICON_WARNING        = 0x30
ICON_QUESTION       = 0x20
IDOK    = 1
IDCANCEL = 2
IDYES   = 6

from HookableMessageBox     import MyMessageBox, AlterIcon, AlterButton
# from TestUtils              import _RECT

class _RECT( ctypes.Structure ):
    _fields_ =  [
                    ( 'left',    ctypes.c_int ),
                    ( 'top',     ctypes.c_int ),
                    ( 'right',   ctypes.c_int ),
                    ( 'bottom',  ctypes.c_int ),
                ]

class InvokableButton:
    def __init__(self, parent, title):
        self.parent = parent
        self.title  = title

    def invoke(self):
        _hWndP  = FindWindow( 0, self.title )
        _hWndB  = FindWindowEx(_hWndP, 0, 0, "OK")
        SetActiveWindow(_hWndP)
        SendMessage(_hWndB, 0x00F5, 0, 0)       # 0x00F5 - BM_CLICK
        # PostMessage(_hWndB, 0x00F5, 0, 0)

current_messagebox = None

def get_current_messagebox():
    return current_messagebox

class CustomMessageBox_:
    def __init__(self, **options):
        self.options    = options
        self.parent     = options.get( 'parent' )
        self.title      = options.get( 'title' )
        self.message    = options.get( 'message' )
        self.type_      = options.get( 'type_' )
        self.icon_path  = options.get( 'icon_path' )
        self._hWnd      = 0 if self.parent is None else self.parent.winfo_id()
        self.ok_button  = InvokableButton( self, self.title )
        rect    = _RECT( 0, 0, 0, 0 )
        ret     = GetWindowRect( self._hWnd, ctypes.byref(rect) )
        # print( '__init__', rect.left, rect.top, rect.right, rect.bottom )
        self.parent_x   = ( rect.left + rect.right ) // 2
        self.parent_y   = ( rect.top + rect.bottom ) // 2

    def show( self ):
        icon_path = self.icon_path
        if icon_path is None:
            icon_path = os.environ.get('OUR_ICON_PATH')

        self.counter = 0

        def hook_callback( _hWnd ):
            if self.counter > 0:
                return

            self.counter += 1
            if icon_path is not None:
                AlterIcon( _hWnd, icon_path )
            # AlterButton( _hWnd )
            rect    = _RECT( 0, 0, 0, 0 )
            ret     = GetWindowRect( _hWnd, ctypes.byref(rect) )
            # print( 'hook_callback', rect.left, rect.top, rect.right, rect.bottom )
            width   = rect.right - rect.left
            height  = rect.bottom - rect.top
            ret = SetWindowPos( _hWnd, 0, self.parent_x, self.parent_y, width, height, 0 )
            # print( 'ret=', ret )
            if ret == 0:
                print( ctypes.FormatError() )

        ret = MyMessageBox(self._hWnd, self.message, self.title, self.type_, hook_callback)
        return ret

"""
    Only a part of functions are implemented. Remember to add ones if required.
"""

def showinfo( title, message, **options ):
    global current_messagebox
    type_ = MB_OK | ICON_INFORMATION
    current_messagebox = CustomMessageBox_( title=title, message=message, type_=type_, **options )
    ret = current_messagebox.show()
    return ret == IDOK

def showerror( title, message, **options ):
    global current_messagebox
    type_ = MB_OK | ICON_ERROR
    current_messagebox = CustomMessageBox_( title=title, message=message, type_=type_, **options )
    current_messagebox.show()

def showwarning( title, message, **options ):
    global current_messagebox
    type_ = MB_OK | ICON_WARNING
    current_messagebox = CustomMessageBox_( title=title, message=message, type_=type_, **options )
    current_messagebox.show()

def askyesno(title=None, message=None, **options):
    global current_messagebox
    type_ = MB_YESNO | ICON_QUESTION
    current_messagebox = CustomMessageBox_( title=title, message=message, type_=type_, **options )
    ret = current_messagebox.show()
    return ret == IDYES

def askokcancel(title=None, message=None, **options):
    global current_messagebox
    type_ = MB_OKCANCEL | ICON_QUESTION
    current_messagebox = CustomMessageBox_( title=title, message=message, type_=type_, **options )
    ret = current_messagebox.show()
    return ret == IDOK
