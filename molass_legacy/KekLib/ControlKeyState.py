# -*- coding: utf-8 -*-
"""

    ファイル名：   ControlKeyState.py

    処理内容：

       shift キーと ctrl キーの状態管理

"""
from __future__ import division, print_function, unicode_literals

ctrl_is_held = None

def set_ctrl_key_state( state ):
    global ctrl_is_held
    ctrl_is_held = state

def get_ctrl_key_state():
    global ctrl_is_held
    return ctrl_is_held

shift_is_held = None

def set_shift_key_state( state ):
    global shift_is_held
    shift_is_held = state

def get_shift_key_state():
    global shift_is_held
    return shift_is_held
