# coding: utf-8
"""

    MultiMonitor.py

    Copyright (c) 2019-2021, Masatsuyo Takahashi, KEK-PF

"""
import os
from ctypes import windll, Structure, c_long, byref

PYTHON_DEMO_MONITOR     = 'PYTHON_DEMO_MONITOR'

"""
learned at
https://stackoverflow.com/questions/33685534/trying-to-retrieve-the-active-monitor-id-in-python/33760154
"""

"""
from
    Getting cursor position in Python
    https://stackoverflow.com/questions/3698635/getting-cursor-position-in-python

    changed from c_ulong to c_long
    to handle cases where the second monitor
    is placed to the left side of the first.
"""
class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

def get_mouse_position():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt.x, pt.y

monitors = None
max_monitor = None

def get_monitor_list():
    global monitors
    if monitors is None:
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
        except:
            print( 'screeninfo is not installed.' )
            monitors = []
    return monitors

def get_max_monitor():
    global monitors, max_monitor

    if max_monitor is not None: return max_monitor

    monitors = get_monitor_list()

    for m in monitors:
        # print( 'get_max_monitor', str( m ) )
        if max_monitor is None or m.width + m.height > max_monitor.width + max_monitor.height:
            max_monitor = m

    return max_monitor


selected_monitor = None

def get_selected_monitor():
    global selected_monitor

    if selected_monitor is not None:
        return selected_monitor

    selected_monitor = get_max_monitor()
    demo_monitor = os.environ.get( PYTHON_DEMO_MONITOR )
    # print( 'demo_monitor=', demo_monitor )
    if demo_monitor is None:
        from MultiMonitor import get_mouse_position
        x, y = get_mouse_position()
        for m in monitors:
            # print(m.x, m.y, m.width)
            if x >= m.x and x <= m.x + m.width:
                selected_monitor = m
                break
    else:
        selected_monitor = monitors[ int(demo_monitor) ]
        # print( 'selected monitor=', monitor )
    return selected_monitor
