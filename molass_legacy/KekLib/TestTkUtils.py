# -*- coding: utf-8 -*-
"""

    ファイル名：   TestTkUtils.py

    処理内容：

        

"""
from __future__ import division, print_function, unicode_literals

import os
import sys
import re

def split_geometry( geometry ):
    n = geometry.split('+')
    s = n[0].split('x')
    w = int( s[0] )
    h = int( s[1] )
    x = int( n[1] )
    y = int( n[2] )
    return [ w, h, x, y ]

def join_geometry( w, h, x, y ):
    return '%dx%d+%d+%d' % ( w, h, x, y )

def geometry_fix( top, x, y ):
    W, H, X, Y = split_geometry( top.geometry() )
    top.geometry( join_geometry( W, H, x, y ) )
    top.update()

"""
    URL: http://effbot.org/tkinterbook/wm.htm
"""
def parsegeometry(geometry):
    m = re.match("(\d+)x(\d+)([-+]\d+)([-+]\d+)", geometry)
    if not m:
        raise ValueError("failed to parse geometry string")
    return map(int, m.groups())
