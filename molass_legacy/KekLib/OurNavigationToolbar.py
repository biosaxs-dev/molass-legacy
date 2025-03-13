# -*- coding: utf-8 -*-
"""

    ファイル名：   OurNavigationToolbar.py

    処理内容：

       NavigationToolbar2TkAgg のカスタマイズ（Windows のみ）
       Ubuntu では元の NavigationToolbar2TkAgg で OK

    Copyright (c) 2016-2019, Masatsuyo Takahashi, KEK-PF

"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import numpy as np
import pylab as pl
import tkinter as Tk
import matplotlib

if matplotlib.__version__ >= '2.2':
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
else:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar2Tk

class NavigationToolbar( NavigationToolbar2Tk ):
    def __init__( self, canvas, window, show_mode=True ):
        self.show_mode = show_mode
        NavigationToolbar2Tk.__init__( self, canvas, window )

    """
        matplotlib.backend_bases.NavigationToolbar2.mouse_move を override
    """
    def mouse_move(self, event):
        self._set_cursor(event)

        if event.inaxes and event.inaxes.get_navigate():

            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
            except (ValueError, OverflowError):
                pass
            else:

                # s += している部分を削除

                if self.show_mode and len(self.mode):
                    self.set_message('%s, %s' % (self.mode, s))
                else:
                    self.set_message(s)
        else:
            if self.show_mode:
                self.set_message(self.mode)
            else:
                self.set_message('')
