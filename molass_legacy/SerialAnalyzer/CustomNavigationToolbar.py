# coding: utf-8
"""
    CustomNavigationToolbar.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
from molass_legacy.KekLib.OurMatplotlib  import NavigationToolbar

class CustomNavigationToolbar( NavigationToolbar ):
    def __init__( self, canvas, window ):
        NavigationToolbar.__init__( self, canvas, window )

    """
        NavigationToolbar.mouse_move を override
    """
    def mouse_move(self, event):
        self._set_cursor(event)
        self.set_message(self.mode)
