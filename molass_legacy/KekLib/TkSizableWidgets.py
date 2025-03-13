# coding: utf-8
"""

    ファイル名：   TkSizableWidgets.py

    処理内容：
        Frame( parent, width=..., height=... )
        と同様にサイズ指定できる Widget

        BlinkingLabel   配下のオブジェクトを点滅できる Label

"""
from __future__ import division, print_function, unicode_literals

import os
import sys
if sys.version_info > (3,):
    import tkinter as Tk
    from tkinter import _cnfmerge
else:
    import Tkinter as Tk
    from Tkinter import _cnfmerge

class SizableLabel( Tk.Frame ):
    def __init__( self, parent, width=60, height=26, **kwargs ):
        Tk.Frame.__init__( self, parent, width=width, height=height )
        self.label = Tk.Label( self, **kwargs )
        self.label.pack( fill=Tk.BOTH, expand=Tk.TRUE )

    def pack( self, **kwargs ):
        Tk.Frame.pack( self, **kwargs )
        self.pack_propagate( 0 )        # don't shrink

    def grid( self, **kwargs ):
        Tk.Frame.grid( self, **kwargs )
        self.pack_propagate( 0 )        # don't shrink

    def config( self, **kwargs ):
        state_ = kwargs.get( 'state' )
        if state_ is not None:
            self.label.config( state=state_ )
            del kwargs[ 'state' ]
        Tk.Frame.configure( self, **kwargs )

class SizableEntry( Tk.Frame ):
    def __init__( self, parent, width=60, height=26, **kwargs ):
        Tk.Frame.__init__( self, parent, width=width, height=height )
        self.entry = Tk.Entry( self, **kwargs )
        self.entry.pack( fill=Tk.BOTH, expand=Tk.TRUE )

    def pack( self, **kwargs ):
        Tk.Frame.pack( self, **kwargs )
        self.pack_propagate( 0 )        # don't shrink

    def grid( self, **kwargs ):
        Tk.Frame.grid( self, **kwargs )
        self.pack_propagate( 0 )        # don't shrink

    def config( self, **kwargs ):
        state_ = kwargs.get( 'state' )
        if state_ is not None:
            self.entry.config( state=state_ )
            del kwargs[ 'state' ]
        Tk.Frame.configure( self, **kwargs )

"""
    Thanks to: http://stackoverflow.com/questions/21419032/flashing-tkinter-labels

    revised version of TkSupplements.BlinkingFrame
    TODO: unification of BlinkingLabel and BlinkingFrame
"""
class BlinkingLabel( Tk.Frame ):
    def __init__( self, parent, width=60, height=26, **kwargs ):
        Tk.Frame.__init__( self, parent, width=width, height=height )

        self.label = Tk.Label( self, **kwargs )
        self.label.pack( fill=Tk.BOTH, expand=Tk.TRUE )

        self.switch     = False
        self.reverse    = False

    def pack( self, **kwargs ):
        Tk.Frame.pack( self, **kwargs )
        self.pack_propagate( 0 )        # don't shrink

    def grid( self, **kwargs ):
        Tk.Frame.grid( self, **kwargs )
        self.pack_propagate( 0 )        # don't shrink

    def flash(self):
        if not self.reverse and not self.switch:
            return

        try:
            bg = self.label.cget("background")
            fg = self.label.cget("foreground")
            self.label.configure(background=fg, foreground=bg)
        except:
            pass

        self.reverse = not self.reverse
        self.after(250, self.flash)

    def start(self):
        if not self.switch:
            self.switch = True
            self.flash()

    def stop(self):
        if self.switch:
            self.switch = False

    def is_blinking( self ):
        return self.switch
