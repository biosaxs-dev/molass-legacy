# coding: utf-8
"""

    ファイル名：   TkSupplements.py

    処理内容：

        BlinkingFrame   配下のオブジェクトを点滅できる Frame

    Copyright (c) 2016-2021, Masatsuyo Takahashi, KEK-PF

"""
from __future__ import division, print_function, unicode_literals

import os
import sys
import inspect
if sys.version_info > (3,):
    import tkinter as Tk
    from tkinter import _cnfmerge
else:
    import Tkinter as Tk
    from Tkinter import _cnfmerge

def pad_text( text, length ):
    return text + ' ' * ( length - len(text) )

from molass_legacy.KekLib.BasicUtils import exe_name, get_caller_module

"""
    Icon files are supposed to be placed in the same folder,
    named as follows.
        name.ico on Windows
        name.png on Linux
"""

cashed_image = None

def tk_set_icon_portable( parent, name=None, module=None, use_cashed_image=False ):
    pass

def set_icon( parent ):
    iconpath = os.environ.get('OUR_ICON_PATH')
    if iconpath is None:
        return

    if os.name == 'nt':         # i.e. on Windows
        if not os.path.exists(iconpath): return
        # see wm_iconbitmap in tkinter.__init__.py
        # see also https://wiki.tcl-lang.org/page/wm+iconbitmap
        parent.iconbitmap(default=iconpath)
    else:
        if not os.path.exists(iconpath): return
        img = Tk.Image( "photo", file=iconpath )
        parent.tk.call( 'wm', 'iconphoto', parent._w, img )

"""
    Thanks to: http://stackoverflow.com/questions/21419032/flashing-tkinter-labels
"""
FLASH_STOP_NORMAL   = 0
FLASH_STOP_JUST     = 1

class BlinkingFrame(Tk.Frame):
    def __init__( self, parent, object_spec_array=None, grid=False,
                    start_proc=None, stop_proc=None,
                    flash_stop_type=FLASH_STOP_NORMAL,
                    debug=False ):
        Tk.Frame.__init__(self, parent)

        self.objects = []

        if object_spec_array is not None:
            for object_spec in object_spec_array:
                obj, pack_info  = object_spec
                self.objects.append( obj )
                if grid:
                    obj.grid( **pack_info )
                else:
                    obj.pack( **pack_info )

        # print('self.objects=', self.objects)

        self.start_proc = start_proc
        self.stop_proc  = stop_proc
        self.switch     = False
        self.reverse    = False
        self.debug      = debug
        self.flash_stop_type    = flash_stop_type
        self.init_bgfg = None

    def flash(self):
        if self.flash_stop_type == FLASH_STOP_NORMAL:
            return_cond = ( not self.reverse and not self.switch )
        else:
            return_cond = not self.switch

        if return_cond:
            return

        for obj in self.objects:
            try:
                bg = obj.cget("background")
                fg = obj.cget("foreground")
                obj.configure(background=fg, foreground=bg)
            except:
                continue

        self.reverse = not self.reverse
        self.after(250, self.flash)

    def start(self):
        if self.init_bgfg is None:
            self.init_bgfg = [ (obj.cget("background"), obj.cget("foreground")) for obj in self.objects ]
        if not self.switch:
            if self.debug:
                print( 'start blinking' )
            self.switch = True
            if self.start_proc:
                self.start_proc()
            self.flash()

    def stop(self):
        self.restore_init()
        if self.switch:
            if self.debug:
                print( 'stop blinking' )
            self.switch = False
            if self.stop_proc:
                self.stop_proc()

    def restore_init(self):
        if self.init_bgfg is not None:
            for k, obj in enumerate(self.objects):
                bg, fg = self.init_bgfg[k]
                obj.configure(background=bg, foreground=fg)
        self.reverse = False

    def is_blinking( self ):
        return self.switch

"""
    pixel-sizable button widget
    # cf. http://www-acc.kek.jp/kekb/control/Activity/Python/TkIntro/introduction/button.htm
    TODO: doesn't seem to work when placed by grid
"""
class SlimButton(Tk.Frame):
    def __init__( self, parent, text, command, state='normal', height=26, width=60 ):
        Tk.Frame.__init__( self, parent, height=height, width=width )
        self.button = Tk.Button( self, text=text, command=command, state=state )
        self.button.pack( fill=Tk.BOTH, expand=1 )

    def pack( self, cnf={}, **kw ):
        cnf = _cnfmerge((cnf, kw))      # cf. tkinter.dialog.py
        Tk.Frame.pack( self, cnf )
        self.pack_propagate( 0 )        # don't shrink

    def invoke( self ):
        # TODO: other args?
        self.button.invoke()

    def configure( self, cnf={}, **kw ):
        cnf = _cnfmerge((cnf, kw))      # cf. tkinter.dialog.py
        self.button.configure( cnf )

    def config( self, cnf={}, **kw ):
        cnf = _cnfmerge((cnf, kw))      # cf. tkinter.dialog.py
        self.button.configure( cnf )

    def cget( self, *args, **kw ):
        return self.button.cget( *args, **kw )

if __name__ == "__main__":
    root = Tk.Tk()

    objects = []
    objects.append( [ Tk.Label( root, text="Hello, world" ), { "fill":"both", "expand":True} ] )

    bf = BlinkingFrame( root, objects )
    Tk.Button( text="Start", command=lambda: bf.start() ).pack()
    Tk.Button( text="Stop", command=lambda: bf.stop() ).pack()

    root.mainloop()
    root.destroy()
