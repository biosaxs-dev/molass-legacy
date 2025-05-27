"""
    OurTkinter.py

    Copyright (c) 2016-2024, Masatsuyo Takahashi, KEK-PF
"""
import os
import gc

import OurTk as Tk
from tkinter import TkVersion
# from tkinter.simpledialog import Dialog
from tkinter import filedialog as FileDialog
from tkinter.scrolledtext import ScrolledText
import tkinter.font as Font
import tkinter.ttk as ttk
# import tkinter.tix as tix

from molass_legacy.KekLib.TkSupplements      import tk_set_icon_portable     # for backward compatibility

try:
    from idlelib.tooltip import Hovertip as ToolTip
except:
    from idlelib.ToolTip import ToolTip

# from molass_legacy.KekLib.TkSupplements      import tk_set_icon_portable

def is_empty_val( val ):
    return val is None or val == '' or val[0] == '<' or val == 'None' 

def checkFolder( parent, folder_var, folder_entry ):
    # TODO: include these checks in a custom widget class
    def set_error( entry ):
        entry.config( fg='red' )
        entry.focus_force()

    ok_ = False
    folder = folder_var.get()
    if is_empty_val( folder ):
        set_error( folder_entry )
        MessageBox.showerror( "Input Folder Error", "The Input Folder is required.", parent=parent )
    elif not os.path.exists( folder ):
        set_error( folder_entry )
        MessageBox.showerror( "Input Folder Error", "'%s' does not exist." % in_folder, parent=parent )
    elif not os.path.isdir( folder ):
        set_error( folder_entry )
        MessageBox.showerror( "Input Folder Error", "'%s' is not a folder." % in_folder, parent=parent )
    else:
        folder_entry.config( fg='black' )
        ok_ = True

    return ok_


class Dialog(Tk.Toplevel):

    '''Class to open dialogs.

    This class is intended as a base class for custom dialogs
    '''

    def __init__(self, parent, title=None, auto_geometry=True, geometry_cb=None, visible=True, block=True, location=None ):

        '''Initialize a dialog.

        Arguments:

            parent -- a parent window (the application window)

            title -- the dialog title
        '''
        Tk.Toplevel.__init__(self, parent)

        self.withdraw() # remain invisible for now
        # If the master is not viewable, don't
        # make the child transient, or else it
        # would be opened withdrawn
        if parent.winfo_viewable():
            self.transient(parent)

        if title:
            self.title(title)

        self.parent = parent
        self.location = location

        self.result = None

        body = Tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)

        self.buttonbox()

        if not self.initial_focus:
            self.initial_focus = self

        self.protocol("WM_DELETE_WINDOW", self.cancel)

        if auto_geometry:
            self._adjust_geometry()

        if geometry_cb is not None:
            geometry_cb()

        if not visible:
            return

        self._show(block=block)

    def _show(self, block=True, pause=None, wait=True, raise_=False):
        self.deiconify() # become visible now

        self.initial_focus.focus_set()

        # wait for window to appear on screen before calling grab_set
        try:
            self.wait_visibility()
        except:
            if raise_:
                raise Exception()
            """
            ignore because this does not seem to be a problem
            _tkinter.TclError: window "..." was deleted before its visibility changed
            """
            pass

        try:
            self.grab_set()
        except:
            if raise_:
                raise Exception()
            """
            ignore because this does not seem to be a problem
            _tkinter.TclError: grab failed: window "..." already has grab
            """
            pass

        if not block:
            interval = 500 if pause is None else int(pause*1000)
            self.parent.after( interval, self.ok )

        if wait:
            self.wait_window(self)
        else:
            # user should self.wait_window instead
            pass

    def destroy(self):
        '''Destroy the window'''
        self.initial_focus = None
        Tk.Toplevel.destroy(self)

        # to avoid "Tcl_AsyncDelete: async handler deleted by the wrong thread" error
        #   from https://pysimplegui.readthedocs.io/en/latest/#multiple-threads
        gc.collect()

    #
    # construction hooks

    def body(self, master):
        '''create dialog body.

        return widget that should have initial focus.
        This method should be overridden, and is called
        by the __init__ method.
        '''
        pass

    def buttonbox( self, frame=None ):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        if frame is None:
            box = Tk.Frame(self)
            box.pack()
        else:
            box = frame

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    #
    # standard button semantics

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        try:
            self.apply()
        finally:
            self.cancel()

    def cancel(self, event=None):

        # put focus back to the parent window
        if self.parent is not None:
            self.parent.focus_set()
        self.destroy()

    #
    # command hooks

    def validate(self):
        '''validate the data

        This method is called automatically to validate the data before the
        dialog is destroyed. By default, it always validates OK.
        '''

        return 1 # override

    def apply(self):
        '''process the data

        This method is called automatically to process the data, *after*
        the dialog is destroyed. By default, it does nothing.
        '''

        pass # override

    def _adjust_geometry(self):
        """
        use this method with Tk-after method before _show
            when
            used with visible=False
            and the window size is not yet what is expected
                during the above __init__
        """

        parent = self.parent
        location = self.location

        rootx = parent.winfo_rootx()
        rooty = parent.winfo_rooty()
        if location is None:
            self.geometry("+%d+%d" % (rootx+50,rooty+30))
        else:
            vloc, hloc = location.split(" ")
            self.update()
            from molass_legacy.KekLib.TkUtils import split_geometry
            W, H, X, Y = split_geometry(parent.geometry())
            w, h, x, y = split_geometry(self.geometry())
            # print([rootx, rooty])
            # print([W, H, X, Y], [w, h, x, y])
            if vloc == 'lower':
                offsety = H - h - 100
            elif vloc == 'center':
                offsety = H//2 - h//2
            else:
                offsety = 30

            if hloc == 'right':
                offsetx = W - w - 100
                self.geometry("+%d+%d" % (rootx+offsetx, rooty+offsety))
            elif hloc == 'center':
                offsetx = W//2 - w//2
            else:
                offsetx = 50
 
            self.geometry("+%d+%d" % (rootx+offsetx, rooty+offsety))
