# coding: utf-8
"""

    OurToplevel.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF

"""

import tkinter as Tk

class OurToplevel(Tk.Toplevel):
    def __init__(self, parent, title):
        Tk.Toplevel.__init__(self, parent)
        self.wm_title(title)
        body = Tk.Frame(self)
        self.initial_focus = self.body(body)
        body.pack(padx=5, pady=5)
        self.buttonbox()
        rootx = parent.winfo_rootx()
        rooty = parent.winfo_rooty()
        self.geometry("+%d+%d" % (rootx+50,rooty+50))
        self.update()
        self.protocol( "WM_DELETE_WINDOW", self.close )
