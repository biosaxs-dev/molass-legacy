# coding: utf-8
"""
    NoExcelWarningDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import re
import os
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

class NoExcelWarningDialog( Dialog ):
    def __init__(self, parent, com_error):
        self.parent = parent
        self.com_error = com_error
        Dialog.__init__( self, parent, "No Excel Warning", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        self.ro_text = ReadOnlyText(body_frame, height=6, relief=Tk.FLAT, fg='orange')
        self.ro_text.pack(padx=20, pady=10)

        warning_massage = (
            "No Excel instance is available in this environment for some reason\n"
            "such as no Excel has yet been properly installed.\n\n"
            "Be aware that some formatting will be skipped in the analysis result book\n"
            "to leave it look a little degraded.\n"
            )
        self.ro_text.insert(Tk.END, warning_massage)

        if self.com_error is not None:
            ro_text = ReadOnlyText(body_frame, height=20, relief=Tk.FLAT, fg='red')
            ro_text.pack(padx=20, pady=10)
            ro_text.insert(Tk.END, self.com_error )

    def buttonbox(self):

        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Do not repeat this message", width=24, command=self.do_not_repeat)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def do_not_repeat(self):
        set_setting('no_excel_warning', False)
        self.cancel()
