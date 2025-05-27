# coding: utf-8
"""
    SaferSpinbox.py

    Copyright (c) 2018-2019, Masatsuyo Takahashi, KEK-PF
"""
from time import time
import tkinter as Tk

class SaferSpinbox(Tk.Spinbox):
    def __init__(self, parent, **kwargs):
        self.inverval = kwargs.pop("inverval", 1)
        command = kwargs.pop("command", None)
        if command is None:
            command = self.command
        Tk.Spinbox.__init__(self, parent, command=command, **kwargs)
        self.user_tracer = None
        self.var = kwargs.get("textvariable", None)
        self.from_ = kwargs.get("from_", 1)
        self.to = kwargs.get("to", 9)
        self.change_history = []

    def config(self, **kwargs):
        # todo: better overriding
        self.var = kwargs.get("textvariable", self.var)
        self.from_ = kwargs.get("from_", self.from_)
        self.to = kwargs.get("to", self.to)
        Tk.Spinbox.config(self, **kwargs)

    def command(self):
        if self.user_tracer is not None:
            # always call user_tracer on arrow button click
            self.user_tracer(*self.last_args)
            self.change_history = []

    def set_tracer(self, tracer):
        self.user_tracer = tracer

    def tracer(self, *args):
        try:
            value = self.var.get()
        except:
            # from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            # etb = ExceptionTracebacker()
            # print(etb.last_lines())
            return

        self.last_value = value
        self.last_args = args
        self.change_history.append(time())
        if self.user_tracer is not None:
            self.after(1100, self.user_tracer_caller)

    def user_tracer_caller(self):
        if len(self.change_history) == 0:
            return

        interval = time() - self.change_history[-1]
        if interval > self.inverval:
            if self.validate():
                self.user_tracer(*self.last_args)
                self.change_history = []
        else:
            self.error_notified = False
            self.after(100, self.user_tracer_caller)

    def validate(self):
        try:
            value = self.var.get()
            if self.from_  <= value and value <= self.to:
                return True

            if not self.error_notified:
                import molass_legacy.KekLib.CustomMessageBox as MessageBox
                MessageBox.showerror( "Value Error",
                    "Values for this spinbox must be within %d and %d." % (self.from_, self.to),
                    parent=self)
                self.error_notified = True
        except:
            pass

        return False
