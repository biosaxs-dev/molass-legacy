# coding: utf-8
"""

    StdoutRedirector.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF

"""
import sys
import tkinter as Tk

class StdoutRedirector:
    def __init__(self, log_text):
        self.saved_stdout = sys.stdout
        self.log_text = log_text
        sys.stdout = self

    def write(self, string):
        try:
            if string is not None:
                self.log_text.insert(Tk.INSERT, string)
            self.log_text.see(Tk.END)
        except:
            # it seems that this can be called after destruction.
            pass

    def flush(self):
        pass

    def __del__(self):
        sys.stdout = self.saved_stdout
        # print('__del__ ok')
