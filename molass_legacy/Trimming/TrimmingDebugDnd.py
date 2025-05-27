# coding: utf-8
"""

    TrimmingDebugDnd.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import re
import tkinterDnD
from TkDebugDnd import TkDebugDnd
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from .TrimmingInfo import TrimmingInfo

class TrimmingDebugDnd(TkDebugDnd):
    def __init__(self, dialog):
        self.dialog = dialog
        self.ret_info_re = re.compile(r"ret_info=(\[.+\])")
        TkDebugDnd.__init__(self, dialog, nrows=2)

    def on_drop(self, event, i, j):
        print("on_drop: ", (i,j), event.data)

        yn = MessageBox.askyesno("Question",
                        "Redrawing %s with %s\n"
                        "Ok?" % (str((i,j)), event.data),
                        parent=self)
        if yn:
            print("redraw")
            ret_info = self.decode_ret_info(event.data)
            self.dialog.redraw(ret_info)

    def decode_ret_info(self, str_data):
        m = self.ret_info_re.search(str_data)
        ret_info = eval(m.group(1))
        return ret_info

    def on_drag(self, event, i, j):
        print("on_drag: ")
        ret_info = self.dialog.get_trimming_info()
        return (tkinterDnD.COPY, "DND_Text", "on_drag text: " + str((i,j)) + "\nret_info=" + str(ret_info))
