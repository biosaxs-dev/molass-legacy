"""
    PropOptMenu.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from MenuButton import MenuButton

class PropOptMenu(Tk.Frame):
    def __init__(self, parent, dialog):
        Tk.Frame.__init__(self, parent)
        self.menu = MenuButton(self, "VP Analysis", [
                            ("using EGH", lambda: dialog.show_vp_analysis("EGH")),
                            ("using EDM", lambda: dialog.show_vp_analysis("EDM")),
                            ("using STC", lambda: dialog.show_vp_analysis("STC")),
                            ])
        self.menu.entryconfig(2, state=Tk.DISABLED)
        self.menu.pack()

    def config(self, **kwargs):
        self.menu.config(**kwargs)
