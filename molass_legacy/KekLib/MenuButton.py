# coding: utf-8
"""

    MenuButton.py

    Copyright (c) 2019-2022, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter import Tk

class MenuButton(Tk.Button):
    def __init__(self, parent, text, command_list=[]):
        self.parent = parent
        Tk.Button.__init__(self, parent, text=text, command=self.popup_menu)
        self.create_menu(command_list)

    def create_menu(self, command_list):
        self.popup_menu = Tk.Menu(self, tearoff=0)
        for label, command in command_list:
            self.popup_menu.add_command(label=label, command=command)

    def popup_menu(self):
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        h = self.winfo_height()
        self.popup_menu.post(x, y+h)

    def entryconfig(self, *args, **kwargs):
        self.popup_menu.entryconfig(*args, **kwargs)
