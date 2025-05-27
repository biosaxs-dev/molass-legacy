"""

    Menus.JupyterMenu.py

    Copyright (c) 2022, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter import Tk

class JupyterMenu(Tk.Menu):
    def __init__(self, parent, menubar):
        self.parent = parent

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade(label="Jupyter", menu=self )
        self.add_command(label="Jupyter Lab", command=self.run_jupyterlab)

    def run_jupyterlab(self):
        from Jupyter.JupyterLab import run_jupyterlab
        self.entryconfig(0, state=Tk.DISABLED)
        self.update()
        run_jupyterlab(parent=self.parent)
        self.entryconfig(0, state=Tk.NORMAL)
