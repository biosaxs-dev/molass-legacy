"""
    Optimizer.NaviFrame.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.V2PropOptimizer.PropOptMenu import PropOptMenu

class NaviFrame(Tk.Frame):
    def __init__(self, parent, canvas, arrows_only=False):
        self.parent = parent
        self.canvas = canvas
        Tk.Frame.__init__(self, parent)
        w = Tk.Frame(self)
        w.pack(side=Tk.LEFT)
        w = Tk.Button(self, text="|◀", width=5, command=canvas.get_first)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="◀◀", width=5, command=canvas.get_previous_best)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="◀", width=5, command=canvas.get_previous)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="Best", width=8, command=canvas.get_best)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="▶", width=5, command=canvas.get_next)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="▶▶", width=5, command=canvas.get_next_best)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="▶|", width=5, command=canvas.get_last)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        if arrows_only:
            return

        w = Tk.Button(self, text="Show Parameters", width=14, command=canvas.show_params)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="Complementary View", width=18, command=canvas.show_complementary_view)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = Tk.Button(self, text="MW Ratios", width=12, command=canvas.show_mw_integrity)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
        w = PropOptMenu(self, canvas)
        w.pack(side=Tk.LEFT, padx=10, pady=5)
