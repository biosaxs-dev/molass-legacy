"""
    TkDebugDnd.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog
import tkinterDnD

class TkDebugDnd(Dialog):
    def __init__(self, parent, nrows=1, ncols=1):
        self.nrows = nrows
        self.ncols = ncols
        Dialog.__init__(self, parent, title="TkDebugDnd")

    def body(self, frame):
        for i in range(self.nrows):
            for j in range(self.ncols):
                label = Tk.Label(frame, text=str((i,j)), width=20, height=2, relief=Tk.RIDGE)
                label.grid(row=i, column=j)
                label.register_drop_target("*")
                label.bind("<<Drop>>", lambda event, i_=i, j_=j: self.on_drop(event, i_, j_))

                label.register_drag_source("*")
                label.bind("<<DragInitCmd>>", lambda event, i_=i, j_=j: self.on_drag(event, i_, j_))

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def on_drop(self, event, i, j):
        print("on_drop", (i,j), event.data)

    def on_drag(self, event, i, j):
        print("on_drag")
        return (tkinterDnD.COPY, "DND_Text", "on_drag text: " + str((i,j)))

