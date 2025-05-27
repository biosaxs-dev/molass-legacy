# coding: utf-8
"""
    CurveSaverDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry

class CurveSaverDialog(Dialog):
    def __init__(self, parent, curve_list, folder_init):
        self.curve_list = curve_list
        self.folder_init = folder_init
        Dialog.__init__(self, parent, "CurveSaverDialog", visible=False)

    def body(self, frame):

        iframe = Tk.Frame(frame)
        iframe.pack(padx=40, pady=10)
        guide_frame = Tk.Label(iframe, text='Check curves and press "Save".')
        guide_frame.pack()

        space = Tk.Frame(iframe, height=20)
        space.pack()

        table_frame = Tk.Frame(iframe)
        table_frame.pack()
        curve_id = Tk.Label(table_frame, text='Curve ID', width=20)
        curve_id.grid(row=0, column=0, padx=20, sticky=Tk.W)
        length = Tk.Label(table_frame, text='Length')
        length.grid(row=0, column=1)
        filename = Tk.Label(table_frame, text='Filename')
        filename.grid(row=0, column=2)

        self.var_list = []

        for i, rec in enumerate(self.curve_list, start=1):
            cbvar = Tk.IntVar()
            cbvar.set(1)
            cb = Tk.Checkbutton(table_frame, variable=cbvar, text=rec[0])
            cb.grid(row=i, column=0, sticky=Tk.W)
            length = Tk.Label(table_frame, text='%d' % len(rec[1]))
            length.grid(row=i, column=1, sticky=Tk.E)
            filename = Tk.StringVar()
            filename.set(rec[2])
            entry = Tk.Entry(table_frame, textvariable=filename, width=30 )
            entry.grid(row=i, column=2, sticky=Tk.W, padx=20)
            self.var_list.append([cbvar, filename])

        space = Tk.Frame(iframe, height=20)
        space.pack()

        folder_frame = Tk.Frame(iframe)
        folder_frame.pack()

        folder_label = Tk.Label(folder_frame, text="Save Folder: ")
        folder_label.grid(row=0, column=0)
        self.save_folder = Tk.StringVar()
        self.save_folder.set(self.folder_init.replace('\\', '/'))
        folder_entry = FolderEntry(folder_frame, textvariable=self.save_folder, width=60)
        folder_entry.grid(row=0, column=1)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Save", width=10, command=self.save, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        self._show()

    def save(self):
        folder = self.save_folder.get()

        if not os.path.exists(folder):
            from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
            mkdirs_with_retry(folder)

        for k, row in enumerate(self.var_list):
            cbvar, filename = row
            if cbvar.get() == 0:
                continue
            dir_path = os.path.join(folder, filename.get())
            np.savetxt(dir_path, self.curve_list[k][1])

        self.ok()
