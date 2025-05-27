# coding: utf-8
"""
    ConcNormalizerDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import re
from difflib import SequenceMatcher
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, is_empty_val
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
try:
    import molass_legacy.KekLib.CustomMessageBox         as MessageBox
except:
    import OurMessageBox            as MessageBox
from SerialDataUtils import get_xray_files
from ConcNormalizer import find_conc_file, normalize_impl

class ConcNormalizerDialog(Dialog):
    def __init__(self, parent):
        self.busy = False
        Dialog.__init__(self, parent, "Concentration Normalizer", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        folder_entry_width = 60
        num_files_lable_width = 6

        grid_row = -1

        grid_row += 1
        label = Tk.Label(iframe, text="Folder Path")
        label.grid(row=grid_row, column=1)
        label = Tk.Label(iframe, text="File Name Pattern")
        label.grid(row=grid_row, column=2, padx=10)
        label = Tk.Label(iframe, text="Files")
        label.grid(row=grid_row, column=3)
        label = Tk.Label(iframe, text="Conc File")
        label.grid(row=grid_row, column=4)


        grid_row += 1
        label = Tk.Label(iframe, text="Input")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        self.in_folder = Tk.StringVar()
        self.in_files = []
        self.name_pattern = Tk.StringVar()
        self.num_files = Tk.IntVar()
        in_folder_entry = FolderEntry(iframe, textvariable=self.in_folder, width=folder_entry_width,
                                on_entry_cb=self.on_entry_in_folder)
        in_folder_entry.grid(row=grid_row, column=1)
        in_folder_entry.bind('<Leave>', lambda *args: self.on_entry_in_folder())

        label = Tk.Label(iframe, textvariable=self.name_pattern)
        label.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        num_files_label = Tk.Label(iframe, textvariable=self.num_files, width=num_files_lable_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3)

        self.conc_file = Tk.StringVar()
        conc_file_label = Tk.Label(iframe, textvariable=self.conc_file, width=12)
        conc_file_label.grid(row=grid_row, column=4)

        grid_row += 1
        label = Tk.Label(iframe, text="Output")
        label.grid(row=grid_row, column=0, sticky=Tk.W)
        self.out_folder = Tk.StringVar()
        out_folder_entry = FolderEntry(iframe, textvariable=self.out_folder, width=folder_entry_width,
                                on_entry_cb=self.on_entry_out_folder)
        out_folder_entry.grid(row=grid_row, column=1)
        out_folder_entry.bind('<Leave>', lambda *args: self.on_entry_out_folder())

        self.name_pattern_o = Tk.StringVar()
        entry = Tk.Entry(iframe, textvariable=self.name_pattern_o, width=24)
        entry.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        self.num_files_o = Tk.IntVar()
        num_files_lable = Tk.Label(iframe, textvariable=self.num_files_o, width=num_files_lable_width, justify=Tk.RIGHT)
        num_files_lable.grid(row=grid_row, column=3)

        grid_row += 1
        space = Tk.Label(iframe)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Progress")
        label.grid(row=grid_row, column=0)
        self.mpb = ttk.Progressbar(iframe,orient ="horizontal", length=700, mode="determinate")
        self.mpb.grid(row=grid_row, column=1, columnspan=4, padx=5)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        self.normalize_btn = Tk.Button(box, text="Normalize", width=10, command=self.normalize, state=Tk.DISABLED)
        self.normalize_btn.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        self._show()

    def on_entry_in_folder(self):
        in_folder = self.in_folder.get()
        files = get_xray_files(in_folder)
        self.in_files.clear()
        self.in_files += files
        if len(files) > 0:
            dir_, file = os.path.split(files[0])
            self.name_pattern.set(file)
        self.num_files.set( len(files) )
        file = find_conc_file(in_folder)
        self.conc_file.set(file)

    def on_entry_out_folder(self):
        name_pattern_o = self.name_pattern.get().replace('.dat', '_cn.dat')
        if is_empty_val(name_pattern_o):
            return

        self.name_pattern_o.set(name_pattern_o)

        out_folder = self.out_folder.get()
        if is_empty_val(out_folder):
            return

        self.normalize_btn.config(state=Tk.NORMAL)

    def normalize(self):
        num_files = len(self.in_files)

        self.busy = True

        conc_file = os.path.join(self.in_folder.get(), self.conc_file.get())

        name_pattern = self.name_pattern_o.get()
        sub_re = re.compile(r'^(.+_\d+)(\D.*)(\.dat)')
        m = sub_re.match(name_pattern)
        assert m
        # print(m.group(1), m.group(2))
        postfix = m.group(2)

        def name_changer(in_file):
            return re.sub(sub_re, lambda m: m.group(1) + postfix + m.group(3), in_file)

        normalize_impl(self.in_files, conc_file, self.out_folder.get(), name_changer, dialog=self)
        self.busy = False

    def is_busy(self):
        return self.busy
