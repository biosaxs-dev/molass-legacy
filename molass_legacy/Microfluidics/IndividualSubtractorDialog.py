# coding: utf-8
"""
    IndividualSubtractorDialog.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
import os
from difflib import SequenceMatcher
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, is_empty_val
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
try:
    import molass_legacy.KekLib.CustomMessageBox         as MessageBox
except:
    import OurMessageBox            as MessageBox
from SerialDataUtils import get_xray_files
from DataSubtractor import subtract_impl

class IndividualSubtractorDialog(Dialog):
    def __init__(self, parent):
        self.busy = False
        Dialog.__init__(self, parent, "Individual Subtractor", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        folder_entry_width = 60
        num_files_label_width = 6

        grid_row = -1

        grid_row += 1
        label = Tk.Label(iframe, text="Folder Path")
        label.grid(row=grid_row, column=1)
        label = Tk.Label(iframe, text="File Name Pattern")
        label.grid(row=grid_row, column=2, padx=10)
        label = Tk.Label(iframe, text="Files")
        label.grid(row=grid_row, column=3)


        grid_row += 1
        label = Tk.Label(iframe, text="Sample Data")
        label.grid(row=grid_row, column=0)

        self.in_folder1 = Tk.StringVar()
        self.in_files1 = []
        self.name_pattern1 = Tk.StringVar()
        self.num_files1 = Tk.IntVar()
        self.in_folder1_cb = lambda: self.on_entry_in_folder(self.in_folder1, self.in_files1, self.name_pattern1, self.num_files1)
        in_folder1_entry = FolderEntry(iframe, textvariable=self.in_folder1, width=folder_entry_width,
                                on_entry_cb=self.in_folder1_cb)
        in_folder1_entry.grid(row=grid_row, column=1)
        in_folder1_entry.bind('<Leave>', lambda *args:self.in_folder1_cb())

        label = Tk.Label(iframe, textvariable=self.name_pattern1)
        label.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        num_files_label = Tk.Label(iframe, textvariable=self.num_files1, width=num_files_label_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3)

        grid_row += 1
        label = Tk.Label(iframe, text="Buffer Data")
        label.grid(row=grid_row, column=0)
        self.in_folder2 = Tk.StringVar()
        self.in_files2 = []
        self.name_pattern2 = Tk.StringVar()
        self.num_files2 = Tk.IntVar()
        self.in_folder2_cb = lambda: self.on_entry_in_folder(self.in_folder2, self.in_files2, self.name_pattern2, self.num_files2)
        in_folder2_entry = FolderEntry(iframe, textvariable=self.in_folder2, width=folder_entry_width,
                                on_entry_cb=self.in_folder2_cb)
        in_folder2_entry.grid(row=grid_row, column=1, pady=5)
        in_folder2_entry.bind('<Leave>', lambda *args:self.in_folder2_cb())

        label = Tk.Label(iframe, textvariable=self.name_pattern2)
        label.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        num_files_label = Tk.Label(iframe, textvariable=self.num_files2, width=num_files_label_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3)

        grid_row += 1
        label = Tk.Label(iframe, text="Subtracted Data")
        label.grid(row=grid_row, column=0)
        self.out_folder = Tk.StringVar()
        out_folder_entry = FolderEntry(iframe, textvariable=self.out_folder, width=folder_entry_width,
                                on_entry_cb=self.on_entry_out_folder)
        out_folder_entry.grid(row=grid_row, column=1)
        out_folder_entry.bind('<Leave>', lambda *args: self.on_entry_out_folder())

        self.name_pattern3 = Tk.StringVar()
        entry = Tk.Entry(iframe, textvariable=self.name_pattern3, width=20)
        entry.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        self.num_files3 = Tk.IntVar()
        num_files_label = Tk.Label(iframe, textvariable=self.num_files3, width=num_files_label_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3)

        grid_row += 1
        space = Tk.Label(iframe)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Progress")
        label.grid(row=grid_row, column=0)
        self.mpb = ttk.Progressbar(iframe,orient ="horizontal", length=600, mode="determinate")
        self.mpb.grid(row=grid_row, column=1, columnspan=3, padx=5)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        self.subtract_btn = Tk.Button(box, text="Subtract", width=10, command=self.subtract, state=Tk.DISABLED)
        self.subtract_btn.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        self._show()

    def on_entry_in_folder(self, in_folder, in_files, name_pattern, num_files):
        files = get_xray_files(in_folder.get())
        in_files.clear()
        in_files += files
        if len(files) > 0:
            dir_, file = os.path.split(files[0])
            name_pattern.set(file)
        num_files.set( len(files) )

    def on_entry_out_folder(self):
        _, in_folder1 = os.path.split(self.in_folder1.get())
        name_pattern1 = self.name_pattern1.get()
        _, in_folder2 = os.path.split(self.in_folder2.get())
        name_pattern2 = self.name_pattern2.get()
        # print((in_folder1, name_pattern1), (in_folder2, name_pattern2))
        _, out_folder = os.path.split(self.out_folder.get())

        if is_empty_val(in_folder1) or is_empty_val(in_folder2) or is_empty_val(out_folder):
            return

        ret_pattern = self.generate_pattern(in_folder1, name_pattern1, in_folder2, name_pattern2, out_folder)
        self.name_pattern3.set(ret_pattern)
        self.subtract_btn.config(state=Tk.NORMAL)

    def subtract(self):
        num_files1 = len(self.in_files1)
        num_files2 = len(self.in_files2)

        if num_files1 != num_files2:
            MessageBox.showerror("", "Numbers of files are different: %d != %d" % (num_files1, num_files2))
            return

        name_pattern = self.name_pattern3.get().replace('_00000', '_%05d')
        print('name_pattern=', name_pattern)

        self.busy = True
        process_option = 0
        subtract_impl(self.in_files1, self.in_files2, process_option, self.out_folder.get(), name_pattern, dialog=self)
        self.busy = False

    def is_busy(self):
        return self.busy

    def generate_pattern(self, in_folder1, name_pattern1, in_folder2, name_pattern2, out_folder):
        debug = False
        match = SequenceMatcher(None, in_folder1, in_folder2).find_longest_match(0, len(in_folder1), 0, len(in_folder2))
        if debug:
            print(match)
            print(in_folder1[match.a: match.a + match.size])

        match = SequenceMatcher(None, name_pattern1, name_pattern2).find_longest_match(0, len(name_pattern1), 0, len(name_pattern2))
        common_pattern = name_pattern1[match.a: match.a + match.size]
        if debug:
            print(match)
            print(common_pattern)

        return "Diff" + common_pattern
