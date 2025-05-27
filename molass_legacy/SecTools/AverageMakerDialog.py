# coding: utf-8
"""
    SecTools.AverageMakerDialog.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
import re
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, is_empty_val
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable, BlinkingFrame
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
try:
    import molass_legacy.KekLib.CustomMessageBox         as MessageBox
except:
    import OurMessageBox            as MessageBox
from SerialDataUtils import get_xray_files
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry, num_files_in
from .DataAverager import average_impl

class AverageMakerDialog(Dialog):
    def __init__(self, parent):
        self.logger = logging.getLogger(__name__)
        self.busy = False
        self.in_folder_prev = None
        self.out_pattern_manually_changed = False
        Dialog.__init__(self, parent, "Average Maker", visible=False)

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
        label.grid(row=grid_row, column=0, sticky=Tk.E, padx=10)

        self.in_folder1 = Tk.StringVar()
        self.in_files1 = []
        self.name_pattern1 = Tk.StringVar()
        self.num_files1 = Tk.IntVar()
        self.in_folder1_cb = lambda: self.on_entry_in_folder(self.in_folder1, self.in_files1, self.name_pattern1, self.num_files1)
        self.in_folder1_entry = FolderEntry(iframe, textvariable=self.in_folder1, width=folder_entry_width,
                                on_entry_cb=self.in_folder1_cb)
        self.in_folder1_entry.grid(row=grid_row, column=1)
        self.in_folder1_entry.bind('<Leave>', lambda *args:self.in_folder1_cb())

        label = Tk.Label(iframe, textvariable=self.name_pattern1, justify=Tk.LEFT)
        label.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        num_files_label = Tk.Label(iframe, textvariable=self.num_files1, width=num_files_label_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3)

        grid_row += 1
        num_frame = Tk.Frame(iframe)
        num_frame.grid(row=grid_row, column=0, columnspan=3, sticky=Tk.W)

        label = Tk.Label(num_frame, text="Number of Files to Average: ")
        label.grid(row=0, column=0, sticky=Tk.W, padx=10)
        self.num_files_to_average = Tk.IntVar()
        self.num_files_to_average.set(11)
        sbox  = Tk.Spinbox(num_frame, textvariable=self.num_files_to_average,
                          from_=1, to=19, increment=1,
                          justify=Tk.CENTER, width=6)
        sbox.grid(row=0, column=1, padx=20, pady=20)

        grid_row += 1
        label = Tk.Label(iframe, text="Averaged Data")
        label.grid(row=grid_row, column=0, sticky=Tk.E, padx=10)
        self.out_folder = Tk.StringVar()
        out_folder_entry = FolderEntry(iframe, textvariable=self.out_folder, width=folder_entry_width,
                                on_entry_cb=self.on_entry_out_folder)
        out_folder_entry.grid(row=grid_row, column=1)
        out_folder_entry.bind('<Leave>', lambda *args: self.on_entry_out_folder(on_leave=True))

        self.name_pattern3 = Tk.StringVar()
        entry = Tk.Entry(iframe, textvariable=self.name_pattern3, width=30)
        entry.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)
        self.name_pattern3_tracing = True
        self.name_pattern3.trace("w", self.name_pattern3_tracer)

        self.num_files3 = Tk.IntVar()
        num_files_label = Tk.Label(iframe, textvariable=self.num_files3, width=num_files_label_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3, sticky=Tk.W)

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

        self.average_btn = Tk.Button(box, text="Make Average", width=15, command=self.make_average, state=Tk.DISABLED)
        self.average_btn.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        self._show()

    def on_entry_in_folder(self, in_folder, in_files, name_pattern, num_files):
        in_folder_ = in_folder.get()
        if is_empty_val(in_folder_) or in_folder_ == self.in_folder_prev:
            return

        self.in_folder_prev = in_folder_
        files = get_xray_files(in_folder_)
        in_files.clear()
        in_files += files
        if len(files) > 0:
            dir_, file = os.path.split(files[0])
            name_pattern.set(file)
        num_files.set( len(files) )
        if self.num_files1.get() > 0:
            self.average_btn.config(state=Tk.NORMAL)

        self.in_folder1_entry.config(fg='black')

        self.out_folder.set(in_folder_ + '/averaged')
        if len(files) > 0:
            self.on_entry_out_folder()

    def on_entry_out_folder(self, on_leave=False):
        _, in_folder1 = os.path.split(self.in_folder1.get())
        name_pattern1 = self.name_pattern1.get()
        _, out_folder = os.path.split(self.out_folder.get())

        if len(name_pattern1) > 0 and (not on_leave or self.need_out_pattern_update()):
            ret_pattern = self.generate_pattern(in_folder1, name_pattern1, out_folder)
            self.name_pattern3_tracing = False
            self.name_pattern3.set(ret_pattern)
            self.out_pattern_manually_changed = False

    def need_out_pattern_update(self):
        name_pattern3 = self.name_pattern3.get()
        # print("need_out_pattern_update: name_pattern3=", name_pattern3, "changed=", self.out_pattern_manually_changed)
        if is_empty_val(name_pattern3):
            return True
        return not self.out_pattern_manually_changed

    def name_pattern3_tracer(self, *args):
        if self.name_pattern3_tracing:
            self.out_pattern_manually_changed = True
        else:
            self.name_pattern3_tracing = True

    def make_average(self):
        out_folder = self.out_folder.get()
        if os.path.exists(out_folder):
            if num_files_in(out_folder) > 0:
                reply = MessageBox.askokcancel(
                    "Clear Folder Confirmation", 
                    "The output folder is not empty.\n"
                    "This operation will clear the output folder.\n"
                    "Are you sure to proceed?",
                    parent=self
                    )
                if not reply:
                    return

        name_pattern = self.name_pattern3.get().replace('_00000', '_%05d')
        # print('name_pattern=', name_pattern)

        self.busy = True
        clear_dirs_with_retry([out_folder])

        num_average = self.num_files_to_average.get()
        average_impl(num_average, self.in_files1, out_folder, name_pattern, dialog=self)
        self.busy = False

    def is_busy(self):
        return self.busy

    def generate_pattern(self, in_folder1, name_pattern1, out_folder):
        name_pattern_re = re.compile(r'(_\d+.*)(\.dat)$')
        sub_filename = re.sub(name_pattern_re, lambda m: m.group(1) + '_avg' + m.group(2), name_pattern1)

        return sub_filename
