# coding: utf-8
"""
    AverageSubtractorDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
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
from DataSubtractor import subtract_impl
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
from SerialDataUtils import get_mtd_filename

class AverageSubtractorDialog(Dialog):
    def __init__(self, parent, use_mtd=True):
        self.logger = logging.getLogger(__name__)
        self.busy = False
        self.background_info = None
        self.in_folder_prev = None
        self.use_mtd = use_mtd
        Dialog.__init__(self, parent, "Average Subtractor", visible=False)

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
        self.in_folder1_entry = FolderEntry(iframe, textvariable=self.in_folder1, width=folder_entry_width,
                                on_entry_cb=self.in_folder1_cb)
        self.in_folder1_entry.grid(row=grid_row, column=1)
        self.in_folder1_entry.bind('<Leave>', lambda *args:self.in_folder1_cb())

        label = Tk.Label(iframe, textvariable=self.name_pattern1)
        label.grid(row=grid_row, column=2, padx=10, sticky=Tk.W)

        num_files_label = Tk.Label(iframe, textvariable=self.num_files1, width=num_files_label_width, justify=Tk.RIGHT)
        num_files_label.grid(row=grid_row, column=3)

        grid_row += 1
        self.maker_btn_frame = BlinkingFrame(iframe)
        self.maker_btn_frame.grid(row=grid_row, column=0, pady=5)

        self.maker_btn = Tk.Button(self.maker_btn_frame, text="Backgroud Maker", command=self.show_backgroud_maker_dialog, state=Tk.DISABLED)
        self.maker_btn.pack()

        self.maker_btn_frame.objects = [self.maker_btn]

        self.in_folder2 = Tk.StringVar()
        self.in_files2 = []
        self.name_pattern2 = Tk.StringVar()
        self.num_files2 = Tk.IntVar()
        in_folder2_entry = FolderEntry(iframe, textvariable=self.in_folder2, width=folder_entry_width,
                                on_entry_cb=None)
        in_folder2_entry.grid(row=grid_row, column=1, pady=5)

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
            self.maker_btn.config(state=Tk.NORMAL)

        filename = get_mtd_filename( in_folder_ )
        if filename is None:
            if self.use_mtd:
                self.in_folder1_entry.config(fg='red')
                self.update()
                MessageBox.showerror("Error", "No MTD file is found.", parent=self)
                return

        self.in_folder1_entry.config(fg='black')
        self.mtd_file_path = filename

        self.in_folder2.set(in_folder_ + '/bg')
        if len(files) > 0:
            name_pattern_re = re.compile(r'(\d+)(\.dat)$')
            bg_file = re.sub(name_pattern_re, lambda m: 'bg' + m.group(2), file)
            self.name_pattern2.set(bg_file)

        self.out_folder.set(in_folder_ + '/sub')
        if len(files) > 0:
            self.on_entry_out_folder()

        self.maker_btn_frame.start()

    def on_entry_out_folder(self):
        _, in_folder1 = os.path.split(self.in_folder1.get())
        name_pattern1 = self.name_pattern1.get()
        _, in_folder2 = os.path.split(self.in_folder2.get())
        name_pattern2 = self.name_pattern2.get()
        # print((in_folder1, name_pattern1), (in_folder2, name_pattern2))
        _, out_folder = os.path.split(self.out_folder.get())

        if len(name_pattern1) > 0:
            ret_pattern = self.generate_pattern(in_folder1, name_pattern1, in_folder2, name_pattern2, out_folder)
            self.name_pattern3.set(ret_pattern)

    def subtract(self):

        name_pattern = self.name_pattern3.get().replace('_00000', '_%05d')
        print('name_pattern=', name_pattern)

        self.busy = True
        background_data = self.background_info[0]
        process_option = 1

        out_folder = self.out_folder.get()
        if not os.path.exists(out_folder):
            mkdirs_with_retry(out_folder)

        subtract_impl(self.in_files1, background_data, process_option, out_folder, name_pattern, dialog=self)
        self.busy = False

    def is_busy(self):
        return self.busy

    def generate_pattern(self, in_folder1, name_pattern1, in_folder2, name_pattern2, out_folder):
        name_pattern_re = re.compile(r'(\d+)(\.dat)$')
        sub_filename = re.sub(name_pattern_re, lambda m: m.group(1) + '_sub' + m.group(2), name_pattern1)

        return sub_filename

    def show_backgroud_maker_dialog(self):
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        from BackgroundMaker import BackgroundMakerDialog
        from molass_legacy._MOLASS.SerialSettings import set_setting, do_microfluidic_temporary_settings
        from SerialTestUtils import prepare_serialdata_env

        self.maker_btn_frame.stop()
        in_folder = self.in_folder1.get()

        if self.use_mtd:
            filename = self.mtd_file_path
            set_setting( 'use_xray_conc', 0 )
            set_setting( 'use_mtd_conc', 1 )
            set_setting( 'mtd_file_path', filename )
            do_microfluidic_temporary_settings()
            mtd_conc = True
        else:
            mtd_conc = False

        sd = prepare_serialdata_env( in_folder, root=self, mtd_conc=mtd_conc, logger=self.logger )
        pre_recog = PreliminaryRecognition(sd)

        dialog = BackgroundMakerDialog(self, pre_recog)
        dialog.show()
        if dialog.applied:
            self.background_info = dialog.get_the_background_info()
            self.save_background_data()
            self.num_files2.set(1)
            self.subtract_btn.config(state=Tk.NORMAL)

    def save_background_data(self):
        from molass_legacy.KekLib.NumpyUtils import np_savetxt_with_comments
        folder = self.in_folder2.get()
        if not os.path.exists(folder):
            mkdirs_with_retry(folder)

        filename = self.name_pattern2.get()
        filepath = os.path.join(folder, filename)
        background_data = self.background_info[0]
        elution_slice = self.background_info[1]
        comments = [
            '# averaged from %d to %d\n' % (elution_slice.start, elution_slice.stop-1)
            ]
        np_savetxt_with_comments(filepath, background_data, comments)
