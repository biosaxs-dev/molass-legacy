# coding: utf-8
"""
    DecompResetDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import re
import os
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.TkCustomWidgets import FileEntry
from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText
from DecompEditorFrame import DECOMP_EDITOR_LOG_HEADER, DECOMP_EDITOR_LOG_TRAILER
from molass_legacy._MOLASS.SerialSettings import get_setting

in_folder_re    = re.compile(r'^.+started loading (\S+)')
log_line_re     = re.compile(r'^.+DecompEditorDialog,')
model_name_re   = re.compile(r'^.+model_name=(\w+)')

params_re   = re.compile(r'\[\d+\] h=(\S+), mu=(\S+), sigma=(\S+), tau=(\S+), a=(\S+); with tau constraints \((\S+), (\S+)\) and uv/xray h ratio (\S+)')
range_re    = re.compile(r'\[\d+\] (\[.*\]) with element curves (\[.*\])')

def extract_params_from_log_lines(lines):
    param_recs = []
    range_recs = []
    for line in lines:
        m = params_re.search(line)
        if m:
            param_recs.append([ float(m.group(i)) for i in range(1,9) ])
            continue
        m = range_re.search(line)
        if m:
            range_recs.append([eval(m.group(i)) for i in range(1,3) ])
            continue
    return param_recs, range_recs

class DecompResetDialog( Dialog ):
    def __init__(self, parent):
        self.logger  = logging.getLogger( __name__ )
        self.parent = parent
        self.in_folder = None
        self.canceled = False
        self.applied = False
        Dialog.__init__( self, parent, "DecompResetDialog", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        filepath_frame = Tk.Frame(body_frame)
        filepath_frame.pack()

        file_label = Tk.Label(filepath_frame, text="Log File Path: ")
        file_label.pack(side=Tk.LEFT)
        self.logfile_path = Tk.StringVar()
        file_entry = FileEntry(filepath_frame, textvariable=self.logfile_path, width=60, on_entry_cb=self.on_entry )
        file_entry.pack(side=Tk.LEFT)

        current_log_btn = Tk.Button(filepath_frame, text="Current Log File", command=self.set_current_logfile)
        current_log_btn.pack(side=Tk.LEFT, padx=10)

        space = Tk.Frame(body_frame)
        space.pack(pady=5)

        text_label = Tk.Label(body_frame, text="The following info has been retrieved from the log file.")
        text_label.pack()
        self.ro_text = ReadOnlyText(body_frame, width=140, height=16)
        self.ro_text.pack(padx=10)

        space = Tk.Frame(body_frame)
        space.pack(pady=5)

        guide_label = Tk.Label(body_frame, text='If you would like to reset with this info, please press "Reset", otherwise "Cancel".')
        guide_label.pack()

        soon_label = Tk.Label(body_frame, text='"Reset", disabled in this version, will soon be available.', fg='orange')
        soon_label.pack()

    def buttonbox(self):

        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Reset", width=10, command=self.reset, default=Tk.ACTIVE, state=Tk.DISABLED)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def on_entry(self):
        file_path = self.logfile_path.get()
        print('on_entry', file_path)
        reading_target_log = False
        model_name = None
        fh = open(file_path)
        for line in fh.readlines():
            if line.find('started loading') > 0:
                m = in_folder_re.search(line)
                self.in_folder = m.group(1)

            if line.find(DECOMP_EDITOR_LOG_HEADER) > 0:
                reading_target_log = True
                model_name = None
                info_lines = []
                continue
            if line.find(DECOMP_EDITOR_LOG_TRAILER) > 0:
                reading_target_log = False

            if reading_target_log:
                cut_line = re.sub(log_line_re, '', line)
                m = model_name_re.search(cut_line)
                if m:
                    model_name = m.group(1)
                info_lines.append(cut_line)
        fh.close()

        self.ro_text.delete('1.0', Tk.END)

        in_folder_info = 'unretrieved' if self.in_folder is None else self.in_folder
        self.ro_text.insert(Tk.END, 'in_folder=' + in_folder_info + '\n\n')
        try:
            for cut_line in info_lines:
                self.ro_text.insert(Tk.END, cut_line)
        except:
            pass

    def set_current_logfile(self):
        import glob
        analysis_folder = get_setting('analysis_folder')
        try:
            logfiles = glob.glob(analysis_folder + '/*.log')
            self.logfile_path.set(logfiles[0].replace('\\', '/'))
            self.on_entry()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            import OurMessageBox as MessageBox
            MessageBox.showerror("Error", "Couldn't find the current log file.\n---\n" + str(etb), parent=self)

    def reset(self):
        pass
