# coding: utf-8
"""
    SubtractorFrame.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, is_empty_val
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.KekLib.TkUtils import is_low_resolution
try:
    import molass_legacy.KekLib.CustomMessageBox         as MessageBox
except:
    import OurMessageBox            as MessageBox
from XrayData import XrayData
from DataSubtractor import subtract_impl

TITLE_TEXTS = ["Sample Data", "Buffer Data", "Subtracted Data"]

class SubtractorFrame(Tk.Frame):
    def __init__(self, parent):
        self.busy = False
        self.data_list = [None]*3
        self.zlim_3d = None
        self.ylim_2d = None
        Tk.Frame.__init__(self, parent)
        self.body(self)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        self.current = 0
        self.frames = []

        frame_3d = Tk.Frame(body_frame)
        frame_3d.pack()
        self.frames.append(frame_3d)
        frame_2d = Tk.Frame(body_frame)
        frame_2d.pack()
        frame_2d.pack_forget()
        self.frames.append(frame_2d)

        cframe_3d = Tk.Frame(frame_3d)
        cframe_3d.pack()
        tbframe_3d = Tk.Frame(frame_3d)
        tbframe_3d.pack(fill=Tk.X, expand=1)

        cframe_2d = Tk.Frame(frame_2d)
        cframe_2d.pack()
        tbframe_2d = Tk.Frame(frame_2d)
        tbframe_2d.pack(fill=Tk.X, expand=1)

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
        self.in_folder1_cb = lambda: self.on_entry_in_folder(self.in_folder1, self.in_files1, self.name_pattern1, self.num_files1, 0)
        in_folder1_entry = FolderEntry(iframe, textvariable=self.in_folder1, width=folder_entry_width,
                                on_entry_cb=self.in_folder1_cb)
        in_folder1_entry.grid(row=grid_row, column=1)
        # in_folder1_entry.bind('<Leave>', lambda *args:self.in_folder1_cb())

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
        self.in_folder2_cb = lambda: self.on_entry_in_folder(self.in_folder2, self.in_files2, self.name_pattern2, self.num_files2, 1)
        in_folder2_entry = FolderEntry(iframe, textvariable=self.in_folder2, width=folder_entry_width,
                                on_entry_cb=self.in_folder2_cb)
        in_folder2_entry.grid(row=grid_row, column=1, pady=5)
        # in_folder2_entry.bind('<Leave>', lambda *args:self.in_folder2_cb())

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
        self.subtract_btn = Tk.Button(iframe, text="Subtract", width=10, command=self.subtract, state=Tk.DISABLED)
        self.subtract_btn.grid(row=grid_row, column=2, columnspan=2, sticky=Tk.E)

        grid_row += 1
        space = Tk.Label(iframe)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Progress")
        label.grid(row=grid_row, column=0)
        self.mpb = ttk.Progressbar(iframe,orient ="horizontal", length=600, mode="determinate")
        self.mpb.grid(row=grid_row, column=1, columnspan=3, padx=5)

        self.build_canvas_3d(cframe_3d, tbframe_3d)
        self.build_canvas_2d(cframe_2d, tbframe_2d)

    def build_canvas_3d(self, cframe, tbframe):
        figsize = (15,5) if is_low_resolution() else (21,7)
        fig, axes_3d = plt.subplots(nrows=1, ncols=3, figsize=figsize, subplot_kw=dict(projection='3d'))
        self.fig = fig
        self.axes_3d = axes_3d
        self.mpl_canvas_3d = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas_3d_widget = self.mpl_canvas_3d.get_tk_widget()
        self.mpl_canvas_3d_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas_3d.mpl_connect( 'button_press_event', self.on_button_press )
        self.toolbar = NavigationToolbar(self.mpl_canvas_3d, tbframe)
        self.toolbar.update()
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        self.mpl_canvas_3d.draw()

    def build_canvas_2d(self, cframe, tbframe):
        figsize = (15,5) if is_low_resolution() else (21,7)
        fig, axes_3d = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.fig_2d = fig
        self.axes_2d = axes_3d
        self.mpl_canvas_2d = FigureCanvasTkAgg( self.fig_2d, cframe )
        self.mpl_canvas_2d_widget = self.mpl_canvas_2d.get_tk_widget()
        self.mpl_canvas_2d_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas_2d.mpl_connect( 'button_press_event', self.on_button_press )
        self.toolbar = NavigationToolbar(self.mpl_canvas_2d, tbframe)
        self.toolbar.update()
        fig.tight_layout()
        fig.subplots_adjust(top=0.92)
        self.mpl_canvas_2d.draw()

    def on_button_press(self, event):
        if event.dblclick:
            self.toggle_2d3d()

    def toggle_2d3d(self):
        previous = self.current
        self.current = 1 - self.current
        print('toggle_2d3d', previous, self.current)
        self.frames[previous].pack_forget()
        self.frames[self.current].pack()

    def on_entry_in_folder(self, in_folder, in_files, name_pattern, num_files, k):
        self.update()

        xray_data = XrayData(in_folder.get())
        self.data_list[k] = xray_data

        files = xray_data.files
        in_files.clear()
        in_files += files
        if len(files) > 0:
            dir_, file = os.path.split(files[0])
            name_pattern.set(file)
        num_files.set( len(files) )
        self.update()

        ax = self.axes_3d[k]
        ax.set_title(TITLE_TEXTS[k], fontsize=16)
        ec_color = 'orange' if k == 0 else 'gray'
        xray_data.plot(ax=ax, title=None, ec_color=ec_color)

        if self.zlim_3d is None:
            self.zlim_3d = ax.get_zlim()
        else:
            ax.set_zlim(self.zlim_3d)

        self.mpl_canvas_3d.draw()

        ax = self.axes_2d[k]
        ax.set_title(TITLE_TEXTS[k], fontsize=16)
        if xray_data.e_curve is None:
            x = xray_data.j
            k = xray_data.e_index
            y = xray_data.data[k,:]
        else:
            x = xray_data.e_curve.x
            y = xray_data.e_curve.y
        color = 'orange' if k == 0 else 'gray'
        ax.plot(x, y, color=color)

        if self.ylim_2d is None:
            self.ylim_2d = ax.get_ylim()
        else:
            ax.set_ylim(self.ylim_2d)

        self.mpl_canvas_2d.draw()

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
        out_folder = self.out_folder.get()
        ret_data_list = subtract_impl(self.in_files1, self.in_files2, process_option, out_folder, name_pattern, dialog=self)
        array = np.array(ret_data_list)
        q = array[0,:,0]
        data = array[:,:,1].T
        error = array[:,:,2].T
        xray_data = XrayData(out_folder, data=data, q=q, error=error)

        k = 2
        ax = self.axes_3d[k]
        ax.set_title(TITLE_TEXTS[k], fontsize=16)
        xray_data.plot(ax=ax, title=None, ec_color='brown')
        ax.set_zlim(self.zlim_3d)
        self.mpl_canvas_3d.draw()

        ax = self.axes_2d[k]
        ax.set_title(TITLE_TEXTS[k], fontsize=16)

        x = xray_data.e_curve.x
        y = xray_data.e_curve.y
        ax.plot(x, y, color='brown')

        ylim = list(ax.get_ylim())  # 'tuple' object does not support item assignment
        ylim[1] = self.ylim_2d[1]
        for ax in self.axes_2d:
            ax.set_ylim(tuple(ylim))

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
