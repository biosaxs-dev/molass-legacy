# coding: utf-8
"""
    ScatteringCurvePlotter.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
import os
from bisect import bisect_right
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color, NavigationToolbar
from SerialDataUtils import serial_np_loadtxt

DEFAULT_CENTER  = 0.15
DEFAULT_WIDTH   = 0.02

title_texts = ["Intensity", "Error", "Error / Intensity"]

class ScatteringCurvePlotter(Dialog):
    def __init__(self, parent):
        self.log_scale = 1
        self.show_flag = 0
        self.num_files = 0
        self.lay_over_scaled = False
        self.drawn_data_list = []
        Dialog.__init__(self, parent, "Scattering Curve Plotter", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tbframe = Tk.Frame(body_frame)
        tbframe.pack(anchor=Tk.W)
        bframe = Tk.Frame(body_frame)
        bframe.pack(anchor=Tk.E)


        self.toggle_scale_btn = Tk.Button(bframe, command=self.toggle_scale)
        self.toggle_scale_btn.grid(row=0, column=0, padx=10)
        self.toggle_scale_btn.grid_forget()

        self.scale_btn_frame = Tk.Frame(bframe)
        self.scale_btn_frame.grid(row=0, column=1, padx=10)
        self.scale_btn_frame.grid_forget()
        scale_btn = Tk.Button(self.scale_btn_frame, text="Lay Over Scale", command=self.lay_over_scale)
        scale_btn.grid(row=0, column=0)
        scale_center_label = Tk.Label(self.scale_btn_frame, text=" adjusting at center")
        scale_center_label.grid(row=0, column=1, padx=5)
        self.scale_center = Tk.DoubleVar()
        self.scale_center.set(DEFAULT_CENTER)
        scale_center_entry = Tk.Entry(self.scale_btn_frame, textvariable=self.scale_center, width=6, justify=Tk.CENTER)
        scale_center_entry.grid(row=0, column=2)
        scale_width_label = Tk.Label(self.scale_btn_frame, text=" with width")
        scale_width_label.grid(row=0, column=3, padx=5)
        self.scale_width = Tk.DoubleVar()
        self.scale_width.set(DEFAULT_WIDTH)
        scale_width_entry = Tk.Entry(self.scale_btn_frame, textvariable=self.scale_width, width=6, justify=Tk.CENTER)
        scale_width_entry.grid(row=0, column=4)
        self.scale_center.trace('w', self.params_tracer)
        self.scale_width.trace('w', self.params_tracer)

        self.clear_btn = Tk.Button(bframe, text="Clear", command=self.clear)
        self.clear_btn.grid(row=0, column=2, padx=10)
        self.clear_btn.grid_forget()

        self.fig, self.axes = plt.subplots( nrows=1, ncols=3, figsize=(21, 7) )
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.clear()

        self.fig.tight_layout()
        # self.toolbar = NavigationToolbar( self.mpl_canvas, tbframe, show_mode=False )
        self.toolbar = NavigationToolbar( self.mpl_canvas, tbframe )
        self.toolbar.update()

        self.add_dnd_bind()

    def show_guide_text(self):

        self.guide_texts = []
        for ax in self.axes:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            t = ax.text(tx, ty, "Drag and drop files here to plot", alpha=0.3, fontsize=30, ha='center')
            self.guide_texts.append(t)

    def params_tracer(self, *args):
        self.lay_over_scaled = False

    def add_dnd_bind(self):
        self.mpl_canvas_widget.register_drop_target("*")

        def dnd_handler(event):
            self.on_entry(event)

        self.mpl_canvas_widget.bind("<<Drop>>", dnd_handler)

    def show(self):
        self._show()

    def on_entry(self, event):
        files = event.data.split(' ')
        # print('on_entry:', files)

        self.draw(files)

    def draw(self, files=[]):
        if self.guide_texts is not None:
            for t in self.guide_texts:
                t.remove()
            self.guide_texts = None

        self.update_scale()
        scale_center = self.scale_center.get()
        scale_width = self.scale_width.get()

        for file in files:
            self.lay_over_scaled = False
            data, _ = serial_np_loadtxt(file)
            self.drawn_data_list.append([file, data])
            x = data[:,0]
            for k, ax in enumerate(self.axes):
                if k < 2:
                    y = data[:,k+1]
                else:
                    y = data[:,2]/data[:,1]
                ax.plot(x, y, label=file)
            self.num_files += 1
            if self.num_files == 2:
                possible_center = x[-1] - scale_width/2
                if possible_center < scale_center:
                    self.scale_center.set(possible_center)
                self.scale_btn_frame.grid(row=0, column=1, padx=10)

        for ax in self.axes:
            ax.legend()
        self.mpl_canvas.draw()

    def update_scale(self):
        if self.log_scale:
            scale = 'log'
            btn_text = 'To Linear'
        else:
            scale = 'linear'
            btn_text = 'To Log'

        for ax in self.axes:
            ax.set_yscale(scale)
        self.toggle_scale_btn.config(text=btn_text)

        self.toggle_scale_btn.grid(row=0, column=0, padx=10)
        self.clear_btn.grid(row=0, column=2, padx=10)

    def toggle_scale(self):
        self.log_scale = 1 - self.log_scale
        self.update_scale()
        self.mpl_canvas.draw()

    def lay_over_scale(self):
        if not self.lay_over_scaled and len(self.drawn_data_list) > 0:
            self.lay_over_scale_drawn_data()

        ax = self.axes[0]
        ax.cla()

        self.update_scale()

        for file, data in self.drawn_data_list:
            x = data[:,0]
            y = data[:,1]
            ax.plot(x, y, label=file)

        if self.lay_over_scaled:
            f, t = self.x[self.index[[0, -1]]]
            ymin, ymax = ax.get_ylim()
            p = Rectangle(
                    (f, ymin),  # (x,y)
                    t - f,   # width
                    ymax - ymin,    # height
                    facecolor   = 'red',
                    alpha       = 0.1,
                )
            ax.add_patch(p)

        ax.legend()
        self.mpl_canvas.draw()

    def lay_over_scale_drawn_data(self):
        _, data = self.drawn_data_list[0]
        x = data[:,0]
        q = self.scale_center.get()
        w = self.scale_width.get()
        i = np.where(np.logical_and(x > q-w/2, x < q+w/2))[0]
        y = data[:,1]
        target = np.average(y[i])

        for _, data in self.drawn_data_list[1:]:
            y = data[:,1]
            y *= target/np.average(y[i])

        self.lay_over_scaled = True
        self.x = x
        self.index = i

    def clear(self):
        for k, ax in enumerate(self.axes):
            ax.cla()
            ax.set_title(title_texts[k], fontsize=20)
        self.drawn_data_list = []
        self.show_guide_text()
        self.mpl_canvas.draw()
