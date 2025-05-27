# coding: utf-8
"""

    Rank.RankView.py

    Copyright (c) 2022, SAXS Team, KEK-PF

"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from MenuButton import MenuButton
from DataUtils import get_in_folder
from .SrrTutor import rankview_plot

class RankView(Dialog):
    def __init__(self, parent, sd, data_type, **kwargs):
        self.sd = sd
        self.data_type = data_type
        self.kwargs = kwargs
        Dialog.__init__(self, parent, "Rank View", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        sd = self.sd
        data_type = self.data_type
        kwargs = self.kwargs
        xr_type = data_type.lower().find("x") >= 0
        if xr_type:
            M, E, vec, ecurve = sd.get_xr_data_separate_ly()
        else:
            M, _, vec, ecurve = sd.get_uv_data_separate_ly()
        self.M = M
        self.vec = vec

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = fig = plt.figure(figsize=(20,9))
        in_folder = get_in_folder()
        kwargs["return_axes"] = True
        self.axes_info = rankview_plot(fig, in_folder, sd, M, vec, data_type, xr_type, **kwargs)

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()

    def buttonbox(self):
        frame = Tk.Frame(self)
        frame.pack(fill=Tk.X)
        tframe = Tk.Frame(frame)
        tframe.pack(side=Tk.LEFT, padx=20)
        bframe = Tk.Frame(frame)
        bframe.pack(side=Tk.RIGHT, padx=20)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        w = Tk.Button(bframe, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=100, pady=5)

        w = Tk.Button(bframe, text="Synchronize", width=10, command=self.synchronize)
        w.pack(side=Tk.LEFT, padx=100, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def synchronize(self):
        print("synchronize")
        ax0, axes = self.axes_info[0:2]
        kwargs = dict(azim=ax0.azim, elev=ax0.elev)
        for ax_row in axes:
            for ax in ax_row:
                ax.view_init(**kwargs)
        self.mpl_canvas.draw()

class RankViewMenu(Tk.Frame):
    def __init__(self, parent, dialog):
        Tk.Frame.__init__(self, parent)
        self.parent = parent
        self.dialog = dialog
        self.view_kw = None
        self.trimming = True
        self.menu = MenuButton(self, "Rank View", [
                            ("UV Rank View", lambda: self.show_rankview("UV")),
                            ("UV Rank View (corrected)", lambda: self.show_rankview("UV", correct=True)),
                            ("Xray Rank View", lambda: self.show_rankview("Xray")),
                            ("Xray Rank View (corrected)", lambda: self.show_rankview("Xray", correct=True)),
                            ("Change View Parameters", self.show_view_params_dialog),
                            ])
        self.menu.pack()
        self.menu.entryconfig(1, state=Tk.DISABLED)

    def config(self, **kwargs):
        self.menu.config(**kwargs)

    def show_rankview(self, data_type, correct=False):
        sd = self.dialog.serial_data
        pre_recog = self.dialog.pre_recog
        if correct:
            from molass_legacy.Baseline.BaselineUtils import get_corrected_sd_impl
            sd = get_corrected_sd_impl(sd, sd, pre_recog)
            data_label = "corrected "
        else:
            data_label = "no correction "
        dialog = RankView(self.dialog, sd, data_type, pre_recog=pre_recog, trim=self.trimming, data_label=data_label, view_kw=self.view_kw)
        dialog.show()

    def show_view_params_dialog(self):
        dialog = RankViewParamsDialog(self.dialog, view_kw=self.view_kw, trim=self.trimming)
        dialog.show()
        if dialog.applied:
            self.trimming, self.view_kw = dialog.get_parameters()

class RankViewParamsDialog(Dialog):
    def __init__(self, parent, view_kw=None, trim=True):
        self.view_kw = view_kw
        self.trim = trim
        self.applied = False
        Dialog.__init__(self, parent, "Rank View Parameters", visible=False, location='center right')

    def show(self):
        self._show()

    def body(self, body_frame):

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        for j in range(3):
            iframe.columnconfigure(j, weight=1)

        grid_row = 0
        space = Tk.Label(iframe, text="    ")
        space.grid(row=grid_row, column=0)

        label = Tk.Label(iframe, text="■  Trimming Option")
        label.grid(row=grid_row, column=0, columnspan=3, pady=5, sticky=Tk.W)

        grid_row += 1
        self.trimming = Tk.IntVar()
        self.trimming.set(int(self.trim))
        cb = Tk.Checkbutton(iframe, text="apply automatic trimming", variable=self.trimming)
        cb.grid(row=grid_row, column=1, columnspan=2)

        grid_row += 1
        space = Tk.Label(iframe, text="    ")
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="■  3D View Parameters")
        label.grid(row=grid_row, column=0, columnspan=3, pady=5, sticky=Tk.W)

        azim, elev = (-60, 30) if self.view_kw is None else (self.view_kw["azim"], self.view_kw["elev"])

        grid_row += 1
        self.azim = Tk.IntVar()
        self.azim.set(azim)
        label = Tk.Label(iframe, text="azimuth: ")
        label.grid(row=grid_row, column=1, sticky=Tk.E)
        entry = Tk.Entry(iframe, textvariable=self.azim, width=6, justify=Tk.CENTER)
        entry.grid(row=grid_row, column=2)

        grid_row += 1
        self.elev = Tk.IntVar()
        self.elev.set(elev)
        label = Tk.Label(iframe, text="elevation: ")
        label.grid(row=grid_row, column=1, sticky=Tk.E)
        entry = Tk.Entry(iframe, textvariable=self.elev, width=6, justify=Tk.CENTER)
        entry.grid(row=grid_row, column=2)

    def apply(self):
        self.applied = True

    def get_parameters(self):
        return self.trimming.get()==1, dict(azim=self.azim.get(), elev=self.elev.get())
