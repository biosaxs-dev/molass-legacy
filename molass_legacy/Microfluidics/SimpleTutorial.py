# coding: utf-8
"""
    SimpleTutorial.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, is_empty_val
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.KekLib.TkUtils import is_low_resolution
from MenuButton import MenuButton
from SimpleUnfolding import SimpleUnfolding

class SimpleTutorial(Dialog):
    def __init__(self, parent):
        self.busy = False
        self.applied = False
        self.data_list = [None]*3
        self.zlim_3d = None
        self.ylim = None
        Dialog.__init__(self, parent, "Simple Model Tutorial", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tbframe = Tk.Frame(body_frame)
        tbframe.pack(fill=Tk.X, expand=1)

        self.build_entryframe(iframe)
        self.build_canvas(cframe, tbframe)
        self.solve()

    def build_entryframe(self, iframe):
        table_frame = Tk.Frame(iframe)
        table_frame.pack()
        button_frame = Tk.Frame(iframe)
        button_frame.pack(fill=Tk.X)

        label = Tk.Label(table_frame, text="Given")
        label.grid(row=0, column=1)
        label = Tk.Label(table_frame, text="Solved")
        label.grid(row=0, column=2)

        self.texts = texts = ["G", "m", "af", "bf", "au", "bu"]
        values = [4, 1, 0.5, -0.01, 0.3, -0.02]

        self.vars1 = []
        self.vars2 = []

        for i, t in enumerate(texts):
            k = i+1
            label = Tk.Label(table_frame, text=t)
            label.grid(row=k, column=0, sticky=Tk.E)
            var = Tk.DoubleVar()
            var.set(values[i])
            self.vars1.append(var)
            entry = Tk.Entry(table_frame, textvariable=var, justify=Tk.RIGHT, width=8)
            entry.grid(row=k, column=1, sticky=Tk.W, padx=10)
            var = Tk.DoubleVar()
            self.vars2.append(var)
            label = Tk.Label(table_frame, textvariable=var, justify=Tk.RIGHT, width=8, bg='white')
            label.grid(row=k, column=2)

        ex_menu = MenuButton(button_frame, "Examples", [
                            ("Example1(Regular1)", self.solve_regular1),
                            ("Example1(Regular2)", self.solve_regular2),
                            ("Example2(Irregular)", self.solve_irregular),
                            ])
        ex_menu.pack(side=Tk.LEFT)

        button = Tk.Button(button_frame, text="Solve", command=self.solve)
        button.pack(side=Tk.RIGHT)

    def build_canvas(self, cframe, tbframe):
        figsize = (17,5) if is_low_resolution() else (23,7)
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        self.fig = fig
        self.axes = axes
        self.axest = [ax.twinx() for ax in axes]
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.mpl_connect( 'button_press_event', self.on_button_press )
        self.toolbar = NavigationToolbar(self.mpl_canvas, tbframe)
        self.toolbar.update()
        fig.subplots_adjust(top=0.92, bottom=0.1, left=0.05, right=0.95, wspace=0.2)

    def show(self):
        self._show(wait=False)
        self.waited_widget = Tk.Frame(self)
        self.wait_window(self.waited_widget)

    def on_button_press(self, event):
        if event.dblclick:
            self.toggle3d()

    def ok(self):
        self.apply()
        self.close()

    def exit(self):
        self.close()

    def close(self):
        self.withdraw()
        self.update_idletasks()
        self.grab_release()
        self.waited_widget.destroy()

    def apply(self):
        self.applied = True

    def make_params(self):
        params = Parameters()
        for k, var in enumerate(self.vars1):
            t = self.texts[k]
            params.add(t, var.get())
        return params

    def draw_ab_params(self, ax, axt):
        ax.cla()
        axt.cla()
        axt.set_axis_off()
        ax.set_title("Given Params(a*,b*)", fontsize=24)
        ax.set_xlabel('a')
        ax.set_ylabel('b')

        xmax = None
        ymax = None
        for t, i, j in [("$(a_F,b_F)$",2,3), ("$(a_U,b_U)$",4,5)]:
            x = self.vars1[i].get()
            y = self.vars1[j].get()
            if xmax is None or abs(x) > xmax:
                xmax = abs(x)
            if ymax is None or abs(y) > ymax:
                ymax = abs(y)
            ax.text(x, y, t)
            ax.annotate('', xy=(x, y), xytext=(0,0),
                arrowprops=dict(headwidth=5, width=0.5, color='black', shrink=0)
               )
        xmax_ = max(1, xmax)
        ymax_ = max(0.03, ymax)
        ax.set_xlim(-xmax_, xmax_)
        ax.set_ylim(-ymax_, ymax_)

    def draw_test_data(self, ax, axt):
        x = np.linspace(0, 8, 100)

        model = SimpleUnfolding()
        params = self.make_params()
        Pf = model.compute_Pf(params, x)
        yf = model.compute_yf(params, Pf, x)
        yu = model.compute_yu(params, Pf, x)
        yfyu = yf+yu
        self.draw(ax, axt, "Given Data", x, Pf, yf, yu, yfyu)
        return yfyu

    def draw(self, ax, axt, title, x, Pf, yf, yu, yfyu):
        ax.cla()
        axt.cla()
        ax.set_title(title, fontsize=24)
        ax.set_xlabel('[Denaturant]')
        ax.set_ylabel('Proportion')
        axt.set_ylabel('Intensity')
        axt.grid(False)
        ax.plot(x, Pf, label=r'$P_F=\frac{1}{1+exp(-\frac{(G-m*x)}{RT})}$')
        axt.plot(x, yf, ':', label='$y_F$')
        axt.plot(x, yu, ':', label='$y_U$')
        axt.plot(x, yfyu, label='$y_F+y_U$')
        ax.legend(loc='upper center', fontsize=16)
        axt.legend(loc='center right', fontsize=16)

    def draw_solved_data(self, ax, axt, data):
        model = SimpleUnfolding()
        x = np.linspace(0, 8, 100)
        y = data
        res = model.fit(x, y)
        params = res.params
        # print('params=', params)
        Pf = model.compute_Pf(params, x)
        yf = model.compute_yf(params, Pf, x)
        yu = model.compute_yu(params, Pf, x)
        yfyu = yf+yu
        self.draw(ax, axt, "Solved Data", x, Pf, yf, yu, yfyu)

        for k, var in enumerate(self.vars2):
            t = self.texts[k]
            f = float('%g' % params[t].value)
            var.set(f)

    def solve(self):
        self.draw_ab_params(self.axes[0], self.axest[0])
        yfyu = self.draw_test_data(self.axes[1], self.axest[1])
        self.draw_solved_data(self.axes[2], self.axest[2], yfyu)
        self.mpl_canvas.draw()

    def set_params(self, values):
        for w, v in zip(self.vars1, values):
            w.set(v)

    def solve_regular1(self):
        self.set_params([4, 1, 0.5, -0.01, 0.3, -0.02])
        self.solve()

    def solve_regular2(self):
        self.set_params([4, 1, 0.5, 0.01, 0.3, 0.02])
        self.solve()

    def solve_irregular(self):
        self.set_params([4, 1, 0.5, -0.01, 0.5, -0.01])
        self.solve()
