"""
    SecTheory.PeakAsymmetry.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import sys
import os
if len(sys.argv) > 1 and sys.argv[1].find("demo") >= 0:
    this_dir = os.path.dirname(os.path.abspath( __file__ ))
    sys.path.append(this_dir + '/..')
    import molass_legacy.KekLib, SerialAnalyzer, DataStructure, Decomposer

import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RadioButtons, CheckButtons, Slider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.Peaks.ElutionModels import egh_pdf, emg
from SecTheory.LanJorgensonEgh import compute_tau, compute_sigma

Ti = -1000

class DisplaySlider(Slider):
    def __init__(self, *args, **kwargs):
        Slider.__init__(self, *args, **kwargs)

    def _update(self, event):
        pass

    def on_changed(self, func):
        pass

class PeakAsymmetry(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.radio2 = None
        self.radio2_ax = None
        self.slider_axes = None
        self.fixed_index = 0
        self.updating = -1
        self.check_list = None
        self.last_change = time()
        Dialog.__init__(self, parent, "Peak Asymmetry Demo", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X)

        fig = plt.figure(figsize=(18,6))
        self.fig = fig

        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        self.start_interactive()

        self.popup_menu = None
        # self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def start_interactive(self):
        fig = self.fig
        fig.suptitle("Plate Theory Constraint Demo", fontsize=20)

        gs = GridSpec(1,3)
        ax1 = fig.add_subplot(gs[0,0:2])

        N = 10000
        mu = 200
        tr = mu - Ti
        sigma = tr/np.sqrt(N)
        tau = 0
        self.currect_params = [N, mu, sigma, tau]

        self.x = x = np.linspace(100, 700, 600)
        self.curve, = ax1.plot(x, egh_pdf(x, mu, sigma, tau))

        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, 0.08)

        self.recs = [
            ["N", 5000, 16000, N],
            [r"$\mu$", 150, 650, mu],
            [r"$\sigma$", 10, 20, sigma],
            [r"$\tau$", 0, 10, tau],
            ]

        self.reset_radio2()

        rax = fig.add_axes([0.7, 0.7, 0.07, 0.2])
        rax.set_title("Models", fontsize=14)
        # radio1 = RadioButtons(rax, ('Gaussian', 'EGH', 'EMG'))
        radio1 = RadioButtons(rax, ('Gaussian', 'EGH'))
        radio1.on_clicked(self.radio1_clicked)
        self.radio1 = radio1
        self.current_model = radio1.value_selected

        self.gau_updaters = [self.slider0_update, self.slider1_update, self.slider2_update, self.slider3_update]
        self.egh_updaters = [self.slider0_egh_update, self.slider1_egh_update, self.slider2_egh_update, self.slider3_egh_update]

        self.reset_sliders([0, 1, 2], active_index=0)

        fig.tight_layout()
        self.mpl_canvas.draw()

    def radio1_clicked(self, label):
        print(label)
        self.current_model = label
        if label == 'Gaussian':
            self.reset_radio2(tau=False)
            self.currect_params[3] = 0
            self.reset_sliders([0, 1, 2], active_index=0)
        else:
            self.reset_check()
            self.reset_sliders([0, 1, 2, 3], active_index=0)

        fig = self.fig
        fig.canvas.draw_idle()

    def radio2_clicked(self, label):
        i = self.radio2_labels.index(label)
        print(label, [i])
        self.fixed_index = i
        self.reset_sliders(self.slider_indeces, active_index=i)

    def reset_radio2(self, tau=False):
        if self.radio2 is not None:
            # self.radio2.remove()
            pass

        self.check_list = None
        fig = self.fig
        rax = fig.add_axes([0.8, 0.7, 0.07, 0.2])
        rax.set_title("Parameters", fontsize=14)
        labels =  ("N", r"$\mu$", r"$\sigma$")
        if tau:
            labels += (r"$\tau$",)
        radio2 = RadioButtons(rax, labels)
        radio2.on_clicked(self.radio2_clicked)
        self.radio2_labels = labels
        self.radio2_ax = rax
        self.radio2 = radio2

    def check_clicked(self, label):
        i = self.radio2_labels.index(label)
        print("before", label, [i], self.check_list)
        self.check_list[i] = not self.check_list[i]
        print("after", label, [i], self.check_list)
        self.reset_sliders(self.slider_indeces, active_index=i)

    def reset_check(self):
        if self.radio2 is not None:
            # self.radio2.remove()
            pass

        fig = self.fig
        rax = fig.add_axes([0.8, 0.7, 0.07, 0.2])
        rax.set_title("Parameters", fontsize=14)
        labels =  ("N", r"$\mu$", r"$\sigma$", r"$\tau$")
        self.check_list = [True, True, False, False]
        radio2 = CheckButtons(rax, labels, actives=self.check_list)
        radio2.on_clicked(self.check_clicked)
        self.radio2_labels = labels
        self.radio2_ax = rax
        self.radio2 = radio2

    def slider0_update(self, val):
        if self.updating > 0:
            return

        # print([0], val)
        _, mu, sigma, tau = self.currect_params
        N = val
        self.updating = 0
        self.currect_params[0] = N

        if self.fixed_index == 1:
            tr = mu - Ti
            sigma = tr/np.sqrt(N)
            self.currect_params[2] = sigma
            self.sliders[2].set_val(sigma)
        else:
            # self.fixed_index == 2:
            tr = sigma*np.sqrt(N)
            mu = tr + Ti
            self.currect_params[1] = mu
            self.sliders[1].set_val(mu)

        y = egh_pdf(self.x, mu, sigma, tau)
        self.curve.set_ydata(y)

        self.last_change = time()
        self.after(200, self.slider_update_cleaner)
        self.fig.canvas.draw_idle()

    def slider_update_cleaner(self):
        interval = time() - self.last_change
        if interval > 1:
            self.updating = -1
        else:
            self.after(200, self.slider_update_cleaner)

    def slider1_update(self, val):
        if self.updating >= 0 and self.updating != 1:
            return

        # print([1], val)
        N, _, sigma, tau = self.currect_params
        mu = val
        self.updating = 1
        self.currect_params[1] = mu
        if self.fixed_index == 0:
            tr = mu - Ti
            sigma = tr/np.sqrt(N)
            self.currect_params[2] = sigma
            self.sliders[2].set_val(sigma)
        else:
            # self.fixed_index == 2:
            tr = mu - Ti
            N = int((tr/sigma)**2)
            self.currect_params[0] = N
            self.sliders[0].set_val(N)

        y = egh_pdf(self.x, mu, sigma, tau)
        self.curve.set_ydata(y)

        self.last_change = time()
        self.after(200, self.slider_update_cleaner)
        self.fig.canvas.draw_idle()

    def slider2_update(self, val):
        if self.updating >= 0 and self.updating != 2:
            return

        # print([2], val)
        N, mu, _, tau = self.currect_params
        sigma = val
        self.updating = 2
        self.currect_params[2] = sigma

        if self.fixed_index == 0:
            tr = sigma*np.sqrt(N)
            mu = tr + Ti
            self.currect_params[1] = mu
            self.sliders[1].set_val(mu)
        else:
            # self.fixed_index == 1
            tr = mu - Ti
            N = int((tr/sigma)**2)
            self.currect_params[0] = N
            self.sliders[0].set_val(N)

        y = egh_pdf(self.x, mu, sigma, tau)
        self.curve.set_ydata(y)

        self.last_change = time()
        self.after(200, self.slider_update_cleaner)
        self.fig.canvas.draw_idle()

    def slider3_update(self, val):
        print([3], val)
        self.fig.canvas.draw_idle()

    def slider0_egh_update(self, val):
        print([0], val)
        self.fig.canvas.draw_idle()

    def slider1_egh_update(self, val):
        print([1], val)
        self.fig.canvas.draw_idle()

    def slider2_egh_update(self, val):
        if self.updating >= 0 and self.updating != 2:
            return

        # print([2], val)
        N, mu, _, tau = self.currect_params
        sigma = val
        self.updating = 2
        self.currect_params[2] = sigma

        tau = compute_tau(Ti, N, mu, sigma)
        self.currect_params[3] = tau
        self.sliders[3].set_val(tau)

        y = egh_pdf(self.x, mu, sigma, tau)
        self.curve.set_ydata(y)

        self.last_change = time()
        self.after(200, self.slider_update_cleaner)
        self.fig.canvas.draw_idle()

    def slider3_egh_update(self, val):
        if self.updating >= 0 and self.updating != 3:
            return

        # print([3], val)
        N, mu, sigma, _ = self.currect_params
        tau = val
        self.updating = 3
        self.currect_params[3] = tau

        sigma = compute_sigma(Ti, N, mu, tau)
        self.currect_params[2] = sigma
        self.sliders[2].set_val(sigma)

        y = egh_pdf(self.x, mu, sigma, tau)
        self.curve.set_ydata(y)

        self.last_change = time()
        self.after(200, self.slider_update_cleaner)
        self.fig.canvas.draw_idle()

    def reset_sliders(self, indeces, active_index=0):
        if self.slider_axes is not None:
            for ax in self.slider_axes:
                ax.remove()

        for rec, val in zip(self.recs, self.currect_params):
            rec[-1] = val

        fig = self.fig
        slider_axes = []
        sliders = []
        for i, j in enumerate(indeces):
            label, valmin, valmax, valinit =  self.recs[j]
            ax = fig.add_axes([0.7, 0.6 - 0.1*i, 0.25, 0.03])
            slider_axes.append(ax)
            if self.check_list is None:
                display_only = j == active_index
            else:
                display_only = self.check_list[j]
            if display_only:
                handle_style = dict(size=0)
                slider  = DisplaySlider(ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit, handle_style=handle_style)
            else:
                slider  = Slider(ax, label=label, valmin=valmin, valmax=valmax, valinit=valinit)

            if self.check_list is None:
                slider.on_changed(self.gau_updaters[j])
            else:
                slider.on_changed(self.egh_updaters[j])
            sliders.append(slider)
        self.slider_indeces = indeces
        self.slider_axes = slider_axes
        self.sliders = sliders

def demo():
    from molass_legacy.KekLib.TkUtils import get_tk_root
    import seaborn
    seaborn.set()
    root = get_tk_root()
    def show_dialog():
        dialog = PeakAsymmetry(root)
        dialog.show()
        dialog.quit()
    root.after(0, show_dialog)
    root.mainloop()

if __name__ == '__main__':
    demo()
