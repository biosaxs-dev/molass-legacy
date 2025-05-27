# coding: utf-8
"""
    QuadMmUtils.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lmfit import minimize, Parameters
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
import molass_legacy.KekLib.DebugPlot as dplt
from .PairedDataSets import PairedDataSets, WAVELENGTH_POINTS, ANGLE_POINTS
from .ProbDensityUtils import plot_hist_data_list
from .EghMixture import egh_pdf, EghMixture
from .QuadMM import QuadMM
from DataUtils import get_in_folder

def set_consistent_base(ax1, axt):
    ymin1, ymax1 = ax1.get_ylim()
    ymint, ymaxt = axt.get_ylim()
    y_ = ymin1/ymax1*ymaxt
    axt.set_ylim(y_, ymaxt)

def get_components(x, y, gmm):
    gy_list = []
    ty = np.zeros(len(x))
    for k in range(len(gmm.pi)):
        w = gmm.pi[k]
        m = gmm.tR[k]
        s = gmm.sigma[k]
        t = gmm.tau[k]
        print([k], w, m, s, t)
        gy = egh_pdf(x, m, s, t, w)
        gy_list.append([gy, m])
        ty += gy

    sorted_gy_list = sorted(gy_list, key=lambda x:x[1])

    if False:
        dplt.push()
        fig = dplt.figure()
        ax = fig.gca()
        ax.plot(ty)
        for k, rec in enumerate(sorted_gy_list):
            gy = rec[0]
            ax.plot(gy, ':', label=str(k))

        ax.legend()
        fig.tight_layout()
        dplt.show
        dplt.pop()

    def obj_func(params):
        S   = params['S']
        return ty*S - y

    params = Parameters()
    S_init = np.max(y)/np.max(ty)
    params.add('S', value=S_init, min=0, max=S_init*100 )
    result = minimize(obj_func, params, args=())

    scale = result.params['S'].value
    return [ scale*rec[0] for rec in sorted_gy_list ], ty*scale

def plot_components(ax, x, cgys):
    ret_lines = []
    for k, gy in enumerate(cgys):
        line, = ax.plot(x, gy, ':', label='c-%d' % k)
        ret_lines.append(line)
    return ret_lines

def plot_decompositions(y_list, data_list, k, max_iter=100, axes=None, titles=None, plot_class=plt, anim_data=False):
    from DataUtils import get_in_folder

    states = np.random.randint(1000, 9999, 4)
    print('random_states:', states)
    qmm = QuadMM(EghMixture, k, max_iter=max_iter, random_states=states, anim_data=anim_data)
    qmm.unified_fit(X=[np.expand_dims(data,1) for data in data_list], bins=[len(y) for y in y_list])

    if axes is None:
        fig, axes = plot_class.subplots(nrows=1, ncols=4, figsize=(20,4))
        fig.suptitle("Decomposition of %s using Dual EGH Mixture Model" % get_in_folder(), fontsize=20)

    mm_list = []
    xy_list = []
    for n, (ax, y, data) in enumerate(zip(axes, y_list, data_list)):
        if titles is not None:
            ax.set_title(titles[n], fontsize=16)

        x = np.arange(len(y))
        ax.plot(x, y, label='input')
        mm = qmm.get_sub(n)
        mm_list.append(mm)
        xy_list.append((x, y))
        cgys, gy = get_components(x, y, mm)
        ax.plot(gy, label='total')
        plot_components(ax, x, cgys)
        ax.legend()

    if axes is None:
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

    return mm_list, xy_list

def get_anim_components(x, y, gmm, n):
    gy_list = []
    ty = np.zeros(len(x))
    for k in range(len(gmm.pi)):
        w = gmm.pi_array[n,k]
        m = gmm.tR_array[n,k]
        s = gmm.sigma_array[n,k]
        t = gmm.tau_array[n,k]
        # print([n,k], w, m, s, t)
        gy = egh_pdf(x, m, s, t, w)
        gy_list.append([gy, m])
        ty += gy

    sorted_gy_list = sorted(gy_list, key=lambda x:x[1])

    if True:
        dplt.push()
        fig = dplt.figure()
        ax = fig.gca()
        ax.plot(ty)
        for k, rec in enumerate(sorted_gy_list):
            gy = rec[0]
            ax.plot(gy, ':', label=str(k))

        ax.legend()
        fig.tight_layout()
        dplt.show
        dplt.pop()

    def obj_func(params):
        S   = params['S']
        return ty*S - y

    params = Parameters()
    S_init = np.max(y)/np.max(ty)
    params.add('S', value=S_init, min=0, max=S_init*100 )
    result = minimize(obj_func, params, args=())

    scale = result.params['S'].value
    return [ scale*rec[0] for rec in sorted_gy_list ], ty*scale

def animate_gmm_elution(y1, y2, gmm1, gmm2, plot_class=plt):
    from DataUtils import get_in_folder

    # fig, axes = plot_class.subplots(nrows=1, ncols=2, figsize=(14,7))     # does not work for animation
    # ax1, ax2 = axes

    fig = plot_class.figure(figsize=(14,7))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fig.suptitle("Decomposition of %s using EGH Mixture Model Animation" % get_in_folder(), fontsize=20)
    ax1.set_title("UV Elution Decomposition", fontsize=16)
    ax2.set_title("X-ray Elution Decomposition", fontsize=16)
    ax1.plot(y1, label='input data')
    ax2.plot(y2, label='input data')

    x1 = np.arange(len(y1))
    cgys1, gy1 = get_anim_components(x1, y1, gmm1, 0)
    line1, = ax1.plot(gy1, label='eghmm-fit')
    lines1 = plot_components(ax1, x1, cgys1)

    x2 = np.arange(len(y2))
    cgys2, gy2 = get_anim_components(x2, y2, gmm2, 0)
    line2, = ax2.plot(gy2, label='eghmm-fit')
    lines2 = plot_components(ax2, x2, cgys2)

    def add_number_text(ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        tx = (xmin + xmax)/2
        ty = (ymin + ymax)/2
        text = ax.text(tx, ty, "_", ha='center', va='center', alpha=0.05, fontsize=240)
        return text

    text1 = add_number_text(ax1)
    text2 = add_number_text(ax2)

    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    anim_artists = [line1] + lines1 + [line2] + lines2 + [text1, text2]

    def animate(i):
        # print('animate', i)

        cgys1, gy1 = get_anim_components(x1, y1, gmm1, i+1)
        line1.set_data(x1, gy1)
        for line, gy in zip(lines1, cgys1):
            line.set_data(x1, gy)

        cgys2, gy2 = get_anim_components(x2, y2, gmm2, i+1)
        line2.set_data(x2, gy2)
        for line, gy in zip(lines2, cgys2):
            line.set_data(x2, gy)

        text_num = str(i+1)
        text1.set_text(text_num)
        text2.set_text(text_num)

        dplt.update()   # need a chance to get events

        return anim_artists

    def reset():
        print('reset')
        return animate(-1)

    num_frames = gmm1.max_iter
    print('num_frames=', num_frames)
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, init_func=reset, interval=100)

    return anim

class SpikeDialog(Dialog):
    def __init__(self, parent, in_folder, **kwargs):
        self.datasets = datasets = PairedDataSets(in_folder, kwargs)
        self.continue_ = True
        self.animator = None
        self.show_extrapolated = kwargs.pop('show_extrapolated', False)
        y_list, data_list = datasets.generate_sample_datasets(quad=True)
        self.kwargs = kwargs
        Dialog.__init__(self, parent, title="SpikeDialog", visible=False)
        self.create_canvas(y_list, data_list)

    def body(self, bframe):
        self.cframe = Tk.Frame(bframe)
        self.cframe.pack()

    def buttonbox( self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Cancel", width=10, command=self._cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        w = Tk.Button(box, text="Next", width=10, command=self.draw_next)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        w = Tk.Button(box, text="Iterate", width=10, command=self.iterate)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        anim_data = self.kwargs.get('anim_data', False)
        state = Tk.NORMAL if anim_data else Tk.DISABLED
        w = Tk.Button(box, text="Animate", width=10, command=self.toggle_animation, state=state)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.toggle_btn = w

        w = Tk.Button(box, text="Step Mode", width=10, command=self.do_step)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.step_btn = w

        w = Tk.Button(box, text="Save Animation", width=16, command=self.save_animation)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.step_btn = w

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def _cancel(self):
        self.continue_ = False
        self.cancel()

    def create_canvas(self, y_list, data_list):
        cframe = self.cframe

        figsize = (20,12)
        nrows, ncols = 3, 4
        # self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig = plt.figure(figsize=(figsize))
        axes_list = []
        gs = GridSpec(nrows, ncols)
        for i in range(nrows):
            axes_row = []
            for j in range(ncols):
                projection = None if self.show_extrapolated or i < 2 else '3d'
                ax = fig.add_subplot(gs[i,j], projection=projection)
                axes_row.append(ax)
            axes_list.append(axes_row)

        self.fig = fig
        self.axes = np.array(axes_list)
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
        self.toolbar.update()

        self.fig.suptitle("Decomposition of %s using Quad EGH Mixture Model" % get_in_folder(), fontsize=20)
        titles = ["UV Data at %d" % WAVELENGTH_POINTS[0], "UV Data at %d" % WAVELENGTH_POINTS[1],
                    "Xray Data at %g" % ANGLE_POINTS[0], "Xray Data at %g" % ANGLE_POINTS[1]]
        plot_hist_data_list(y_list, data_list, axes=self.axes[0,:], titles=titles)
        n_components = self.kwargs.pop('n_components', 2)
        self.mm_args = [y_list, data_list, n_components]
        self.draw_next()
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)
        self.mpl_canvas.draw()

    def show(self):
        self._show()

    def draw_next(self, anim_data=False):
        self.toggle_btn.config(text="Animate")
        if self.animator is not None:
            self.animator.stop()
            self.animator = None

        axes = self.axes[1,:]
        for ax in axes:
            ax.cla()

        self.mm_list, self.xy_list = plot_decompositions(*self.mm_args, axes=axes, **self.kwargs)
        if self.show_extrapolated:
            for ano, (mm, xy, ax) in enumerate(zip(self.mm_list, self.xy_list, self.axes[2,:])):
                ax.cla()
                C = mm.get_anim_C(*xy, -1)
                self.datasets.draw_exprapolated(ano, ax, C)
        self.mpl_canvas.draw()

    def iterate(self):
        while self.continue_:
            self.draw_next()
            time.sleep(1)
            self.update()

    def create_animator(self, step_mode=False):
        from Prob.MmAnimator import MmAnimator
        axes = self.axes
        ylims = []
        for ax in axes[1,:]:
            ylims.append(ax.get_ylim())
            ax.cla()
        return MmAnimator(self.fig, axes, ylims, self.xy_list, self.mm_list, self.datasets,
                            step_mode=step_mode, show_extrapolated=self.show_extrapolated)

    def show_animation(self):
        self.animator = self.create_animator()

    def toggle_animation(self):
        text = self.toggle_btn.cget("text")
        if text == "Animate":
            self.show_animation()
            text = "Stop"
        elif text == "Stop":
            self.animator.stop()
            text = "Start"
        elif text == "Start":
            self.animator.start()
            text = "Stop"
        else:
            assert False
        self.toggle_btn.config(text=text)

    def do_step(self):
        text = self.step_btn.cget("text")
        if text == "Step Mode":
            self.start_step_mode()
            self.step_btn.config(text="Step")
        else:
            self.step_animator.step(1)
        self.mpl_canvas.draw()

    def start_step_mode(self):
        self.step_animator = self.create_animator(step_mode=True)

    def save_animation(self):
        self.animator = self.create_animator()
        self.animator.save('anim.mp4', writer="ffmpeg")
        self.animator.stop()

def spike_dialog(parent, in_folder, **kwargs):
    dialog = SpikeDialog(parent, in_folder, **kwargs)
    dialog.show()
