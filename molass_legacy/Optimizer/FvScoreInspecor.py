"""
    Optimizer.FvScoreInspecor.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.ScrolledFrame import ScrolledFrame
from molass_legacy._MOLASS.SerialSettings import get_setting

class FvScoreInspecor(Dialog):
    def __init__(self, parent, js_canvas):
        self.parent = parent
        self.js_canvas = js_canvas
        self.dsets = dsets = js_canvas.dsets
        self.optimizer = js_canvas.fullopt
        self.score_names = self.optimizer.get_score_names(major_only=True)
        self.x_array = js_canvas.demo_info[1]
        Dialog.__init__(self, parent, "FV Score Inspection", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = ScrolledFrame(body_frame)
        cframe.pack(side=Tk.LEFT)
        cframe.add_bind_mousewheel()

        cframe_ = Tk.Frame(cframe.interior)
        cframe_.pack()
        tframe_ = Tk.Frame(cframe.interior)
        tframe_.pack(side=Tk.LEFT)

        self.compute_scores()

        fig = plt.figure(figsize=(18,11))
        self.fig = fig
        self.draw_scores()
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe_)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe_)
        self.toolbar.update()
        fig.tight_layout()
        self.mpl_canvas.draw()

    def compute_scores(self):
        fv_variations = []
        # NUM_MAJOR_SCORES = get_setting("NUM_MAJOR_SCORES")
        # self.score_variations = np.zeros((self.x_array.shape[0], NUM_MAJOR_SCORES))
        self.score_variations = np.zeros((self.x_array.shape[0], 13))
        for i, p in enumerate(self.x_array):
            p = self.x_array[i,:]
            ret = self.optimizer.objective_func(p, return_full=True)
            if np.isscalar(ret):
                fv = ret
                score_list = np.ones(6)*np.nan
            else:
                fv, score_list = ret[0:2]
            print([i], fv, score_list)
            fv_variations.append(fv)
            self.score_variations[i,:] = score_list

        self.fv_variations = np.array(fv_variations)

    def draw_scores(self):
        fig = self.fig
        ax = fig.add_subplot(111)
        ax.set_title("Variation of FV Scores", fontsize=20)

        for i, label in enumerate(self.score_names):
            ax.plot(self.score_variations[:,i], ":", label=label)

        ax.plot(self.fv_variations, color="red", label="FV")

        ax.set_ylim(-2, 2.5)

        ax.legend()
