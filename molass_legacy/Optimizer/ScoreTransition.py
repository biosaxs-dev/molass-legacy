"""
    Optimizer.ScoreTransition.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.Optimizer.NaviFrame import NaviFrame
from molass_legacy.Optimizer.FvScoreConverter import convert_score

class ScoreTransition(Dialog):
    def __init__(self, parent, js_canvas, optimizer, x_array, fv_array, best_index):
        self.optimizer = optimizer
        self.js_canvas = js_canvas
        self.x_array = x_array
        self.fv_scores = fv_array[:,1]
        self.best_index = best_index
        Dialog.__init__(self, parent, "Score Transition", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack()

        gs = GridSpec(8,1)

        fig = plt.figure(figsize=(8,8))
        ax0 = fig.add_subplot(gs[0:7,:])
        ax1 = fig.add_subplot(gs[7,:])
        # ax1.set_axis_off()
        self.fig = fig
        self.axes = ax0, ax1
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1)
        self.draw_score(self.best_index)

        nframe = NaviFrame(bframe, self, arrows_only=True)
        nframe.pack()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def get_params(self, index):
        return self.x_array[index]

    def draw_score(self, index, devel=True):
        if devel:
            import Optimizer.TheDebugUtils
            from importlib import reload
            reload(Optimizer.TheDebugUtils)
        from molass_legacy.Optimizer.TheDebugUtils import plot_scores

        self.curr_index = index

        fig = self.fig
        ax0, ax1 = self.axes
        optimizer = self.optimizer
        # in_folder, seeds, result = fig_info

        score_names = optimizer.get_score_names()

        params = self.get_params(self.best_index)
        fv, score_list = optimizer.objective_func(params, return_full=True)[0:2]
        ax0.cla()
        ax1.cla()
        ax1.set_ylim(0, 1)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.set_yticks([0.5], ["evaluation position"])

        i = self.best_index
        ax1.plot([i,i], [0,1], color="red", lw=2, label="best [%d]" % self.best_index)

        dx = len(self.fv_scores)*0.05
        ax1.set_xlim(-dx, len(self.fv_scores)+dx)

        if index == self.best_index:
            score = convert_score(self.fv_scores[index])
            ax0.set_title("Score at Best %.1f at [%d]" % (score, index), fontsize=16)
        else:
            if index < self.best_index:
                i, j = (index, self.best_index)
            else:
                i, j = (self.best_index, index)
            scores = [convert_score(self.fv_scores[k]) for k in [i, j]]
            ax0.set_title("Score Transition from %.1f at [%d] to %.1f at [%d]" % (scores[0], i, scores[1], j), fontsize=16)
            ax1.plot([index,index], [0,1], color="yellow", lw=2, label="current [%d]" % index)

        if index == self.best_index:
            plot_scores(ax0, score_list, score_names, label="best [%d]" % self.best_index)
        else:
            plot_scores(ax0, score_list, score_names, label="best [%d]" % self.best_index, alpha=0.5)
            params = self.get_params(index)
            fv, score_list = optimizer.objective_func(params, return_full=True)[0:2]
            plot_scores(ax0, score_list, score_names, label="current [%d]" % self.curr_index, alpha=0.5, invert=False, add_patch=False)
        ax0.legend(loc="lower right")
        ax1.legend()

        fig.tight_layout()
        self.mpl_canvas.draw()

    def get_first(self):
        self.draw_score(0)

    def get_previous_best(self):
        index = self.get_best_index(stop=self.curr_index)
        self.draw_score(index)

    def get_previous(self):
        if self.curr_index > 0:
            index = self.curr_index - 1
        else:
            index = self.curr_index
        self.draw_score(index)

    def get_best_index(self, stop=None):
        m = np.argmin(self.fv_scores[:stop,1])
        return m

    def get_best(self):
        self.draw_score(self.best_index)

    def get_next(self):
        if self.curr_index < len(self.fv_scores) - 1:
            index = self.curr_index + 1
        else:
            index = self.curr_index
        self.draw_score(index)

    def get_next_best(self):
        fv_scores = self.fv_scores
        stop = min(len(fv_scores), self.curr_index + 1)
        best_fv_until_now = np.min(fv_scores[:stop,1])
        w  = np.where(fv_scores[stop:,1] < best_fv_until_now)[0]
        if len(w) > 0:
            self.draw_score(stop + w[0])

    def get_last(self):
        self.draw_score(len(self.fv_scores)-1)
