"""
    Qmm3dFigure.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from bisect import bisect_right
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from MatrixData import simple_plot_3d

COLOR_ZOOMED = 'yellow'

class Qmm3dFigure(Dialog):
    def __init__(self, parent, dataset, frame):
        self.dataset = dataset
        self.frame = frame
        Dialog.__init__(self, parent, title="QMM 3d Figure", visible=False)

    def show(self):
        self._show()

    def body(self, bframe):

        fig = plt.figure(figsize=(16,7))
        self.mpl_canvas = FigureCanvasTkAgg(fig, bframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        self.find_where_to_zoom()
        self.draw_whole(ax1)
        self.draw_zoomed(ax2)

        fig.tight_layout()
        self.mpl_canvas.draw()

    def draw_whole(self, ax):
        ax.set_title("Linear Plot of the Whole Region", fontsize=16)
        self.dataset.plot_3d(ax, alpha=0.3)
        simple_plot_3d(ax, self.zoom_data, x=self.zoom_x, y=self.zoom_y, color=COLOR_ZOOMED, edgecolors=COLOR_ZOOMED)
        ax.set_xlabel("$Q$")
        ax.set_ylabel("$Eno$")
        ax.set_zlabel("$Intensity$")

    def draw_zoomed(self, ax):
        ax.set_title("Guinier Plot of the Small Angle Peak Region", fontsize=16)
        data = np.log(self.zoom_data)
        y = self.zoom_y
        px = self.zoom_x**2
        simple_plot_3d(ax, data, x=px, y=y, color=COLOR_ZOOMED, alpha=0.2)

        frame = self.frame
        gy = frame.gy_list[0]
        mu = frame.mu_list[0]

        aslice = self.zoom_info[0]
        e11n = frame.result_list[0]  # e11n
        for k in self.zoom_w:
            m = int(round(mu[k]))
            py = np.ones(len(px))*m
            z = e11n.P[aslice, k]
            pz = np.log(z*gy[m])
            ax.plot(px, py, pz, label='c-%d' % k, color='C%d' % (k+2))

        ax.legend()
        ax.set_xlabel("$Q^2$")
        ax.set_ylabel("$Eno$")
        ax.set_zlabel("$Ln(Intensity)$")

    def find_where_to_zoom(self):
        self.zoom_info = find_where_to_zoom_impl(self.dataset, self.frame)
        aslice, eslice, w = self.zoom_info
        self.zoom_data = self.dataset.data[aslice, eslice]
        self.zoom_x = self.dataset.vector[aslice]
        self.zoom_y = np.arange(eslice.start, eslice.stop)
        self.zoom_w = w

def find_where_to_zoom_impl(dataset, frame):
    astart = 0
    x = dataset.vector
    astop = bisect_right(x, 0.04)

    gy = frame.gy_list[0]
    C = frame.C_list[0]
    mu = frame.mu_list[0]
    sigma = frame.sigma_list[0]
    max_y = np.max(gy)
    n = len(mu)
    flags = np.zeros(n, dtype=int)
    for k, c in enumerate(C[:n]):
        m = int(mu[k])
        ratio = c[m]/max_y
        flags[k] = round(ratio)

    w = np.where(flags==1)[0]
    w0 = w[0]
    estart = int(round(mu[w0] - sigma[w0]))
    w1 = w[-1]
    estop = int(round(mu[w1] + sigma[w1])) + 1
    print('flags=', flags, w, estart, estop)
    return slice(astart, astop), slice(estart, estop), w
