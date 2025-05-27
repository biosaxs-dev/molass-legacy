"""
    Wave.Scattering2.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class BeamAnimDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, "Beam Animatio", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.BOTH, expand=1)
        tframe = Tk.Frame(bframe)
        tframe.pack(side=Tk.LEFT)
        pframe = Tk.Frame(bframe)
        pframe.pack(side=Tk.RIGHT)

        fig, axes = plt.subplots(ncols=2, figsize=(14, 7))
        self.fig  = fig
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()
        self.draw_anim(fig, axes)
        self.mpl_canvas.draw()

    def draw_anim(self, fig, axes):
        ax1, ax2 = axes

        for ax in axes:
            ax.grid(False)

        N = 800
        n = N//2
        r = np.arange(N)
        rr = r**2
        i = np.arange(N)
        ii, jj = np.meshgrid(i, i)
        labels1 = np.searchsorted(rr, (ii - n)**2 + (jj-n)**2)

        canvas = np.zeros((N, N))

        def color_filter(v, maxv):
            return 1/(1 + np.exp(-2*(v - (maxv-0.01))))

        cmap = 'plasma'
        target_particle1 = labels1 <= 5

        d = 15
        m = n - d
        labels2 = np.searchsorted(rr, (ii - m)**2 + (jj-m)**2)
        target_particle2 = labels2 <= 5

        Lmax = np.max([labels1, labels2])
        x = np.linspace(0, np.pi*96, N+Lmax)

        def animate_func(i):
            t = np.pi*i/2
            beam1 = color_filter(1 + np.sin(x[0:n] - t), 2)
            canvas[n-2:n+3,0:n] = beam1     # as if linewith=5
            beam2 = color_filter(1 + np.sin(x[0:m] - t), 2)
            canvas[m-2:m+3,0:m] = beam2     # as if linewith=5
            canvas[target_particle1] = 1
            canvas[target_particle2] = 1
            im1 = ax1.imshow(canvas, animated=True, cmap=cmap, origin='lower')
            scattered1 = 1 + np.sin(x[n:] - t)
            intensity1 = scattered1[labels1]
            scattered2 = 1 + np.sin(x[m:] - t)
            intensity2 = scattered2[labels2]
            im2 = ax2.imshow(color_filter(intensity1+intensity2, 4), animated=True, cmap=cmap, origin='lower')
            return [im1, im2]

        fig.tight_layout()
        self.ani = animation.FuncAnimation(fig, animate_func, frames=60, interval=50, blit=True)
        # self.ani.save('scattering.mp4', writer="ffmpeg")

def demo2():
    from molass_legacy.KekLib.TkUtils import get_tk_root

    root = get_tk_root()

    def show_demo():
        dialog = BeamAnimDialog(root)
        dialog.show()
        root.quit()

    root.after(100, show_demo)
    root.mainloop()
    root.destroy()
