"""
    Wave.Beam.py

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

        x = np.linspace(0, 10*np.pi, 100)
        y = np.sin(x)

        ax1.set_ylim(-3, 3)
        line, = ax1.plot(x, y)

        ax2.set_ylim(0,1)
        z = np.ones(len(x))*0.5
        artists = ax2.scatter(x, z, vmin=-1, vmax=1, c=y, cmap=cm.plasma)

        fig.tight_layout()

        def init():
          line.set_data([],[])
          artists.set_array(y)
          return [line, artists]

        num_frames = 40

        def animate(i):
          y = np.sin(x-10*np.pi*i/num_frames)
          line.set_data(x,y)
          artists.set_array(y)
          return [line, artists]

        self.anim = animation.FuncAnimation(fig, animate, init_func=init,
                                        frames=num_frames, interval=100, blit=True)

        # self.anim.save('beam-wave.mp4', writer="ffmpeg")

def demo1():
    from molass_legacy.KekLib.TkUtils import get_tk_root

    root = get_tk_root()

    def show_demo():
        dialog = BeamAnimDialog(root)
        dialog.show()
        root.quit()

    root.after(100, show_demo)
    root.mainloop()
    root.destroy()
