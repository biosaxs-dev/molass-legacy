"""
    EdPlotter.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.cm as cm
from matplotlib import animation
from .EdBoundary import guess_boundary_value

class EdPlotter:
    def __init__(self, fig, ax, data, cmap, file, denss_results=True, log_folder=None):
        self.pause = False
        self.fig = fig
        self.ax = ax
        self.total_angle = 360
        # fig.canvas.mpl_connect('button_press_event', self.on_click)

        n = data.shape[0]
        boundary = guess_boundary_value(data, debug=False)
        # print('boundary=', boundary)
        w = np.where(data > boundary)
        wi = np.array(w, dtype=int).T

        xyz = wi - n/2
        # print(xyz.shape)
        # print(xyz)

        cx, cy, cz = np.average(xyz, axis=0)
        x = xyz[:,0]
        y = xyz[:,1]
        z = xyz[:,2]
        d = (x-cx)**2 + (y-cy)**2 + (z-cz)**2
        m = np.average(d)
        s = np.std(d)
        s3y = m+s*3

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            plt.plot(d)
            plt.plot([0, len(d)-1], [m, m], ':')
            plt.plot([0, len(d)-1], [s3y, s3y], ':')
            plt.show()

        # remove too distant voxels
        xyz_ = xyz[d<s3y,:]
        w_ = (w[0][d<s3y], w[1][d<s3y], w[2][d<s3y])
        self.v = v = data[w_]

        vmin = v.min()
        vmax = v.max()
        vmin_ = vmin*1.1 + vmax*(-0.1)
        vmax_ = vmin*(-0.1) + vmax*1.1

        ax.set_axis_off()
        ax.set_title("Density Distribution in Voxels", fontsize=20, y=1.08)

        self.sc = sc = ax.scatter3D(xyz_[:,0], xyz_[:,1], xyz_[:,2], vmin=vmin_, vmax=vmax_, c=v, cmap=cmap, alpha=1, s=30)

        if denss_results:
            """
            Since the texts below must be opsitioned with figure coordinates,
            we need to trasform the box position.

            c.f.
            https://stackoverflow.com/questions/41267733/getting-the-coordinates-of-a-matplotlib-annotation-label-in-figure-coordinates
            """
            box = ax.patch.get_extents()
            tcbox = fig.transFigure.inverted().transform(box)
            x0, y0 = tcbox[0,:]
            x1, y1 = tcbox[1,:]
            # print('x0, x1=', x0, x1)
            tx1 = x0*0.99 + x1*0.01
            tx2 = x0*0.77 + x1*0.23
            tx3 = x0*0.6  + x1*0.4

            if log_folder is not None:
                try:
                    ret = get_log_items(log_folder + '/denss.log')
                    for xy, text in [ ((tx1, y0), r'$\chi^2$=%.3g' % ret.get('Chi2')),
                                    ((tx2, y0), r'$R_g$=%.3g' % ret.get('Rg')),
                                    ((tx3, y0), r'Support Volume=%.3g' % ret.get('Support Volume'))
                                    ]:
                        fig.text(*xy, text, fontsize=16)
                except:
                    from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                    etb = ExceptionTracebacker()
                    print(etb)

        fig.colorbar(sc, ax=ax)
        self.set_cubic_limits(n//4, ax)

    def make_anim(self, random=False):
        fig = self.fig
        ax = self.ax

        def init():
            # print('init:')
            return []

        self.azim = 0
        self.elev = 10
        def animate_(i):
            # print('animate', i)
            if random:
                if not self.pause:
                    self.azim += np.sign(np.random.rand()-0.5)*10
                    self.elev += np.sign(np.random.rand()-0.5)*10
            else:
                if not self.pause:
                    self.azim += 1
                    if self.azim == self.total_angle:
                        self.azim = 0
            ax.view_init(elev=self.elev, azim=self.azim)
            return []

        self.anim = animation.FuncAnimation(fig, animate_, init_func=init,
                                   frames=self.total_angle, interval=20, blit=True)

    def set_cubic_limits(self, min_size, ax):
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        zmin, zmax = ax.get_zlim()

        size = np.max([min_size, xmax-xmin, ymax-ymin, zmax-zmin])/2

        def get_limits(_min, _max):
            _mid = (_min+_max)/2
            return _mid - size, _mid + size

        ax.set_xlim(*get_limits(xmin, xmax))
        ax.set_ylim(*get_limits(ymin, ymax))
        ax.set_zlim(*get_limits(zmin, zmax))

    def on_click(self, event):
        self.pause ^= True

def ed_scatter(fig, axes, data, file):
    filename = get_name_for_title(file)
    fig.suptitle('Electron Density from ' + filename, fontsize=24)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    ax1, ax2 = axes

    # cmap = cm.bwr
    # cmap = cm.hot
    cmap = cm.plasma

    esc = EdPlotter(fig, ax1, data, cmap, file)

    v = esc.v
    h = np.histogram(v)
    # print('h=', h)

    colors = cmap(h[1])
    ax2.set_title("Volume Proportions in Voxels", fontsize=20)
    ax2.pie(h[0], colors=colors, startangle=90, counterclock=False, center=(0, 0.2), radius=0.6)

    ax2.set_ylim(-1, 1)

    pc = len(v)/np.prod(data.shape)*100

    ax2.text(0.0, -0.8,
        "This chart shows the volume proportions within\n"
        "the visible marked voxels in the left 3D figure,\n"
        "which in all occupy only %.2g%% voxels of\n"
        "the whole %dx%dx%d cubic space." % (pc, *data.shape),
        ha='center', fontsize=16)

    return esc

def get_log_items(path):
    from molass_legacy.DENSS.DenssUtils import get_denss_log_items
    return get_denss_log_items(path)

def get_name_for_title(file):
    nodes = file.replace('\\', '/').split('/')
    name = '/'.join(nodes[-4:])
    return name
