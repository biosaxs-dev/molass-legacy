# coding: utf-8
"""
    RankInspection.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as ticker
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar, get_color
from OurPlotUtils import draw_as_image
from SvdDenoise import get_denoised_data

NUM_RANKS = 5

@ticker.FuncFormatter
def major_formatter(x, pos):
    return "%.0f" % x

class RankInspectionFigure( Dialog ):
    def __init__( self, parent, dialog, M, C, q, eslice, row):
        self.grab = 'local'     # used in grab_set
        self.parent = parent
        self.dialog = dialog
        self.M = M
        self.svd = np.linalg.svd(M)
        self.C = C
        self.q = q
        self.eslice = eslice
        self.row = row
        self.Cpinv = np.linalg.pinv(self.C)
        print(self.M.shape, self.C.shape, self.Cpinv.shape)
        Dialog.__init__( self, parent, "Rank Analysis", visible=False )

    def show( self ):
        self._show()

    def body( self, body_frame ):
        tk_set_icon_portable( self )

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        nrows = 5
        self.fig = fig = plt.figure( figsize=(15, 8) )
        gs = GridSpec( nrows*2, 2 )
        axLs = []
        for row in range(nrows):
            row2 = row*2
            ax = fig.add_subplot(gs[row2:row2+2, 0])
            axLs.append(ax)
        axR1 = fig.add_subplot(gs[0:nrows, 1])
        axR2 = fig.add_subplot(gs[nrows:nrows*2, 1])
        fig.tight_layout()

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        fig.suptitle("Rank Inspection in Elution Range %d-%d" % (self.eslice.start, self.eslice.stop), fontsize=20)

        try:
            ax0 = axLs[0]
            axins = inset_axes(ax0, width=2, height=1.4,
                                bbox_to_anchor=(0, 0, 2.5, 2.5),
                                bbox_transform=ax0.transAxes, loc='upper left')
            axins.set_axis_off()
            from_ax = self.dialog.axis_array[self.row][0]
            draw_as_image(axins, self.dialog.fig, from_ax)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)

        axR1.set_title("Top Five Singular Values", fontsize=16)
        axR1.set_ylabel("Scale", fontsize=12)
        axR1.xaxis.set_major_locator(plt.NullLocator())
        n = 5
        sigmas = self.svd[1]
        axR1.plot(np.arange(5), sigmas[0:n], ':', marker='o')

        axR2.set_title('Deviations caused by SVD-denoise with various ranks', fontsize=16)
        axR2.set_ylabel('nRMSD', fontsize=12)
        axR2.set_xlabel('Rank used in denoising', fontsize=12)

        axR2.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
        axR2.xaxis.set_major_formatter(major_formatter)

        P0 = np.dot(self.M, self.Cpinv)
        y0 = P0[:,0]
        max_y = np.percentile(y0, 95)

        nrmsd_list = []
        for i in range(NUM_RANKS):
            rank = i + 1
            D = get_denoised_data(None, rank=rank, svd=self.svd)
            P_ = np.dot(D, self.Cpinv)
            y1 = P_[:,0]
            ax = axLs[i]
            ax.set_ylabel('$Log_{10}$(I)')
            ax.plot(self.q, np.log10(y0), label='A(q) from raw data')
            ax.plot(self.q, np.log10(y1), label='A(q) from denoised data')
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = xmin*0.6 + xmax*0.4
            ty = ymin*0.9 + ymax*0.1
            ax.text(tx, ty, "Rank %d denoise" % rank, fontsize=20, alpha=0.3)
            ax.legend()
            nrmsd = np.sqrt(np.average((y0 - y1)**2))/max_y
            nrmsd_list.append(nrmsd)

        nrmsd_array = np.array(nrmsd_list)
        ranks = np.arange(NUM_RANKS) + 1
        axR2.plot(ranks, nrmsd_array, marker='o')

        fig.subplots_adjust( top=0.88, bottom=0.08, left=0.07, right=0.95, wspace=0.2 )
        self.mpl_canvas.draw()

    def buttonbox( self ):
        bottom_frame = Tk.Frame(self)
        bottom_frame.pack(fill=Tk.BOTH, expand=1)

        width = int(self.mpl_canvas_widget.cget('width'))
        padx = width*0.1

        tframe = Tk.Frame(bottom_frame)
        tframe.pack(side=Tk.LEFT, padx=padx)
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        box = Tk.Frame(bottom_frame)
        box.pack(side=Tk.RIGHT, padx=padx*2)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
