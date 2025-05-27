# coding: utf-8
"""
    ScatteringCurveViewer.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkUtils import adjusted_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable

class ScatteringCurveViewer( Dialog ):
    def __init__(self, parent, editor_frame, sd, ft, scaled_y):
        self.editor_frame = editor_frame
        self.uv_y = editor_frame.uv_y
        self.xr_y = scaled_y
        self.xr_j0 = sd.xr_j0
        self.qvector = sd.qvector
        self.data = sd.intensity_array[:,:,1].T
        self.ft = ft

        Dialog.__init__( self, parent, "Scattering Curve Viewer", visible=False )

    def show(self):
        self._show()

    def body( self, body_frame ):
        tk_set_icon_portable( self )

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        self.fig = fig
        self.axes = axes
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.linear_plot()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X)

        upper_subbox = Tk.Frame(box)
        upper_subbox.pack(fill=Tk.X)
        lower_subbox = Tk.Frame(box)
        lower_subbox.pack()

        w = Tk.Button(upper_subbox, text="to Kratky", width=10, command=self.kratky_plot)
        w.pack(side=Tk.RIGHT, padx=5, pady=5)
        w = Tk.Button(upper_subbox, text="to Log", width=10, command=self.log_plot)
        w.pack(side=Tk.RIGHT, padx=5, pady=5)
        w = Tk.Button(upper_subbox, text="to Linear", width=10, command=self.linear_plot, default=Tk.ACTIVE)
        w.pack(side=Tk.RIGHT, padx=5, pady=5)

        w = Tk.Button(lower_subbox, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(lower_subbox, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def kratky_plot(self):
        self.plain_plot(kratky=True)

    def log_plot(self):
        self.plain_plot(log=True)

    def linear_plot(self):
        self.plain_plot()

    def plain_plot(self, log=False, kratky=False):
        for ax in self.axes:
            ax.cla()
            ax.set_xlabel('Q')
            if kratky:
                ax.set_ylabel('I/C*$Q^2$')
            else:
                if log:
                    ax.set_yscale('log')
                    ax.set_ylabel('log(I/C)')
                else:
                    ax.set_ylabel('I/C')

        fig = self.fig
        ax1, ax2 = self.axes
        x = self.qvector
        if kratky:
            xx = x*x
        ft = self.ft
        ft_ = [j - self.xr_j0 for j in ft]
        ft_label = r'(No. %d$\sim$%d)' % tuple(ft)

        plot_type = 'Kratky' if kratky else ('Log' if log else 'Linear')
        fig.suptitle(plot_type + " Plot of Normalized Curves", fontsize=20)
        ax1.set_title("Mapped UV Normalized " + ft_label, fontsize=16)
        ax2.set_title("Scaled Xray Normalized " + ft_label, fontsize=16)

        for j in range(ft_[0], ft_[1]+1):
            y = self.data[:,j]
            if kratky:
                ax1.plot(x, y/self.uv_y[j]*xx)
                ax2.plot(x, y/self.xr_y[j]*xx)
            else:
                ax1.plot(x, y/self.uv_y[j])
                ax2.plot(x, y/self.xr_y[j])

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        self.mpl_canvas.draw()
