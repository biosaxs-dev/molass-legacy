"""
Peaks.ComponentsView.py

A thin Tkinter dialog that embeds the library ``plot_components_impl``
figure as the "Complementary View" replacement.

This is the library-based successor to
``molass_legacy.Optimizer.ComplementaryView``.  It is used by
``PeakEditor.show_complementary_view()`` when a library ``Decomposition``
object is available (i.e. when ``self.decomposition`` is not None).
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog


class ComponentsView(Dialog):
    """Dialog that shows the library plot_components figure for a Decomposition."""

    def __init__(self, parent, decomposition):
        self.parent = parent
        self.decomposition = decomposition
        Dialog.__init__(self, parent, "Plot Components (Library)", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig = plt.figure(figsize=(18, 9))
        from molass.PlotUtils.DecompositionPlot import plot_components_impl
        # Use the cached Rg curve if the precompute thread has finished.
        # Read from decomposition._rgcurve (set by _build_library_decomposition),
        # NOT from decomposition.ssd._rgcurve which lives on a different ssd object.
        rgcurve = getattr(self.decomposition, '_rgcurve', None)
        plot_components_impl(self.decomposition, fig=fig, rgcurve=rgcurve)

        self.fig = fig
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()

    def buttonbox(self):
        lower_frame = Tk.Frame(self)
        lower_frame.pack(fill=Tk.BOTH, expand=1)

        tframe = Tk.Frame(lower_frame)
        tframe.pack(side=Tk.LEFT, padx=20)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        btn_frame = Tk.Frame(lower_frame)
        btn_frame.pack(side=Tk.RIGHT, padx=20)
        Tk.Button(btn_frame, text="Close", width=10, command=self.cancel).pack()

    def cancel(self):
        plt.close(self.fig)
        Dialog.cancel(self)
