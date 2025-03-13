"""
    SSDC.SsdcAnalysis.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

class SsdcAnalysis(Dialog):
    def __init__(self, parent, info_list, num_components):
        self.parent = parent
        self.info_list = info_list
        self.num_components = num_components
        Dialog.__init__(self, parent, title="SSDC Analysis", visible=False)

    def show(self):
        self._show()
    
    def body(self, body_frame):
        cframe = Tk.Frame(body_frame)
        cframe.pack()
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        self.draw(fig, axes)
        fig.tight_layout()
        self.mpl_canvas = FigureCanvasTkAgg(fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
        self.toolbar.update()
    
    def draw(self, fig, axes):
        from importlib import reload
        import SSDC.MappingDecomposer
        reload(SSDC.MappingDecomposer)
        from SSDC.MappingDecomposer import decompose_paired_elution_curves, compute_monopore_curves

        moments_list = []
        for in_folder, lrf_src in self.info_list:
            moments_list.append(lrf_src.get_egh_moments_list())

        temp_fix_slice = slice(None, None)
        if in_folder.find("20230706/ALD_OA") >= 0:
            temp_fix_slice = slice(1, None)

        for ax, (in_folder, lrf_src), moments in zip(axes[:,0], self.info_list, moments_list):
            in_folder = get_in_folder(in_folder)
            ax.set_title("Naive Decomposition of %s" % in_folder, fontsize=16)
            model = lrf_src.model
            x = lrf_src.xr_x
            y = lrf_src.xr_y
            ax.plot(x, y, color="orange")
            for params in lrf_src.get_peaks()[temp_fix_slice]:
                cy = model(x, params)
                ax.plot(x, cy, ":")
            for M in moments[temp_fix_slice]:
                ax.axvline(x=M[0], color="blue")

        ret = decompose_paired_elution_curves(self.info_list, moments_list, self.num_components)
        if ret is None:
            return
        mnp_params, rgs, mapping, scale = ret
        print("mapping=", mapping)
        print("rgs=", rgs)

        for k, (ax, (in_folder, lrf_src), moments) in enumerate(zip(axes[:,1], self.info_list, moments_list)):
            in_folder = get_in_folder(in_folder)
            ax.set_title("Mapped Decomposition of %s" % in_folder, fontsize=16)
            model = lrf_src.model
            x = lrf_src.xr_x
            y = lrf_src.xr_y
            if k == 0:
                ax.plot(x, y, color="orange")
                cy_list, tty = compute_monopore_curves(x, mnp_params, rgs)
                peakpos = []
                for cy in cy_list:
                    ax.plot(x, cy, ":")
                    peakpos.append(x[np.argmax(cy)])
                peakpos = np.array(peakpos)
                x_ = x
                last_x = x
                times_scale = 1
                intensity_scale = 1
                volume_scale = 1
            else:
                spline = UnivariateSpline(x, y, s=0, ext=3)
                a, b  = mapping
                x_ = a*last_x + b
                y_ = spline(x_)
                ax.plot(x_, y_, color="orange")
                for cy in cy_list:
                    ax.plot(x_, scale*cy, ":")
                peakpos = a*peakpos + b
                times_scale = a
                intensity_scale = scale
                volume_scale = a*scale
            for px in peakpos:
                ax.axvline(x=px, color="blue")

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            for k, (name, value) in enumerate([("Time Scale", times_scale), ("Intensity Scale", intensity_scale), ("Volume Scale", volume_scale)]):
                tx = xmin*0.95 + xmax*0.05
                w = 0.8 - 0.13*k
                ty = ymin*(1 - w)+ ymax*w
                ax.text(tx, ty, "%s = %.3g" % (name, value), alpha=0.3, fontsize=30)