"""
    QuickAnalysis.ModeledPeaksTester.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy._MOLASS.SerialSettings import get_setting

class TestDialog(Dialog):
    def __init__(self, parent, sd):
        self.sd = sd
        Dialog.__init__(self, parent, "", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        sbox_frame = Tk.Frame(body_frame)
        sbox_frame.pack()

        label = Tk.Label(sbox_frame, text="Num Peaks")
        label.pack(side=Tk.LEFT)

        xr_curve = self.sd.get_xray_curve()
        n = len(xr_curve.peak_info)

        self.num_peaks = Tk.IntVar()
        self.num_peaks.set(n+1)
        sbox  = Tk.Spinbox(sbox_frame, textvariable=self.num_peaks,
                          from_=1, to=7, increment=1,
                          justify=Tk.CENTER, width=6)
        sbox.pack(side=Tk.LEFT)

        self.debug = Tk.IntVar()
        debug_cb = Tk.Checkbutton(body_frame, text="debug", variable=self.debug)
        debug_cb.pack()
        run_btn = Tk.Button(body_frame, text="run", command=self.run)
        run_btn.pack()

    def run(self):
        from importlib import reload
        import QuickAnalysis.ModeledPeaks
        reload(QuickAnalysis.ModeledPeaks)
        from molass_legacy.QuickAnalysis.ModeledPeaks import get_curve_xy_impl, get_modeled_peaks_impl

        unified_baseline_type = get_setting("unified_baseline_type")
        print("unified_baseline_type=", unified_baseline_type)
        uv_x, uv_y, xr_x, xr_y, details = get_curve_xy_impl(self.sd, baseline_type=unified_baseline_type, return_details=True)
        num_peaks = self.num_peaks.get()
        debug = self.debug.get()
        a, b = self.sd.pre_recog.mapped_info[0]
        uv_peaks, xr_peaks = get_modeled_peaks_impl(a, b, uv_x, uv_y, xr_x, xr_y, num_peaks=num_peaks, exact_num_peaks=num_peaks, debug=debug)
        plot_modeled_peaks(uv_x, uv_y, xr_x, xr_y, uv_peaks, xr_peaks, a, b, details=details)

def plot_modeled_peaks(uv_x, uv_y, xr_x, xr_y, uv_peaks, xr_peaks, a, b, details=None, suptitle="unspecified title"):
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.Peaks.ElutionModels import egh

    with plt.Dp():
        fig, axes = plt.subplots(ncols=3, figsize=(18,5))
        fig.suptitle(suptitle)
        ax1, ax2, ax3 = axes
        ax1.set_title("UV Elution")
        ax2.set_title("Xray Elution")
        ax3.set_title("UV/Xray Mapping")

        ax1.plot(uv_x, uv_y, color="blue")
        uv_ty = np.zeros(len(uv_y))
        for h, m, s, t in uv_peaks:
            uv_cy = egh(uv_x, h, m, s, t)
            uv_ty += uv_cy
            ax1.plot(uv_x, uv_cy, ":")
            ax1.plot(m, h, "o", color="yellow")
        ax1.plot(uv_x, uv_ty, ":", color="red")

        ax2.plot(xr_x, xr_y, color="orange")
        xr_ty = np.zeros(len(xr_y))
        for h, m, s, t in xr_peaks:
            xr_cy = egh(xr_x, h, m, s, t)
            xr_ty += xr_cy
            ax2.plot(xr_x, xr_cy, ":")
            ax2.plot(m, h, "o", color="yellow")
        ax2.plot(xr_x, xr_ty, ":", color="red")

        ax3.plot(uv_x, uv_y, color="blue")
        axt = ax3.twinx()
        axt.grid(False)
        axt.plot(xr_x*a + b, xr_y, color="orange")

        if details is not None:
            uv_by, xr_by = details.baselines
            ax1.plot(uv_x, uv_by, color="red")
            ax2.plot(xr_x, xr_by, color="red")

        fig.tight_layout()
        plt.show()
