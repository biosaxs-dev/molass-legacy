"""
    GradualRegionAnalysis.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import re
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from ScrolledFrame import ScrolledFrame
from DataUtils import get_in_folder
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy._MOLASS.SerialSettings import get_setting

class GradualRegionAnalysis(Dialog):
    def __init__(self, parent, dialog, sd, trimming_info, pre_recog, debug=False):
        self.logger = logging.getLogger(__name__)
        print("pre_recog=", pre_recog)

        D, E, qv, xr_ecurve = sd.get_xr_data_separate_ly()

        self.qv = qv
        i_smp = sd.xray_index
        eslice, m = self.determine_analysis_range(xr_ecurve)
        self.i_smp = i_smp
        self.y = D[:,m]
        cv = pre_recog.cs.get_uniformly_mapped_a_curve()

        if debug:
            import molass_legacy.KekLib.DebugPlot as dplt
            x = xr_ecurve.x
            y = xr_ecurve.y
            with dplt.Dp():
                fig, ax = dplt.subplots()
                ax.set_title("cv debug")
                ax.plot(x, y)
                axt = ax.twinx()
                axt.grid(False)
                axt.plot(x, cv)
                fig.tight_layout()
                dplt.show()

        self.eslice = eslice
        self.aslice = trimming_info.get_slice()
        self.info_list = self.compute_info_with_changed_trimming(D, E, qv, i_smp, eslice, self.aslice, cv)

        Dialog.__init__(self, parent, "Gradual Region Analysis", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        in_folder = get_in_folder()
        suptitle = Tk.Label(body_frame, text="Guinier Analysis Results with gradually changed trimming on %s" % in_folder, font=('', 18))
        suptitle.pack(pady=10)

        cframe1 = Tk.Frame(body_frame)
        cframe1.pack(side=Tk.LEFT)
        space_width = 20
        space = Tk.Frame(body_frame, width=space_width)
        space.pack(side=Tk.LEFT)

        cframe2 = Tk.Frame(body_frame)
        cframe2.pack(side=Tk.LEFT, fill=Tk.X, expand=1)

        cframe21 = Tk.Frame(cframe2, bg="white")
        cframe21.pack(fill=Tk.X, expand=1)
        cframe22 = Tk.Frame(cframe2)
        cframe22.pack()
        cframe2_ = ScrolledFrame(cframe22, bg="white")
        cframe2_.pack()
        self.scrolled_frame = cframe2_

        fig1, ax1 = plt.subplots(figsize=(6,8))
        self.fig1 = fig1
        self.ax1 = ax1

        qv = self.qv
        n = len(self.info_list)

        data_title = "" if get_setting('test_pattern') is None else " in " + in_folder

        ax1.set_title("$R_g$ Changes" + data_title, fontsize=16)
        ax1.set_ylabel("$R_g$")
        ax1.set_xlabel("trimming start Q")
        rg_list = [rec[1].Rg for rec in self.info_list]
        ax1.plot(qv[0:n], rg_list, "-o")

        ymin, ymax = ax1.get_ylim()
        ax1.set_ylim(ymin, ymax)
        tq = self.qv[self.aslice.start]
        ax1.plot([tq, tq], [ymin, ymax], color="yellow")

        fig1.tight_layout()

        self.mpl_canvas1 = FigureCanvasTkAgg(fig1, cframe1)
        self.mpl_canvas_widget1 = self.mpl_canvas1.get_tk_widget()
        self.mpl_canvas_widget1.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas1.mpl_connect('button_press_event', self.on_click)

        label = Tk.Label(cframe21, text="Guinier Plots", font=('Arial', 14), bg="white")
        label.pack(fill=Tk.X, expand=1, pady=5)

        nrows = len(self.info_list)
        height = 3*nrows
        fig2 = plt.figure(figsize=(11,height))
        self.fig2 = fig2

        qv2 = self.qv**2
        gstop = np.max([rec[1].guinier_stop for rec in self.info_list])

        gs = GridSpec(nrows, 5)
        for i, (pv, sg) in enumerate(self.info_list):
            ax0 = fig2.add_subplot(gs[i,0])
            ax0.set_xticks([])
            ax0.set_yticks([])
            ax0.set_facecolor("white")
            ax0.text(0.5, 0.5, "Q[%d]\n%.3g\n\n$R_q$=%.3g" % (i, qv[i], sg.Rg), fontsize=16, ha="center", va="center", alpha=0.5)

            ax1 = fig2.add_subplot(gs[i,1:])
            ax1.set_xlabel("$Q^2$")
            ax1.set_ylabel("$ln(I)$")

            x2 = qv2[i:gstop]
            if i == 0:
                scaled_y = self.y[:gstop] * pv[self.i_smp]/self.y[self.i_smp]
            ax1.plot(qv2[:gstop], np.log(scaled_y), "o", markersize=3, label="measured raw data")
            ax1.plot(x2, np.log(pv[:len(x2)]), "o", color="C1", markersize=3, label="LRF-corrected data")
            ax1.plot(sg.guinier_x, sg.guinier_y, color="red", alpha=0.5, label="estimated guinier interval")
            ax1.legend()

            if i == 0:
                xmin, xmax = ax1.get_xlim()
            else:
                ax1.set_xlim(xmin, xmax)

        fig2.tight_layout()
        self.mpl_canvas2 = FigureCanvasTkAgg(fig2, cframe2_.interior)
        self.mpl_canvas_widget2 = self.mpl_canvas2.get_tk_widget()
        self.mpl_canvas_widget2.pack(fill=Tk.BOTH, expand=1)

        self.set_geometry(space_width)
        self.update()

        self.buttonbox(build=True)

    def buttonbox(self, build=False):
        if not build:
            # temporary bug fix
            # ignore this standared callback
            return

        box = Tk.Frame(self)
        box.pack(side=Tk.BOTTOM)

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def set_geometry(self, space_width):
        self.update()
        canvas_width1 = int(self.mpl_canvas_widget1.cget('width'))
        canvas_width2 = int(self.mpl_canvas_widget2.cget('width'))

        geometry = self.geometry()
        print("geometry=", geometry, canvas_width1, canvas_width2)
        width = canvas_width1 + space_width + canvas_width2 + 40
        height = int(self.mpl_canvas_widget1.cget('height')) + 60

        new_geometry = re.sub( r'(\d+x\d+)(.+)', lambda m: "%dx%d" % (width, height) + m.group(2), geometry)
        print("new_geometry=", new_geometry)
        self.geometry(new_geometry)

    def determine_analysis_range(self, xr_ecurve):
        eranges = xr_ecurve.get_default_editor_ranges()
        print("eranges=", eranges)
        pno = xr_ecurve.primary_peak_no
        range_ = eranges[pno]
        f = range_[0][0]
        m = range_[0][1]
        t = range_[-1][-1]
        return slice(f, t+1), m

    def compute_info_with_changed_trimming(self, D, E, qv, i_smp, eslice, aslice, cv):
        cv_ = cv[eslice]
        C = np.array([cv_, cv_**2])
        Cinv = np.linalg.pinv(C)

        info_list = []
        for i in range(i_smp):
            aslice_ = slice(i, aslice.stop)
            qv_ = qv[aslice_]
            D_ = D[aslice_,eslice]
            E_ = E[aslice_,eslice]
            P = D_@Cinv
            Dinv = np.linalg.pinv(D_)
            W   = np.dot(Dinv, P)
            Pe  = np.sqrt(np.dot(E_**2, W**2))
            data = np.array([qv_, P[:,0], Pe[:,0]]).T
            sg = SimpleGuinier(data)
            # print([i], sg.Rg)
            info_list.append((P[:,0], sg))

        return info_list

    def on_click(self, event):
        if event.inaxes != self.ax1:
            return

        q = event.xdata
        qv_ = self.qv[:self.i_smp]
        k = np.argmin((qv_ - q)**2)
        self.scrolled_frame.canvas.yview_moveto(k/len(self.info_list))

    def cancel(self):
        # overiding cancel to cleanup self.fig1 and self.fig2
        # because the call to the destructor __del__ seems to be delayed
        plt.close(self.fig1)
        plt.close(self.fig2)
        Dialog.cancel(self)

def survey_summary(survey_csv):
    from scipy.stats import linregress
    data_list = []
    with open(survey_csv) as fh:
        for line in fh:
            fields = line[:-1].split(',')
            values = np.array([float(v) for v in fields[2:]])
            n = int(values[0])
            qv = values[1:1+n]
            rgv = values[1+n:1+2*n]
            # print(qv)
            # print(rgv)
            slope, intercept, r_value, p_value, std_err = linregress(qv, rgv)
            if std_err > 100:
                print("skipping", fields[1], r_value, std_err)
                continue
            print(fields[1], r_value, std_err)
            data_list.append((r_value, std_err))
    data = np.array(data_list)

    fig, ax = plt.subplots()
    ax.set_title("Scatter Plot of Rg Variations in Gradual Region Analysis", fontsize=16)
    ax.plot(data[:,0], data[:,1], "o", markersize=3)

    ax.set_xlim(-1, 1)

    ax.set_xlabel("r_value", fontsize=16)
    ax.set_ylabel("std_error", fontsize=16)
    fig.tight_layout()
    plt.show()
