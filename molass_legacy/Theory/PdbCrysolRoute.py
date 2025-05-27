"""
    Theory.PdbCrysolRoute.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from glob import glob
from molass_legacy.KekLib.BasicUtils import Struct
from DataUtils import get_in_folder
from CrysolUtils import np_loadtxt_crysol
from molass_legacy.ATSAS.Crysol import get_info_from_crysol_log, get_info_from_crysol_fit_log
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkCustomWidgets import FileEntry, FolderEntry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.DENSS.DenssUtils import fit_data
from molass_legacy._MOLASS.Version import is_developing_version
from molass_legacy.Saxs.SaxsCurveUtils import percentile_normalize

drive = __file__[0:2]
crysol_folder = drive + r"\PyTools\Data\CRYSOL"
crysol_int_file = os.path.join(crysol_folder, "1wbh-ba1.int")

def demo(dialog):
    print("PdbCrysolRoute demo")

TEST_FOLDERS = [
    drive + r"\TODO\20231031\FIND\crysol_folders\2eph-ba1",
    # drive + r"\TODO\20231031\FIND\crysol_folders\6ald-ba1",
    # drive + r"\TODO\20231031\FIND\crysol_folders\6v20-ba1",
    drive + r"\TODO\20231031\FIND\crysol_folders\5f4x-ba1",
    ]

class CrysolResultsEntry(Dialog):
    def __init__(self, parent, num_files):
        self.num_files = num_files
        self.applied = False
        Dialog.__init__(self, parent, "Crysol Results Entry", visible=False, location="lower right")

    def show(self):
        self._show()

    def body(self, body_frame):

        entry_frame = Tk.Frame(body_frame)
        entry_frame.pack()

        row = 0

        text_vars = []
        for i in range(self.num_files):
            label = Tk.Label(entry_frame, text="CRYSOL Output Folder %s:" % str([i+1]))
            label.grid(row=row+i, column=0, padx=5, sticky=Tk.E)
            var = Tk.StringVar()
            if is_developing_version():
                var.set(TEST_FOLDERS[i])
            text_vars.append(var)
            entry = FolderEntry(entry_frame, textvariable=var, width=60)
            entry.grid(row=row+i, column=1, padx=5, sticky=Tk.W)

        self.text_vars = text_vars

    def get_file_paths(self):
        ret_paths = []
        for var in self.text_vars:
            ret_paths.append(var.get())
        return ret_paths

    def validate(self):
        num_paths = 0
        for var in self.text_vars:
            val = var.get()
            if val is None or val == "":
                pass
            else:
                num_paths += 1
        if num_paths == 0:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            MessageBox.showerror('No entry',
                            "You should specify a file for at least one component.\n",
                            parent=self)
            return 0
        else:
            return 1

    def apply(self):
        self.applied = True

class PdbCrysolRouteDialog(Dialog):
    def __init__(self, parent, view_dialog, crysol_folders):
        self.view_dialog = view_dialog
        self.crysol_folders = crysol_folders
        Dialog.__init__(self, parent, "Pdb Crysol Route", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        to_draw_components = []
        to_draw_folders = []
        for k, path in enumerate(self.crysol_folders):
            if path is None or path == "":
                pass
            else:
                to_draw_components.append(k)
                to_draw_folders.append(path)

        num_components = len(to_draw_components)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        fig, axes = plt.subplots(nrows=num_components, ncols=3, figsize=(18,4*num_components))
        self.draw_curves(fig, axes, to_draw_components, to_draw_folders)
        self.fig = fig

        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.draw()

        tframe = Tk.Frame(body_frame)
        tframe.pack(side=Tk.LEFT, padx=20)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe )
        self.toolbar.update()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=50, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def get_info(self, folder):
        int_file = None
        short_name_len = None
        Rg = None
        chi_square = None
        for node in os.listdir(folder):
            if short_name_len is None:
                short_name_len = len(node)
            path = os.path.join(folder, node)
            ext = node[-4:]
            if ext == ".int":
                int_file = path
            elif ext == ".log":
                if len(node) == short_name_len:
                    info = get_info_from_crysol_log(path)
                    Rg = info.Rg
                else:
                    info = get_info_from_crysol_fit_log(path)
                    Chi_square = info.Chi_square
        return Struct(int_file=int_file, Rg=Rg, Chi_square=Chi_square)

    def draw_curves(self, fig, axes, to_draw_components, to_draw_folders):

        to_draw_infos = [self.get_info(f)  for f in to_draw_folders]

        in_folder = get_in_folder()

        view_dialog = self.view_dialog

        lrf_info = view_dialog.lrf_info
        x = lrf_info.x
        y = lrf_info.y
        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices

        zero_base = np.zeros(len(x))

        optimizer = view_dialog.optimizer
        qv = optimizer.qvector
        xrD = optimizer.xrD
        xrE = optimizer.xrE
        xr_i = optimizer.xr_index

        cy_list = []
        peak_positions = []
        my_curves = []
        py_curves = []
        py_ift_list = []
        for k, cy in enumerate(Cxr):
            m = np.argmax(cy)
            peak_positions.append(m)
            my = xrD[:,m]
            my_curves.append(my)
            scale = Pxr[xr_i,k]
            cy_list.append(scale*cy)
            if k in to_draw_components:
                py = Pxr[:,k]
                py_curves.append(py)
                sasrec = fit_data(qv,py, xrE[:,m], return_sasrec=True)
                py_ift_list.append((sasrec.qc, sasrec.Ic, sasrec.Icerr, sasrec.rg))
        ty = np.sum(cy_list, axis=0)

        fig.suptitle("Comparison to PDB-generated Curves on %s" % (in_folder), fontsize=20)

        if len(axes) == 3:
            ax1, ax2, ax3 = axes
            axes_ = [axes]
        else:
            ax1, ax2, ax3 = axes[0,:]
            axes_ = axes
        ax1.set_title("Component Elution", fontsize=16)
        ax2.set_title("Peak Top, LRF and IFT", fontsize=16)
        ax3.set_title("IFT and CRYSOL", fontsize=16)

        for k, (ax1, ax2, ax3) in enumerate(axes_):
            info = to_draw_infos[k]
            path = info.int_file
            data = np_loadtxt_crysol(path)[0]
            dk = to_draw_components[k]

            ax1.plot(x, y)
            for j, cy_ in enumerate(cy_list):
                if j == dk:
                    ax1.fill_between(x, zero_base, cy_, fc="cyan", alpha=0.2)
                ax1.plot(x, cy_, ":")
            ax1.plot(x, ty, ":", color="red")

            ax1.set_xlabel("Eno")
            ax1.set_ylabel("Intensity")

            for ax in (ax2, ax3):
                ax.set_yscale('log')
                ax.set_xlabel("Q")
                ax.set_ylabel("$Log_{10}(Intensity)$")

            ax2.plot(qv, percentile_normalize(my_curves[k]), label="peak top")
            ax2.plot(qv, percentile_normalize(py_curves[k]), label="LRF")
            qc, Ic, Icerr, rg = py_ift_list[k]
            nIc = percentile_normalize(Ic)

            for ax in [ax2, ax3]:
                ax.plot(qc, nIc, label="LRF IFT; $R_g$=%.1f" % rg, color="C2")

            qv_ = data[:,0]
            py_ = percentile_normalize(data[:,1])
            file = path.replace('/', '\\').split('\\')[-1].replace('.int', '')
            file_ = file[:4].upper() + file[4:].lower()
            ax3.plot(qv_, py_, label=r"%s; $R_g$=%.1f, $\chi^2$=%.3g" % (file_, info.Rg, info.Chi_square), color="C3")

            for ax in [ax2, ax3]:
                ax.legend()

            crysol_spline = UnivariateSpline(qv_, py_, s=0, ext=3)
            residual = (nIc - crysol_spline(qc))/Icerr
            axt = ax3.twinx()
            axt.grid(False)
            axt.bar(qc, residual, width=0.002, color='purple', alpha=0.1, label='residual (diff/error)')
            ymin, ymax = axt.get_ylim()
            maxabs = max(abs(ymin), abs(ymax))
            axt.set_ylim(-maxabs, maxabs)
            axt.legend(bbox_to_anchor=(1, 0.4), loc='upper right')

        fig.tight_layout()


def compare_to_pdb_impl(view_dialog):
    print("PdbCrysolRoute demo")
    entry = CrysolResultsEntry(view_dialog, 2)
    entry.show()
    if not entry.applied:
        return

    crysol_folders = entry.get_file_paths()

    dialog = PdbCrysolRouteDialog(view_dialog.parent, view_dialog, crysol_folders)
    dialog.show()
