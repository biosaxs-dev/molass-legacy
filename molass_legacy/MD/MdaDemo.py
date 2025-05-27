"""
    MdaDemo.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import os
import numpy as np
import glob
from shutil import move
import MDAnalysis as mda
from MDAnalysisTests.datafiles import PSF, DCD
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
from molass_legacy._MOLASS.SerialSettings import get_setting
import matplotlib.pyplot as plt
# import molass_legacy.KekLib.DebugPlot as plt

def make_folders():
    md_folder = os.path.join(get_setting('analysis_folder'), 'MD')
    pdb_folder = os.path.join(md_folder, 'pdb')
    mrc_folder = os.path.join(md_folder, 'mrc')
    sas_folder = os.path.join(md_folder, 'sas')
    for folder in [pdb_folder, mrc_folder, sas_folder]:
        mkdirs_with_retry(folder)
    return pdb_folder, mrc_folder, sas_folder

def make_pdb_files():
    pdb_folder = make_folders()[0]

    num_files = len(os.listdir(pdb_folder))
    print(num_files)
    if num_files > 80:
        return

    u = mda.Universe(PSF, DCD)
    protein = u.select_atoms('protein')

    Rgyr = []
    for k, ts in enumerate(u.trajectory):
        Rgyr.append((u.trajectory.time, protein.radius_of_gyration()))
        file = os.path.join(pdb_folder, '%03d.pdb' % k)
        print('writing', file)
        protein.write(file)
    Rgyr = np.array(Rgyr)

    import molass_legacy.KekLib.DebugPlot as dplt
    fig, ax = dplt.subplots()
    ax.plot(Rgyr[:,0], Rgyr[:,1], 'r--', lw=2, label=r"$R_G$")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel(r"radius of gyration $R_G$ ($\AA$)")
    # ax.figure.savefig("Rgyr.pdf")
    dplt.show()

def make_mrc_files():
    from molass_legacy.DENSS.DenssUtils import run_pdb2mrc

    pdb_folder, mrc_folder = make_folders()[0:2]
    num_files = len(os.listdir(mrc_folder))
    print(num_files)
    if num_files > 80:
        return

    for path in glob.glob(pdb_folder + r'\*.pdb'):
        mrc_file = run_pdb2mrc(path)
        print(mrc_file)
        move(mrc_file, mrc_folder)

def make_scattering_curves_from_mrc():
    import mrcfile
    from molass_legacy.Saxs.ReciprocalData import ReciprocalData

    mrc_folder, sas_folder = make_folders()[1:3]
    num_files = len(os.listdir(sas_folder))
    print(num_files)
    if num_files > 80:
        return

    q = np.linspace(0, 0.5, 500)
    for path in glob.glob(mrc_folder + r'\*.mrc'):
        print(path)
        with mrcfile.open(path) as mrc:
            data = mrc.data
        rdata = ReciprocalData(data.shape)
        F = rdata.get_reciprocal(data)
        curve = rdata.get_scattering_curve(q, F)
        error = curve*0.01
        folder, file = os.path.split(path)
        out_file = os.path.join(sas_folder, file.replace('_pdb.mrc', '.dat'))
        np.savetxt(out_file, np.array([q, curve, error]).T)

def make_scattering_curves_from_pdb():
    from SerialAtsasTools import CrysolExecutor

    pdb_folder, _, sas_folder = make_folders()
    num_files = len(os.listdir(sas_folder))
    print(num_files)
    if num_files > 80:
        return

    sas_denss_folder = sas_folder.replace('sas', 'sas-denss')

    crysol = CrysolExecutor()

    q = np.linspace(0, 0.5, 500)
    for path in glob.glob(pdb_folder + r'\*.pdb'):
        if path.find('_centered.pdb') > 0:
            continue
        print(path)
        folder, file = os.path.split(path)
        exp_file = os.path.join(sas_denss_folder, file.replace('.pdb', '.dat'))
        print('crysol.execute')
        crysol.execute(path, exp_file)
        crysol.move(path, sas_folder)

import matplotlib.animation as animation
from molass_legacy.KekLib.OurTkinter import Tk, Dialog

class SasAnimation(Dialog):
    def __init__(self, parent, sas_files):
        self.sas_files = sas_files
        Dialog.__init__(self, parent, "SasAnimation", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        self.fig = fig = plt.figure()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
        self.toolbar.update()
        self.make_animation(fig)
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    def make_animation(self, fig):
        from molass_legacy.KekLib.NumpyUtils import np_loadtxt
        sas_files = self.sas_files
        num_frames = len(sas_files)

        ax = fig.gca()

        ax.set_title("Scattering Curve Variation corresponding to Trajectory Variation", fontsize=16)
        ax.set_yscale('log')

        def get_data(k):
            path = sas_files[k]
            print(path)
            # data = np.loadtxt(path)
            data, _ = np_loadtxt(path)
            return data

        data = get_data(0)
        qv = data[:,0]
        max_intensity = data[0,1]

        line_init, = ax.plot(data[:,0], data[:,1], ':')
        line, = ax.plot(data[:,0], data[:,1])

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin/2, ymax*2)

        tx = (xmin+xmax)/2
        ty = np.sqrt(ymin*ymax)
        text = ax.text(tx, ty, str(0), fontsize=50, alpha=0.1, ha='center', va='center')

        def init():
            # initialize an empty list of cirlces
            return [line_init, line, text]

        def animate(i):
            data = get_data(i)
            y = data[:,1]*max_intensity/data[0,1]
            line.set_data(qv, y)
            text.set_text(str(i))
            return [line_init, line, text]

        fig.tight_layout()

        self.anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=num_frames, interval=200, blit=True)
        # plt.show()

def make_animation(parent):
    sas_folder = make_folders()[2]
    sas_files = glob.glob(sas_folder + r'\*.dat')
    if len(sas_files) == 0:
        sas_files = glob.glob(sas_folder + r'\*.int')

    sas = SasAnimation(parent, sas_files)
    sas.show()

def demo(parent=None, use_denss=False):
    make_pdb_files()
    if use_denss:
        make_mrc_files()
        make_scattering_curves_from_mrc()
        make_animation(parent)
    else:
        make_scattering_curves_from_pdb()
        make_animation(parent)
