# coding: utf-8
"""
    ExtrapolationProof.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import re
import copy
import numpy as np
from OurStatsModels import WLS, add_constant
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from ScrolledFrame import ScrolledFrame
import molass_legacy.KekLib.DebugPlot as plt
from SerialTestUtils import get_tttt_data_objects, debug_plot_runner
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from SvdDenoise import get_denoised_data
from SmoothFactorizer import SmoothFactorizer
from SerialTestUtils import get_tk_root
from DataUtils import get_pytools_folder, serial_folder_walk, cut_upper_folders
from SerialDataUtils import get_mtd_filename
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
from SerialAtsasTools import AlmergeExecutor

class ZxSolver:
    def plot_func(self):
        q   = self.q
        P   = self.P
        AB  = self.AB
        ABe = self.ABe

        fig = plt.figure()
        ax2 = fig.gca()
        ax2t = ax2.twinx()

        ax2.plot(q, np.log10(P[:,0]), label='A-matrix')
        ax2t.plot(q, P[:,1], label='B-matrix')
        # ax2.plot(q, np.log10(e[:,0]), label='A-matrix error')
        ax2.plot(q, np.log10(AB[:,0]), label='A-iter')
        # ax2.plot(q, ABe[:,0]), label='A-iter error')
        ax2t.plot(q, AB[:,1], label='B-iter')
        fig.tight_layout()
        plt.show()

    def plot(self):
        debug_plot_runner(self.plot_func)

    def plot_A(self, ax, label):
        q   = self.q
        P   = self.P
        ax.plot(q, np.log10(P[:,0]), label=label)

    def plot_B(self, ax, label):
        q   = self.q
        P   = self.P
        ax.plot(q, P[:,1], label=label)

class MpInverse(ZxSolver):
    def __init__(self, q, M, E, c):
        self.q  = q
        self.M  = M
        self.E  = E
        self.c  = c

    def solve(self, with_const=False):
        c_ = self.c
        c_list = [c_, c_**2]
        if with_const:
            c_list.append( np.ones(len(c_)) )

        self.C = np.array( c_list )
        self.Cpinv = np.linalg.pinv(self.C)
        self.P = np.dot(self.M, self.Cpinv)
        self.e = np.sqrt( np.dot(self.E**2, self.Cpinv**2) )
        print( self.P.shape, self.e.shape )

class IteratedWLS(ZxSolver):
    def __init__(self, q, M, E, c):
        self.q  = q
        self.M  = M
        self.E  = E
        self.c  = c
        c_ = c
        self.C = np.array( [c_, c_**2] )

    def solve(self, weighted=True, with_const=False):
        AB_list = []
        ab_error_list = []
        for i in range(self.M.shape[0]):
        # for i in [10]:
            Mi = np.array( [ self.M[i,:] ] )
            # print(M.shape, C.shape)
            if True:
                e_ = np.array( [ self.E[i,:] ] ).flatten()
                e2 = e_**2
                if weighted:
                    w = 1/e2
                else:
                    w = None
                x = Mi.flatten()
                if with_const:
                    X = add_constant(x)
                else:
                    X = x
                model = WLS(x, self.C.T, weights=w)
                result  = model.fit()
                Pi = result.params
                ab_error = np.sqrt( np.dot( model.XtWX_inv_XtW**2, e2 ) )
                ab_error_list.append( ab_error )
            else:
                Pi = np.dot(Mi, self.Cpinv)
            # print( Pi )
            AB_list.append( Pi.flatten() )

        self.P = np.array(AB_list)
        self.e = np.array(ab_error_list)
        print( self.P.shape, self.e.shape )

class NormMinizerSmoothed(ZxSolver):
    def __init__(self, q, M, E, c):
        self.q  = q
        self.M  = M
        self.E  = E
        self.c  = c

    def solve(self, apositive=False, smoothed=True):
        factorizer = SmoothFactorizer(self.M, self.E, self.c)
        factorizer.solve(apositive=apositive, smoothed=smoothed)
        self.P = factorizer.P

class AlMergeSolver:
    def __init__(self, q, M, E, c):
        self.q  = q
        self.M  = M
        self.E  = E
        self.c  = c
        self.temp_folder = get_setting('temp_folder') + '/almerge'
        clear_dirs_with_retry( [self.temp_folder] )

    def solve(self):
        # write files into temp_folder
        files = []
        for j in range(self.M.shape[1]):
            path = self.temp_folder + "/ALMERGE_IN_%03d.dat" % j
            files.append(path)
            data = np.array([self.q, self.M[:,j], self.E[:,j]]).T
            np.savetxt(path, data)

        executor = AlmergeExecutor()
        indeces = np.arange(len(files))
        out_file =  self.temp_folder + '/almerge_out.dat'
        result = executor.execute(self.c, files, indeces, out_file)
        self.A = result.exz_array[:,1]
        self.P = self.A.reshape( (self.A.shape[0], 1) )

    def scale(self, other95):
        this95 = np.percentile(self.A, 95)
        self.A = self.A * (other95/this95)
        self.P = self.A.reshape( (self.A.shape[0], 1) )

    def copy(self):
        clone = AlMergeSolver(self.q, self.M, self.E, self.c)
        clone.A = copy.deepcopy(self.A)
        clone.P = copy.deepcopy(self.P)
        return clone

    def plot_A(self, ax, label):
        q   = self.q
        ax.plot(q, np.log10(self.A), label=label)

    def plot_B(self, ax, label):
        pass

def create_plot_specs(in_folder, uv_folder, range_callback, range_=None):

    _, sd, mapper = get_tttt_data_objects(in_folder, uv_folder, analysis_copy=True, adjust=True )
    c_all = mapper.get_conc_vector( 5 )
    sd.apply_baseline_correction( mapper.get_mapped_info() )

    spec_list = []

    if range_ is None:
        for pno, range_ in enumerate(mapper.get_int_ranges()):
            print(range_)
            lower, middle, upper = range_
            for ad, ft in enumerate([ [lower, middle], [middle, upper] ]):
                f, t = ft
                spec = range_callback(in_folder, sd, pno, ad, (f, t), c_all)
                spec_list.append(spec)
    else:
        pno, ad, range__ = range_
        spec = do_a_range(in_folder, sd, pno, ad, range__, c_all)
        spec_list.append(spec)

    return spec_list

def do_a_range(in_folder, sd, pno, ad, range_, c_all):
    eslice = slice(*range_)

    q   = sd.intensity_array[0,:,0]
    D   = sd.intensity_array[eslice,:,1].T
    E   = sd.intensity_array[eslice,:,2].T
    c = c_all[eslice]

    solver0 = MpInverse(q, D, E, c)
    solver0.solve()
    solver1 = MpInverse(q, D, E, c)
    solver1.solve(with_const=True)
    # solver2 = IteratedWLS(q, D, E, c)
    # solver2.solve(weighted=False)
    solver3 = IteratedWLS(q, D, E, c)
    solver3.solve(weighted=True)
    solver4 = NormMinizerSmoothed(q, D, E, c)
    solver4.solve(smoothed=False)
    solver5 = NormMinizerSmoothed(q, D, E, c)
    solver5.solve(smoothed=False)
    solver6 = NormMinizerSmoothed(q, D, E, c)
    solver6.solve(smoothed=False, apositive=True)

    data_name = in_folder + "/pk%d_%s_*.dat" % (pno+1, 'asc' if ad == 0 else 'dsc')
    return data_name, [
            # [ [ "MP-inverse",  solver0], [ "iterated-OLS", solver2 ] ],
            [ [ "MP-inverse",  solver0], [ "iterated-WLS", solver3 ] ],
            [ [ "MP-inverse",  solver0], [ "MP-inverse-with-1",  solver1] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min", solver4 ] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min-smooth", solver5] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min-A_positive", solver6] ],
            ]

def do_a_range_denoise(in_folder, sd, pno, ad, range_, c_all):
    eslice = slice(*range_)

    q   = sd.intensity_array[0,:,0]
    D   = sd.intensity_array[eslice,:,1].T
    D2_  = get_denoised_data(D, rank=2)
    D3_  = get_denoised_data(D, rank=3)
    D4_  = get_denoised_data(D, rank=4)
    D5_  = get_denoised_data(D, rank=5)

    E   = sd.intensity_array[eslice,:,2].T
    c = c_all[eslice]

    solver0 = MpInverse(q, D, E, c)
    solver0.solve()
    solver1 = IteratedWLS(q, D, E, c)
    solver1.solve(weighted=False)
    solver5 = MpInverse(q, D5_, E, c)
    solver5.solve()
    solver4 = MpInverse(q, D4_, E, c)
    solver4.solve()
    solver3 = MpInverse(q, D3_, E, c)
    solver3.solve()
    solver2 = MpInverse(q, D2_, E, c)
    solver2.solve()

    data_name = in_folder + "/pk%d_%s_*.dat" % (pno+1, 'asc' if ad == 0 else 'dsc')
    return data_name, [
            [ [ "MP-inverse",  solver0], [ "iterated-WLS",  solver1] ],
            [ [ "MP-inverse",  solver0], [ "MP-inverse (rank 5 denoised)",  solver5, "rank=5"] ],
            [ [ "MP-inverse",  solver0], [ "MP-inverse (rank 4 denoised)",  solver4, "rank=4"] ],
            [ [ "MP-inverse",  solver0], [ "MP-inverse (rank 3 denoised)",  solver3, "rank=3"] ],
            [ [ "MP-inverse",  solver0], [ "MP-inverse (rank 2 denoised)",  solver2, "rank=2"] ],
            ]

def do_a_range_definitive(in_folder, sd, pno, ad, range_, c_all):
    eslice = slice(*range_)

    q   = sd.intensity_array[0,:,0]
    D   = sd.intensity_array[eslice,:,1].T
    D3_  = get_denoised_data(D, rank=3)
    E   = sd.intensity_array[eslice,:,2].T
    c = c_all[eslice]

    solver0 = MpInverse(q, D3_, E, c)
    solver0.solve()
    solver1 = IteratedWLS(q, D3_, E, c)
    solver1.solve(weighted=True)
    solver2 = NormMinizerSmoothed(q, D3_, E, c)
    solver2.solve(smoothed=False, apositive=False)
    solver3 = NormMinizerSmoothed(q, D3_, E, c)
    solver3.solve(smoothed=True, apositive=False)
    solver4 = NormMinizerSmoothed(q, D3_, E, c)
    solver4.solve(smoothed=False, apositive=True)
    solver5 = NormMinizerSmoothed(q, D3_, E, c)
    solver5.solve(smoothed=True, apositive=True)

    data_name = in_folder + "/pk%d_%s_*.dat" % (pno+1, 'asc' if ad == 0 else 'dsc')
    return data_name, [
            [ [ "MP-inverse",  solver0], [ "iterated-WLS",  solver1] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min",  solver2] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min-smooth)",  solver3] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min-A_positive)",  solver4] ],
            [ [ "MP-inverse",  solver0], [ "Norm-min-smooth-A_positive)",  solver5] ],
            ]

def do_a_range_almerge(in_folder, sd, pno, ad, range_, c_all):
    eslice = slice(*range_)

    q   = sd.intensity_array[0,:,0]
    D   = sd.intensity_array[eslice,:,1].T
    D3_  = get_denoised_data(D, rank=3)
    E   = sd.intensity_array[eslice,:,2].T
    c = c_all[eslice]

    solver0 = MpInverse(q, D3_, E, c)
    solver0.solve()
    solver1 = AlMergeSolver(q, D3_, E, c)
    solver1.solve()
    solver3 = solver1.copy()
    solver5 = solver1.copy()
    solver1.scale(np.percentile(solver0.P[:,0], 95))
    solver2 = NormMinizerSmoothed(q, D3_, E, c)
    solver2.solve(smoothed=True, apositive=False)
    solver3.scale(np.percentile(solver2.P[:,0], 95))
    solver4 = NormMinizerSmoothed(q, D3_, E, c)
    solver4.solve(smoothed=True, apositive=True)
    solver5.scale(np.percentile(solver4.P[:,0], 95))

    data_name = in_folder + "/pk%d_%s_*.dat" % (pno+1, 'asc' if ad == 0 else 'dsc')
    return data_name, [
            [ [ "MP-inverse",  solver0], [ "ALMERGE",  solver1] ],
            [ [ "Norm-min-smooth",  solver2], [ "ALMERGE",  solver3] ],
            [ [ "Norm-min-smooth-A_positive",  solver4], [ "ALMERGE",  solver5] ],
            ]

def compute_nrmsd(max_y, y1, y2):
    return np.sqrt( np.average( (y1 - y2)**2 ) )/max_y*100

class ProofViewer(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.applied = False

    def show(self, data_name, plot_specs, save_images=True, auto=True, show_bq=True):
        self.data_name = cut_upper_folders(data_name)
        self.plot_specs = plot_specs
        self.image_folder = get_setting('temp_folder') + '/images'
        self.save_images = save_images
        self.auto = auto
        self.show_bq = show_bq
        title   = "ProofViewer"
        Dialog.__init__( self, self.parent, title )

    def body( self, body_frame ):

        sframe = ScrolledFrame(body_frame )
        sframe.pack()
        toolbar_frame = Tk.Frame(body_frame)
        toolbar_frame.pack()

        cframe = sframe.interior
        nrows = max(2, len(self.plot_specs))
        ncols = 2 if self.show_bq else 1

        height = 2 * nrows if self.show_bq else 3 * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, height))
        fig.tight_layout()

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.toolbar = NavigationToolbar( self.mpl_canvas, toolbar_frame )
        self.toolbar.update()

        self.fig = fig
        self.axes = axes

        fig.suptitle("Comparison of Extrapolation Methods using " + self.data_name, fontsize=20)

        t_ymin1 = []
        t_ymax1 = []
        t_ymin2 = []
        t_ymax2 = []
        for i, spec_row in enumerate(self.plot_specs):
            if self.show_bq:
                ax1 = axes[i,0]
                ax2 = axes[i,1]
            else:
                ax1 = axes[i]
            ax1.set_title("A(q)")
            ax1.set_ylabel('$Log_{10}$( Intensity )')
            if self.show_bq:
                ax2.set_title("B(q)")
                ax2.set_ylabel('$Linear$( Intensity )')

            for spec in spec_row:
                label, solver = spec[0:2]
                solver.plot_A(ax1, label)
                if self.show_bq:
                    solver.plot_B(ax2, label)

                ymin, ymax = ax1.get_ylim()
                t_ymin1.append(ymin)
                t_ymax1.append(ymax)
                if self.show_bq:
                    ymin, ymax = ax2.get_ylim()
                    t_ymin2.append(ymin)
                    t_ymax2.append(ymax)

            ax1.legend()
            if self.show_bq:
                ax2.legend()

        f_ymin1 = np.percentile( t_ymin1, 5 )
        f_ymax1 = np.percentile( t_ymax1, 100 )
        if self.show_bq:
            f_ymin2 = np.percentile( t_ymin2, 10 )
            f_ymax2 = np.percentile( t_ymax2, 90 )
        for i, spec_row in enumerate(self.plot_specs):
            if self.show_bq:
                ax1 = axes[i,0]
                ax2 = axes[i,1]
            else:
                ax1 = axes[i]
            ax1.set_ylim(f_ymin1, f_ymax1)
            if self.show_bq:
                ax2.set_ylim(f_ymin2, f_ymax2)

            if self.show_bq:
                def get_txext_xy_ax2():
                    xmin, xmax = ax2.get_xlim()
                    ymin, ymax = ax2.get_ylim()
                    tx = xmin*0.9 + xmax*0.1
                    if (ymin + ymax)/2 < 0:
                        ty = ymin*0.8 + ymax*0.2
                    else:
                        ty = ymin*0.4 + ymax*0.6
                    return tx, ty

            P_list = []
            text_drawn = False
            for spec in spec_row:
                solver = spec[1]
                P_list.append(solver.P)
                if len(spec) > 2:
                    text = spec[2]
                else:
                    text = None
                if text is not None:
                    text_drawn = True
                    if self.show_bq:
                        tx, ty = get_txext_xy_ax2()
                        ax2.text(tx, ty, text, ha='left', fontsize=30, alpha=0.3)

            max_y = np.percentile(P_list[0][:,0], 95)
            if not text_drawn and self.show_bq:
                if P_list[1].shape[1] > 1:
                    b_nrsmd = compute_nrmsd(max_y, P_list[0][:,1], P_list[1][:,1] )
                    text = "nRMSD=%5.2f%%" % (b_nrsmd)
                    tx, ty = get_txext_xy_ax2()
                    ax2.text(tx, ty, text, ha='left', fontsize=30, alpha=0.3)

            a_nrmsd = compute_nrmsd(max_y, P_list[0][:,0], P_list[1][:,0] )
            xmin, xmax = ax1.get_xlim()
            ymin, ymax = ax1.get_ylim()
            tx = xmin*0.8 + xmax*0.2
            ty = ymin*0.3 + ymax*0.7
            text = "nRMSD=%5.2f%%" % (a_nrmsd)
            ax1.text(tx, ty, text, ha='left', fontsize=30, alpha=0.3)
        if self.show_bq:
            top = 0.75 + (nrows-2)*0.05     # i.e., 0.75 if nrows == 2, 0.9 if nrows == 5
        else:
            top = 0.85 + (nrows-2)*0.05     # i.e., 0.85 if nrows == 2, 0.9 if nrows == 3
        fig.subplots_adjust( top=top, left=0.05, wspace=0.18 )

        if self.save_images:
            self.save_image()

    def apply(self):
        self.applied = True

    def save_image(self):
        name_ = self.data_name.replace('_*.dat', '')
        filename = name_.replace('/', '-') + '.png'
        path = os.path.join( self.image_folder, filename ).replace('/', '\\')
        print('save_image: path=', path)
        self.fig.savefig( path )
        if self.auto:
            self.after(1000, self.ok)

def survey_for_all_data(restart=None, auto=True, callback=do_a_range, show_bq=True):
    root = get_tk_root()
    started = False

    def do_a_folder(in_folder, uv_folder, plot):
        nonlocal started
        mtd_file = get_mtd_filename(uv_folder)
        if mtd_file is not None:
            return True, None

        if restart is not None:
            if not started:
                if in_folder.find(restart) < 0:
                    return True, None
                else:
                    started = True

        print(in_folder)
        try:
            spec_list = create_plot_specs(in_folder, uv_folder, callback)
            for spec in spec_list:
                data_name, plot_specs = spec
                viewer = ProofViewer(root)
                viewer.show(data_name, plot_specs, auto=auto, show_bq=show_bq)
                if not viewer.applied:
                    break

            return viewer.applied, None
        except:
            print('create_plot_specs failed for', in_folder )
            return True, None

    pytools = get_pytools_folder()
    data_folder = pytools + '/Data'
    print( data_folder )
    serial_folder_walk( data_folder, do_a_folder )
    