# coding: utf-8
"""
    MctDecompDialog.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from lmfit import Parameters, minimize
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkUtils import split_geometry
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable, BlinkingFrame
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, get_xray_picking
from MicrofluidicElution import get_mtd_elution
from TwoStateSolver import TwoStateSolver
from ThreeStateMonomerSolver import ThreeStateMonomerSolver
from ThreeStateDimerSolver import ThreeStateDimerSolver
from PreviewButtonFrame import PreviewButtonFrame
from SvdDenoise import get_denoised_data

MODEL_ID_LIST = ["One State", "Two-State", "Three-State Monomer", "Three-State Dimer"]
DRAW_COEFF_VECTORS = False

class MctDecompDialog(Dialog):
    def __init__(self, parent, xdata, in_folder):
        self.logger = logging.getLogger(__name__)
        self.busy = False
        self.applied = False
        self.xdata = xdata
        self.mapper = None
        self.in_folder = in_folder
        self.popup_menu = None
        self.popup_label0 = None
        self.popup_label1 = None
        set_setting('conc_factor', 5)   # TODO: change setting according to data date
        Dialog.__init__(self, parent, "Micorfluidic Decomposer", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tbframe = Tk.Frame(body_frame)
        tbframe.pack(fill=Tk.X, expand=1)
        iframe = Tk.Frame(body_frame)
        iframe.pack(fill=Tk.X, expand=1, padx=20, pady=10)

        self.build_canvas(cframe, tbframe)
        self.builf_param_widgets(iframe)
        self.draw_init()

    def get_canvas_width( self ):
        return int( self.mpl_canvas_widget.cget('width'))

    def build_canvas(self, cframe, tbframe):
        figsize = (17,6) if is_low_resolution() else (23,8)
        fig = plt.figure(figsize=figsize)
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.mpl_canvas.mpl_connect('button_press_event', self.on_mpl_button_press)
        self.toolbar = NavigationToolbar(self.mpl_canvas, tbframe)
        self.toolbar.update()

        ax1 = fig.add_subplot(131, projection='3d') # it seems this must be done after creation of self.mpl_canvas
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        axt = ax3.twinx()
        self.axins = None
        axes = [ax1, ax2, ax3, axt]
        self.fig = fig
        self.axes = axes

        fig.subplots_adjust(top=0.88, bottom=0.1, left=0.03, right=0.95, wspace=0.2)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, expand=1, padx=5)
        bottom_space = Tk.Frame(self, height=10)
        bottom_space.pack()

        for j in range(2):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="◀ Main", command=self.cancel)
        w.grid(row=0, column=0, sticky=Tk.W, padx=10, pady=5)
        self.back_button = w

        w = Tk.Button(box, text="▶ Serial Analysis", command=self.ok)
        w.grid(row=0, column=1, sticky=Tk.E, padx=10, pady=5)
        self.next_button = w

    def show(self):
        self._show(wait=False)

        fully_automatic = get_setting( 'fully_automatic' )
        if fully_automatic == 1:
            self.parent.after(2000, lambda: self.next_button.invoke() )

        self.waited_widget = Tk.Frame(self)
        self.wait_window(self.waited_widget)

    def ok(self):
        self.apply()
        self.close()

    def exit(self):
        self.close()

    def close(self):
        self.withdraw()
        self.update_idletasks()
        self.grab_release()
        self.waited_widget.destroy()

    def apply(self):
        self.preview_frame.update_settings()
        self.applied = True

    def draw_init(self):
        xdata = self.xdata
        in_folder = self.in_folder
        num_files = xdata.data.shape[1]
        mtd_elution = get_mtd_elution(in_folder, num_files)
        self.slice_ = slice_ = mtd_elution.guess_range(num_files)
        ax1, ax2 = self.axes[0:2]
        self.fig.suptitle("Decomposition for " + in_folder, fontsize=20)
        xdata.plot(ax1)
        self.draw_elution(ax2, xdata, mtd_elution, slice_)

    def decompose(self, draw_decomposition=False):
        model_id = self.model_id.get()
        if model_id == MODEL_ID_LIST[0]:
            return

        self.solve_btn_blink.stop()

        self.drawing_curves = False
        self.showing_detail = False
        ax3, axt = self.axes[2:4]
        ax3.cla()
        axt.cla()
        axt.grid(False)
        if self.axins is not None:
            self.axins.remove()
            self.axins = None

        xdata = self.xdata

        self.ax3_title = 'Decomposition using %s Model' % model_id
        ax3.set_title(self.ax3_title, fontsize=16)

        model_id = self.model_id.get()
        if model_id == MODEL_ID_LIST[1]:
            solver = TwoStateSolver()
            data = xdata.data[:,self.slice_]
            solver.solve(data)
            self.solver = solver
            self.draw_decomposition(ax3, data, xdata.e_index, axt)
            if DRAW_COEFF_VECTORS:
                self.draw_coeff_vectors(ax3, self.slice_)
        elif model_id == MODEL_ID_LIST[2]:
            solver = ThreeStateMonomerSolver()
            data = xdata.data[:,self.slice_]
            solver.solve(data)
            self.solver = solver
            self.draw_decomposition(ax3, data, xdata.e_index, axt)
        elif model_id == MODEL_ID_LIST[3]:
            solver = ThreeStateDimerSolver()
            data = xdata.data[:,self.slice_]
            solver.solve(data)
            self.solver = solver
            self.draw_decomposition(ax3, data, xdata.e_index, axt)
        self.mpl_canvas.draw()

    def get_model(self):
        return self.solver.get_fit_model()

    def draw_elution(self, ax, xdata, mtd_elution, slice_):
        pick_pos = get_xray_picking()
        ax.set_title("Elution at Q=%.2g" % pick_pos, fontsize=16)
        e_y = xdata.e_y
        ax.plot(e_y, color='orange')
        axt = ax.twinx()
        axt.grid(False)
        # x, y = mtd_elution.get_elution_data(len(e_y))
        t, x, y = mtd_elution.make_simulation_data()
        axt.plot(x, y, ':', color='blue', alpha=0.3)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

        f, t = [slice_.start, slice_.stop]
        for k, p in enumerate([f, t]):
            ax.plot([p, p], [ymin, ymax], ':', color='gray')
            self.range_vars[k].set(p)

        rect = mpl_patches.Rectangle(
                (f, ymin),      # (x,y)
                t - f,          # width
                ymax - ymin,    # height
                facecolor   = 'cyan',
                alpha       = 0.2,
            )
        ax.add_patch(rect)

    def builf_param_widgets(self, iframe):

        for j in range(3):
            iframe.columnconfigure(j, weight=1)

        width = self.get_canvas_width()//3
        print('width=', width)

        space_frame = Tk.Frame(iframe, width=width)
        space_frame.grid(row=0, column=0)

        range_frame = Tk.Frame(iframe, width=width)
        range_frame.grid(row=0, column=1, sticky=Tk.N)

        decomp_frame = Tk.Frame(iframe, width=width)
        decomp_frame.grid(row=0, column=2, sticky=Tk.N)

        range_label = Tk.Label(range_frame, text="Range Selection: ")
        range_label.grid(row=0, column=0)

        j_max = len(self.xdata.j) - 1

        self.range_vars = []
        for k, t in enumerate(["from", "to"]):
            label = Tk.Label(range_frame, text=t)
            label.grid(row=0, column=1+k*2)
            var = Tk.IntVar()
            self.range_vars.append(var)
            spinbox = Tk.Spinbox( range_frame, textvariable=var,
                                    from_=0, to=j_max, increment=1,
                                    justify=Tk.CENTER, width=6 )
            spinbox.grid(row=0, column=2+k*2, padx=10)

        model_label = Tk.Label(decomp_frame, text="Model Selection: ")
        model_label.grid(row=0, column=0)

        self.model_id = Tk.StringVar()
        self.model_id.set(MODEL_ID_LIST[1])
        model_id_box = ttk.Combobox(decomp_frame, textvariable=self.model_id, width=24, justify=Tk.CENTER)
        model_id_box[ 'values' ] = MODEL_ID_LIST
        model_id_box.grid(row=0, column=1)

        self.model_id.trace('w', self.model_id_tracer)

        self.solve_btn_blink = BlinkingFrame(decomp_frame)
        self.solve_btn_blink.grid(row=0, column=2)
        self.solve_btn = Tk.Button(self.solve_btn_blink, text="Solve", command=self.decompose)
        self.solve_btn.pack()
        self.solve_btn_blink.objects = [self.solve_btn]

        space = Tk.Frame(decomp_frame, height=20)
        space.grid(row=1, column=0)

        self.preview_frame = PreviewButtonFrame(decomp_frame, bd=3, relief=Tk.RIDGE, dialog=self)
        self.preview_frame.grid(row=2, column=0, columnspan=3)

    def model_id_tracer(self, *args):
        model_id = self.model_id.get()
        print(model_id)
        self.solve_btn_blink.start()
        if model_id == MODEL_ID_LIST[2]:
            self.preview_frame.set_guard_for_three_state_model()
        else:
            self.preview_frame.remove_guard_for_three_state_model()

    def on_mpl_button_press(self, event):
        if event.xdata is None:
            return

        if event.button == 3:
            self.update_popup_menu()
            self.create_popup_menu()
            w, h, x, y = split_geometry(self.geometry())
            self.popup_menu.post(x + event.x + 20, y + h - event.y - 180)
            # TODO: do this without dialog dependent params 20, 180
            return

    def create_popup_menu(self):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu( self, tearoff=0 )
            self.popup_menu.add_command(label=self.popup_label0, command=self.redraw)
            self.popup_menu.add_command(label=self.popup_label1, command=lambda: self.redraw(toggle=False, toggle_detail=True))
            self.popup_menu.add_command(label="Inspection in various angles", command=self.inspection_dialog)
        else:
            self.popup_menu.entryconfigure(0, label=self.popup_label0)
            self.popup_menu.entryconfigure(1, label=self.popup_label1)
        state1 = Tk.NORMAL if not self.drawing_curves and self.model_id.get() == MODEL_ID_LIST[1] else Tk.DISABLED
        self.popup_menu.entryconfigure(1, state=state1)

    def update_popup_menu(self):
        self.popup_label0 = 'Draw with bands' if self.drawing_curves else 'Draw with curves'
        self.popup_label1 = ('Hide' if self.showing_detail else 'Show') + ' denaturant dependency'

    def redraw(self, toggle=True, toggle_detail=False):
        xdata = self.xdata
        data = xdata.data[:,self.slice_]

        ax3, axt = self.axes[2:4]
        for ax in [ax3, axt]:
            ax.cla()
        ax3.set_title(self.ax3_title, fontsize=16)
        axt.grid(False)
        if toggle:
            self.drawing_curves ^= True
        if toggle_detail:
            self.showing_detail ^= True
        self.draw_decomposition(ax3, data, xdata.e_index, axt)

    def draw_decomposition(self, ax, data, e_index, axt, toggle_detail=False):
        if self.drawing_curves:
            self.solver.plot_components(ax, data, e_index, axt=axt, start=self.slice_.start)
        else:
            self.solver.draw_components_with_bands(ax, data, e_index, start=self.slice_.start, detail=self.showing_detail)
        self.mpl_canvas.draw()
        self.drawn_info = (self.ax3_title, data, e_index, self.slice_.start, self.showing_detail)

    def draw_coeff_vectors(self, ax, slice_):
        e_index = self.xdata.e_index
        values = self.solver.P[e_index,:]
        xspan = max(abs(values[0]), abs(values[2]))*2
        yspan = max(abs(values[1]), abs(values[3]))*2

        side = self.solver.guess_blank_location(slice_)

        if side == 'left':
            bbox = (0.1, 0.2, 1, 1)
            loc = 'lower left'
        else:
            bbox = (0, 0.2, 0.9, 1)
            loc = 'lower right'

        axins = inset_axes(ax, width="30%", height="30%",
                            bbox_to_anchor=bbox,
                            bbox_transform=ax.transAxes, loc=loc)
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axins.set_facecolor('xkcd:cream')
        axins.patch.set_alpha(0.3)
        axins.grid(False)
        axins.set_title("Coefficient Vectors", fontsize=14)
        axins.plot(0, 0, 'o', color='black', markersize=3)
        axins.set_xlim(-xspan, xspan)
        axins.set_ylim(-yspan, yspan)
        for t, p in [['(af,bf)', values[0:2]], ['(au,bu)', values[2:4]]]:
            x, y = p
            print(t, x, y)
            axins.text(x, y, t)
            axins.annotate('', xy=(x, y), xytext=(0,0),
                arrowprops=dict(headwidth=5, headlength=6, width=0.5, color='black', shrink=0)
               )
        self.axins = axins

    def get_decomp_info(self):
        from MctRangeInfo import MctRangeInfo
        conc = self.get_conc()
        ranges = self.make_alt_ranges()
        curves = self.get_conc_curves()
        return MctRangeInfo(conc, ranges, curves)

    def get_conc(self):
        # return scaled C list

        model_id = self.model_id.get()

        C = self.solver.C
        y = self.xdata.e_curve.y[self.slice_]

        if model_id == MODEL_ID_LIST[1]:
            scale1 = y[0]/C[0,0]
            scale2 = y[-1]/C[2,-1]
            rest_list = [scale1*C[0,:], scale1*C[1,:], scale2*C[2,:], scale2*C[3,:]]
        elif model_id == MODEL_ID_LIST[2]:
            scale1 = y[0]/C[0,0]
            m = len(y)//2
            scale2 = y[m]/C[1,m]
            scale3 = y[-1]/C[2,-1]
            rest_list = [scale1*C[0,:], scale2*C[1,:], scale3*C[2,:]]
        else:
            assert False

        return rest_list

    def get_conc_curves(self):
        return self.solver.get_modeled_curves(self.xdata.e_index)

    def make_range_info(self):
        return None

    def make_alt_ranges(self):
        from molass_legacy.DataStructure.PeakInfo import PeakInfo
        from molass_legacy.DataStructure.AnalysisRangeInfo import PairedRange

        start, stop = self.slice_.start, self.slice_.stop

        if False:
            import molass_legacy.KekLib.DebugPlot as dplt
            from molass_legacy.Elution.CurveUtils import simple_plot

            e_curve = self.xdata.e_curve
            fig = dplt.figure()
            ax = fig.gca()
            ax.set_title("make_range_info debug")
            simple_plot(ax, e_curve)

            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            for x in [start, stop]:
                ax.plot([x,x], [ymin, ymax], ':', color='gray')

            dplt.show()

        """
        make a range analogous to a SEC peak range
        """
        ret_list = []

        pno = 0
        pinfo = PeakInfo(pno, start)
        range_list = [[start, stop-1]]
        ret_list.append(PairedRange(pinfo, *range_list))

        model_id = self.model_id.get()
        if model_id == MODEL_ID_LIST[2]:
            itx = int(start + self.solver.get_imermediate_topx())
            pno += 1
            pinfo = PeakInfo(pno, itx)
            range_list = [[start, stop-1]]
            ret_list.append(PairedRange(pinfo, *range_list))

        pno += 1
        pinfo = PeakInfo(pno, stop-1)
        range_list = [[start, stop-1]]
        ret_list.append(PairedRange(pinfo, *range_list))

        return ret_list

    def inspection_dialog(self):
        from InspectionInAngles import InspectionInAnglesDialog
        dialog = InspectionInAnglesDialog(self, self.xdata.vector, self.xdata.e_index, self.drawn_info)
        dialog.show()
