"""
    Selective.VariedPropAnalysis.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import molass_legacy.KekLib.DebugPlot as dplt
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar, get_color
from molass_legacy.KekLib.ScrolledFrame import ScrolledFrame
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def get_code_text(editor):
    text = editor.advanced_frame.code_text.get(1.0, Tk.END)
    print("text=", text)
    return text

def check_code_text(code_text):
    context = {}
    exec(code_text, context)
    return True, context['proportions']

class VariedPropAnalysis(Dialog):
    def __init__(self, parent, editor, params_info, num_variations=20):
        self.parent = parent
        self.editor = editor
        code_text = get_code_text(editor)
        ret, func = check_code_text(code_text)
        self.proportions_func = func
        self.modelname = editor.get_current_modelname()
        self.params_info = params_info
        self.num_variations = num_variations
        self.current_index = None
        Dialog.__init__(self, parent, "Varied Proportion Analysis", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        self.scrolled_frame = ScrolledFrame(body_frame)
        self.scrolled_frame.pack(anchor=Tk.N)
        cframe = self.scrolled_frame.interior

        self.ncols = 5
        self.nrows = self.num_variations//self.ncols
        height = 2.5 * self.nrows
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=(20,height))
        self.draw_varied_proportions(fig, axes)
        fig.tight_layout()

        self.fig = fig
        self.axes = axes
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.popup_menu = None
        self.mpl_canvas.mpl_connect('button_press_event', self.on_figure_click)

    def buttonbox(self):
        # task: pack the close button so that it won't hide
        box = Tk.Frame(self)
        box.pack(fill=Tk.X)

        tframe = Tk.Frame(box)
        tframe.pack(side=Tk.LEFT, padx=20, pady=10)
        bframe = Tk.Frame(box)
        bframe.pack(side=Tk.RIGHT, padx=20, pady=10)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

        w = Tk.Button(bframe, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=20)

        w = Tk.Button(bframe, text="OK", width=10, command=self.ok)
        w.pack(side=Tk.LEFT, padx=20)

        w = Tk.Button(bframe, text="RDR Chart", width=10, command=self.show_rdr_chart)
        w.pack(side=Tk.LEFT, padx=20)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        self.set_geometry()

    def set_geometry(self):
        self.update()
        canvas_width = int(self.mpl_canvas_widget.cget( 'width' ))
        canvas_height = int(self.mpl_canvas_widget.cget( 'height' ))
        print("canvas_height=", canvas_height)
        height = min(900, canvas_height+200)
        margin_width = 40
        wxh = '%dx%d' % (canvas_width + margin_width, height)
        geometry = self.geometry()
        new_geometry = re.sub( r'(\d+x\d+)(.+)', lambda m: wxh + m.group(2), geometry)
        self.geometry(new_geometry)

    def draw_varied_proportions(self, fig, axes, devel=True):
        if devel:
            from importlib import reload
            import Selective.PropOptimizerImpl
            reload(Selective.PropOptimizerImpl)
            import Selective.PropOptimizer
            reload(Selective.PropOptimizer)
        from Selective.PropOptimizerImpl import compute_range_rgs
        from Selective.PropOptimizer import PropOptimizer
        from molass_legacy._MOLASS.SerialSettings import set_setting
        from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve

        sd = self.editor.corbase_info.sd
        D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
        x = xr_curve.x
        y = xr_curve.y

        self.prop_optimizer = prop_optimizer = PropOptimizer(self.modelname, x, y)

        editor_ranges = self.editor.get_current_frame().editor_ranges

        # set_setting('local_debug', True)
        # ecurve = ElutionCurve(y, x=x)     # there still is a bug
        ecurve = ElutionCurve(y)
        self.paired_ranges = paired_ranges = ecurve.get_default_paired_ranges()
        # set_setting('local_debug', False)
        print("paired_ranges=", paired_ranges)
        print("editor_ranges=", editor_ranges)

        fig.suptitle("Varied Proportion Avalysis of %s using %s" % (get_in_folder(), self.modelname), fontsize=20)

        pv, fv_list, rgs_list, rdr_list, peaks_list = self.params_info

        self.min_n = n = np.argmin(rdr_list)
        minRDR = rdr_list[n]

        self.pv = pv

        for i, p in enumerate(self.pv):
            if i != 17:
                # continue
                pass

            print([i], "optimizing with p=%.3g" % p)

            opt_peaks = peaks_list[i]
            rdr = rdr_list[i]
            rgs = rgs_list[i]

            j, k = divmod(i,5)

            ax = axes[j,k]
            ax.plot(x, y, color="orange")

            cy_list = prop_optimizer.compute_cy_list(opt_peaks)
            for cy in cy_list:
                ax.plot(x, cy, ":")
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red")
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = xmin*0.9 + xmax*0.1
            ty = ymin*0.2 + ymax*0.8
            ax.text(tx, ty, "p=%.3g" % p, fontsize=20, alpha=0.3)
            ty = ymin*0.4 + ymax*0.6
            ax.text(tx, ty, "Rgs=%.3g, %.3g" % tuple(rgs), fontsize=20, alpha=0.3)

        if n is not None:
            j, k = divmod(n,5)
            ax = axes[j,k]
            ax.patch.set_facecolor('green')
            ax.patch.set_alpha(0.1)

        self.fv = np.array(fv_list)
        self.rdr = np.array(rdr_list)
        self.peaks_list = peaks_list

    def show_fv_chart(self):
        indeces = [str(i) for i in range(self.num_variations)]
        with dplt.Dp(window_title='FV Chart', ok_only=True, ok_text="Close"):
            fig, ax = dplt.subplots()
            ax.set_title("FV Chart", fontsize=16)
            ax.bar(indeces, self.fv)
            fig.tight_layout()
            dplt.show()

    def show_rdr_chart(self, devel=True):
        if devel:
            from importlib import reload
            import Selective.RdrChart
            reload(Selective.RdrChart)
        from .RdrChart import draw_rdr_chart
        title = "RDR_AD Chart of %s with %s" % (get_in_folder(), self.modelname)
        draw_rdr_chart(title, self.pv, self.rdr)

    def on_figure_click(self, event):
        if event.button == 3:
            axes = self.axes
            for i in range(axes.shape[0]):
                for j in range(axes.shape[1]):
                    if event.inaxes == axes[i,j]:
                        self.current_index = (i,j)
                        self.current_event = event
                        self.show_popup_menu(event)
                        break

    def show_popup_menu(self, event):
        from molass_legacy.KekLib.TkUtils import split_geometry
        self.create_popup_menu(event)
        canvas = self.mpl_canvas_widget
        cx = canvas.winfo_rootx()
        cy = canvas.winfo_rooty()
        w, h, x, y = split_geometry(canvas.winfo_geometry())
        self.popup_menu.post(cx + int(event.x), cy + h - int(event.y))

    def create_popup_menu(self, event):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0 )
            self.popup_menu.add_command(label='Copy Proportions', command=self.copy_proportions)

    def copy_proportions(self):
        k = self.get_current_seq_index()
        print([k], self.pv[k])

    def get_current_seq_index(self):
        if self.current_index is None:
            return self.min_n
        else:
            i, j = self.current_index
            return self.ncols*i + j

    def get_current_peaks(self):
        k = self.get_current_seq_index()
        return self.peaks_list[k]

    def validate(self):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        ret = MessageBox.askokcancel("Confirmation",
                "This will replace the decomposition with the optimal result.\n"
                + "Are you sure to proceed?",
                parent=self.parent)
        if ret:
            return 1
        else:
            return 0

    def apply(self):
        decomp_result = self.make_decomp_result()
        def apply_closure():
            self.editor.advanced_frameupdate_button_status(change_id="VPA")
            self.editor.update_current_frame_with_params(decomp_result)
        self.editor.after(100, apply_closure)

    def make_decomp_result(self, devel=True):
        if devel:
            from importlib import reload
            import Selective.V1ParamsAdapter
            reload(Selective.V1ParamsAdapter)
        from Selective.V1ParamsAdapter import make_decomp_result_impl

        peaks = self.get_current_peaks()
        return make_decomp_result_impl(self.editor, peaks)

def compute_peak_params(queue, modelname, sd, pv, fv_list, rgs_list, rdr_list, peaks_list, prop_func, devel=True):
    from time import sleep
    if devel:
        from importlib import reload
        import Selective.PropOptimizer
        reload(Selective.PropOptimizer)
        import Selective.PropOptimizer
        reload(Selective.PropOptimizer)
    from Selective.PropOptimizerImpl import compute_range_rgs
    from Selective.PropOptimizer import PropOptimizer

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    x = xr_curve.x
    y = xr_curve.y
    prop_optimizer = PropOptimizer(modelname, x, y)
    init_peaks = prop_optimizer.get_init_params()
    props = prop_optimizer.compute_props(init_peaks)
    print("props=", props)
    paired_ranges = xr_curve.get_default_paired_ranges()

    for i, p in enumerate(pv):
        print([i], "optimizing with p=%.3g" % p)
        sleep(1)

        props = prop_func(p)
        opt_ret = prop_optimizer.optimize(props, init_params=init_peaks)
        opt_peaks = opt_ret.x.reshape(init_peaks.shape)
        cy_list = prop_optimizer.compute_cy_list(opt_peaks)
        C = np.array(cy_list)
        rgs = compute_range_rgs(qv, D, E, paired_ranges, C)
        rdr = abs(rgs[0] - rgs[1])*2/(rgs[0] + rgs[1])

        fv_list.append(opt_ret.fun)
        peaks_list.append(opt_peaks)
        rgs_list.append(rgs)
        rdr_list.append(rdr)
        if queue is None:
            print([i], "done")
        else:
            queue.put([i])

    if queue is None:
        print([-1], "finished")
    else:
        queue.put([-1])

def show_vp_analysis_impl(button_frame, prop_func=None, in_another_thread=True):
    from time import sleep
    if in_another_thread:
        import queue
        from threading import Thread

    editor = button_frame.editor
    modelname = editor.get_current_modelname()
    sd = editor.corbase_info.sd

    pv = button_frame.get_prop_vector()
    fv_list = []
    rgs_list = []
    rdr_list = []
    peaks_list = []

    if in_another_thread:
        queue = queue.Queue()
        thread = Thread(
                        target=compute_peak_params,
                        name='ComputePeakParams',
                        args=[queue, modelname, sd, pv, fv_list, rgs_list, rdr_list, peaks_list, prop_func]
                        )
        thread.start()

        while True:
            sleep(0.5)
            info = queue.get()
            if info is not None:
                button_frame.progress_update(info[0])
                if info[0] < 0:
                    break

        thread.join()
    else:
        compute_peak_params(None, modelname, sd, pv, fv_list, rgs_list, rdr_list, peaks_list)

    params_info = [pv, fv_list, rgs_list, rdr_list, peaks_list]
    dialog = VariedPropAnalysis(editor.parent, editor, params_info)
    dialog.show()
