"""
    Selective.AdvancedFrame.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import sys
import numpy as np
try:
    from molass_legacy.KekLib.OurTkinter import Tk, ttk
except:
    import tkinter as Tk
    import tkinter.ttk as ttk
from molass_legacy._MOLASS.Version import is_developing_version

ALLOW_TRADITIONAL = is_developing_version()
USE_COMBOBOX_FOR_PROPS_TEXT = True
MAX_NUM_COMPONENTS = 7
CHANGE_NAMES_DICT = {"CNC": "Components Change",
                     "PSD": "PSD Estimate",
                     "RDR": "RDR Minimization",
                     "KPD": "KPD Minimization",
                     "VPA": "VP Analysis",
                     }

def get_change_name(change_id):
    return CHANGE_NAMES_DICT[change_id]

class AdvancedFrame(Tk.LabelFrame):
    def __init__(self, parent, editor, **kwargs):
        from molass_legacy.KekLib.SaferSpinbox import SaferSpinbox as Spinbox
        from molass_legacy.Selective.Proportions import Proportions_
        global PROPORTIONS_
        PROPORTIONS_ = Proportions_()

        self.parent = parent
        self.editor = editor
        Tk.LabelFrame.__init__(self, parent, text="Advanced Optimization", labelanchor=Tk.N, **kwargs)

        iframe = Tk.Frame(self)
        iframe.pack(padx=0, pady=10)
        irow = 0

        params_array = editor.get_current_params_array()
        num_components = len(params_array)

        nc_frame = Tk.Frame(iframe)
        nc_frame.grid(row=irow, column=0, columnspan=8, padx=10, sticky=Tk.W)
        label = Tk.Label(nc_frame, text="Num Components", state=Tk.DISABLED)
        label.grid(row=0, column=0, sticky=Tk.E)
        self.num_components_label = label
        self.num_components = Tk.IntVar()
        self.num_components.set(num_components)
        sb = Spinbox(nc_frame,
                    textvariable=self.num_components, from_=2, to=MAX_NUM_COMPONENTS,
                    increment=1,
                    justify=Tk.CENTER, width=3, state=Tk.DISABLED )
        sb.grid(row=0, column=1, sticky=Tk.W, padx=5)
        self.num_components_sb = sb
        self.num_components.trace_add('write', self.num_components_tracer)

        self.with_props = Tk.IntVar()
        self.with_props.set(0)
        w = Tk.Checkbutton(nc_frame, text="with", variable=self.with_props)
        w.grid(row=0, column=2, padx=5, pady=5)

        self.with_props_text_var = Tk.StringVar()
        if USE_COMBOBOX_FOR_PROPS_TEXT:
            w = ttk.Combobox(nc_frame, textvariable=self.with_props_text_var, width=18, justify=Tk.CENTER)
            w.grid(row=0, column=3, sticky=Tk.W)
            # w.bind("<<ComboboxSelected>>", self.with_props_tracer)
            self.props_text = w
        else:       
            w = Tk.Text(nc_frame, height=1, width=20)
            w.grid(row=0, column=3, sticky=Tk.W)
            self.props_text = w

        self.with_props_tracer()    # let the states of the widgets be consistent
        self.with_props.trace_add('write', self.with_props_tracer)

        w = Tk.Button(nc_frame, text="Change Num Components", width=20, command=self.change_num_components, state=Tk.DISABLED)
        w.grid(row=0, column=4, columnspan=2, padx=10, pady=5)
        self.cnc_button = w

        irow += 1
        minimizer_frame = Tk.Frame(iframe)
        minimizer_frame.grid(row=irow, column=0, columnspan=8, padx=5, pady=5, sticky=Tk.W+Tk.E)
        for i, weight in enumerate([1, 2, 2]):
            minimizer_frame.columnconfigure(i, weight=weight)

        w = Tk.Button(minimizer_frame, text="Estimate PSD", width=12, command=self.estimate_psd, state=Tk.DISABLED)
        w.grid(row=0, column=0, padx=10, pady=5)
        self.psd_button = w

        w = Tk.Button(minimizer_frame, text="Minimize Rg Diff Ratio", width=20, command=self.minimize_rdr, state=Tk.DISABLED)
        w.grid(row=0, column=1, padx=10, pady=5)
        self.rdr_button = w

        w = Tk.Button(minimizer_frame, text="Minimize Kratky Plot Diff", width=20, command=self.minimize_kpd, state=Tk.DISABLED)
        w.grid(row=0, column=2, padx=10, pady=5)
        self.kpd_button = w

        minv, maxv = self.get_min_max_values()
        self.min_val = Tk.DoubleVar()
        self.min_val.set(minv)
        self.max_val = Tk.DoubleVar()
        self.max_val.set(maxv)
        self.num_divs = Tk.IntVar()
        self.num_divs.set(20)

        irow += 1
        w = Tk.Label(iframe, text="Parameter Limits")
        w.grid(row=irow, column=0, columnspan=5, padx=5, pady=0)
        w = Tk.Label(iframe, text="Number of Divisions")
        w.grid(row=irow, column=5, padx=5)

        irow += 1
        w = Tk.Label(iframe, text="Min")
        w.grid(row=irow, column=0, padx=0, pady=0, sticky=Tk.E)
        sb = Spinbox(iframe,
                    textvariable=self.min_val, from_=0, to=10,
                    increment=0.05,
                    justify=Tk.CENTER, width=6 )
        sb.grid(row=irow, column=1, padx=0)
        w = Tk.Label(iframe, text="")
        w.grid(row=irow, column=2, padx=5)
        w = Tk.Label(iframe, text="Max")
        w.grid(row=irow, column=3, padx=0, sticky=Tk.E)
        sb = Spinbox(iframe,
                    textvariable=self.max_val, from_=0, to=10,
                    increment=0.05,
                    justify=Tk.CENTER, width=6 )
        sb.grid(row=irow, column=4, padx=0)
        sb = Spinbox(iframe,
                    textvariable=self.num_divs, from_=10, to=100,
                    increment=1,
                    justify=Tk.CENTER, width=6, state=Tk.DISABLED )
        sb.grid(row=irow, column=5, padx=5)

        irow += 1
        w = Tk.Label(iframe, text="Proportions\nDefinition")
        w.grid(row=irow, column=0, columnspan=2, padx=5, sticky=Tk.E)

        self.code_text = Tk.Text(iframe, width=36, height=3)
        self.code_text.grid(row=irow, column=2, columnspan=5, padx=5, pady=10, sticky=Tk.W)
        self.set_func_code_text()

        w = Tk.Button(iframe, text="VP Analysis", width=12, command=self.show_vp_analysis, state=Tk.DISABLED)
        w.grid(row=1, rowspan=irow, column=6, padx=10, pady=5)
        self.vpa_button = w

        w = Tk.Button(iframe, text="Reset", width=8, command=self.reset_to_initstatus, state=Tk.DISABLED)
        w.grid(row=irow, rowspan=2, column=6, padx=10, pady=5)

        irow += 1
        progress_frame = Tk.Frame(iframe)
        progress_frame.grid(row=irow, column=0, columnspan=8, padx=10, pady=5, sticky=Tk.W + Tk.E)

        progress_label = Tk.Label(progress_frame, text="Progress")
        progress_label.grid(row=0, column=0, sticky=Tk.E)

        self.progress_bar = ttk.Progressbar(progress_frame, orient ="horizontal", length=430, mode="determinate")
        self.progress_bar.grid(row=0, column=1, padx=10, pady=5)

        if sys.executable.find("pythonw") < 0:
            self.popup_menu = None
            widget = self.parent if self.editor is None else self.editor
            widget.bind("<Button-3>", self.on_right_click)

        self.advanced_widgets = [self.num_components_label, self.num_components_sb, self.cnc_button, self.psd_button]
        self.traditional_widgets = [self.rdr_button, self.kpd_button, self.vpa_button]
        self.after(100, self.enable_buttons)
        self.last_change_id = None  # will be set to one of CHANGE_IDS
        self.change_button_dict = {
            "CNC" : self.cnc_button,
            "PSD" : self.psd_button,
            "RDR" : self.rdr_button,
            "KPD" : self.kpd_button,
            "VPA" : self.vpa_button,
        }

    def get_last_change_id(self):
        return self.last_change_id

    def set_func_code_text(self):
        n = self.num_components.get()
        func_code = PROPORTIONS_.generate_funccode(n)
        self.code_text.delete(1.0, Tk.END)
        self.code_text.insert(1.0, func_code)
        opt_props_text = PROPORTIONS_.get_optimal_props_text(n)
        if USE_COMBOBOX_FOR_PROPS_TEXT:
            PROPORTIONS_.get_proportions_func(func_code)
            values = PROPORTIONS_.get_with_props_values(n)
            self.with_props_text_var.set(values[0])
            self.props_text.config(values=values)
        else:
            self.props_text.config(state=Tk.NORMAL)
            self.props_text.delete(1.0, Tk.END)
            self.props_text.insert(1.0, opt_props_text)
            self.with_props_tracer()

    def with_props_tracer(self, *args):
        if self.with_props.get() == 1:
            self.props_text.config(state=Tk.NORMAL)
        else:
            self.props_text.config(state=Tk.DISABLED)

    def get_min_max_values(self):
        n = self.num_components.get()
        return PROPORTIONS_.get_min_max_values(n)
    
    def num_components_tracer(self, *args):
        minv, maxv = self.get_min_max_values()
        self.min_val.set(minv)
        self.max_val.set(maxv)
        self.num_components_sb.config(state=Tk.NORMAL)
        self.set_func_code_text()

    def get_current_model(self):
        return self.editor.get_current_model()

    def disable_buttons(self):
        for buton in self.advanced_widgets + self.traditional_widgets:
            buton.config(state=Tk.DISABLED)
        self.update()

    def enable_buttons(self):
        widegts = []
        widegts += self.traditional_widgets
        model = self.get_current_model()
        if model.is_traditional():
            allow = ALLOW_TRADITIONAL
        else:
            allow = True
        if allow:
            widegts += self.advanced_widgets
        for w in widegts:
            w.config(state=Tk.NORMAL)
        self.update()

    def update_mode_dependent_state(self):
        self.disable_buttons()
        self.enable_buttons()

    def update_button_status(self, change_id=None):
        if change_id is not None:
            if self.last_change_id is not None:
                self.change_button_dict[self.last_change_id].config(bg="SystemButtonFace", fg="black")
            self.last_change_id = change_id
        if self.last_change_id is not None:
            self.change_button_dict[self.last_change_id].config(bg="green", fg="white")

    def get_prop_vector(self):
        return np.linspace(self.min_val.get(), self.max_val.get(), self.num_divs.get())

    def progress_update(self, progress_value):
        if progress_value >= 0:
            value = progress_value
        else:
            value = self.progress_bar['maximum']
        self.progress_bar['value'] = value
        self.update()

    def show_vp_analysis(self, devel=True):
        if devel:
            from importlib import reload
            import Selective.VariedPropAnalysis
            reload(Selective.VariedPropAnalysis)
        from Selective.VariedPropAnalysis import show_vp_analysis_impl
        prop_func = self.get_proportions_func()
        self.execute_and_monitor(show_vp_analysis_impl, self.num_divs.get(), prop_func=prop_func)

    def execute_and_monitor(self, impl_func, max_progress, **kwargs):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox

        ret = MessageBox.askokcancel("Confirmation",
                "This may take a few minutes.\n"
                + "Are you sure to proceed?",
                parent=self.parent)
        if not ret:
            return

        prop_func = kwargs.pop('prop_func', None)

        self.disable_buttons()
        self.progress_bar['maximum'] = max_progress
        self.progress_bar['value'] = 0
        self.update()

        if prop_func is not None:
            kwargs['prop_func'] = prop_func

        impl_func(self, **kwargs)
        self.enable_buttons()

    def reset_to_initstatus(self):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        ret = MessageBox.askokcancel("Confirmation",
                "This will reset the current decomposition to the initial state.\n"
                + "Are you sure to proceed?",
                parent=self.parent)
        if not ret:
            return

        self.editor.reset_current_frame()

    def on_right_click(self, event, devel=True):
        if devel:
            from importlib import reload
            import molass_legacy.KekLib.PopupMenuUtils
            reload(KekLib.PopupMenuUtils)
        from molass_legacy.KekLib.PopupMenuUtils import post_popup_menu
        self.create_popup_menu()
        post_popup_menu(self.popup_menu, self.parent, event)

    def create_popup_menu(self):
        if self.popup_menu is None:
            self.popup_menu = Tk.Menu(self, tearoff=0)
            self.popup_menu.add_command(label='Estimate Lognormal PSD with PDF', command=self.estimate_lognormal_psd)
            self.popup_menu.add_command(label='Estimate Lognormal PSD with CF', command=lambda: self.estimate_lognormal_psd(with_cf=True))
            self.popup_menu.add_command(label='CF Optimization Study', command=self.cf_optimization_study)
            self.popup_menu.add_command(label='Test Lognormalpore', command=self.test_lognormalpore)
            self.popup_menu.add_command(label='Test Tripore', command=self.test_tripore)

    def get_proportions_func(self, debug=False):
        code_text = self.code_text.get("1.0", Tk.END)
        return PROPORTIONS_.get_proportions_func(code_text, debug=debug)

    def change_num_components(self, devel=True):
        if devel:
            from importlib import reload
            import Selective.NumComponents
            reload(Selective.NumComponents)
        from Selective.NumComponents import change_num_components_impl
        if self.with_props.get() == 1:
            proportions = self.get_proportions_func()
            locals()['proportions'] = proportions      # this will define the proportions function which may be called in the eval below
            if USE_COMBOBOX_FOR_PROPS_TEXT:
                props_text = self.with_props_text_var.get()
            else:
                props_text = self.props_text.get("1.0", Tk.END)
            target_props = eval(props_text)
            if devel:
                print("props_text=", props_text)
                print("target_props=", target_props)
        else:
            target_props = None
        change_num_components_impl(self, target_props=target_props)
        self.update_button_status()

    def minimize_rdr(self, devel=True):
        if devel:
            from importlib import reload
            import Selective.RdrMinimizer
            reload(Selective.RdrMinimizer)
        from Selective.RdrMinimizer import try_rdr_minimization
        self.execute_and_monitor(try_rdr_minimization, 20)
    
    def minimize_kpd(self, devel=True):
        if devel:
            from importlib import reload
            import Selective.KpdMinimizer
            reload(Selective.KpdMinimizer)
        from Selective.KpdMinimizer import try_kpd_minimization
        self.execute_and_monitor(try_kpd_minimization, 20)

    def estimate_psd(self, devel=True):
        if devel:
            from importlib import reload
            import Models.Stochastic.TriporeColumn
            reload(Models.Stochastic.TriporeColumn)
        from molass_legacy.Models.Stochastic.TriporeColumn import estimate_psd_impl
        self.execute_and_monitor(estimate_psd_impl, 10)

    def estimate_lognormal_psd(self, with_cf=False, devel=True):
        if devel:
            from importlib import reload
            import Models.Stochastic.LognormalPoreColumn
            reload(Models.Stochastic.LognormalPoreColumn)
        from molass_legacy.Models.Stochastic.LognormalPoreColumn import estimate_lognormal_psd_impl
        self.execute_and_monitor(estimate_lognormal_psd_impl, 20, with_cf=with_cf)

    def cf_optimization_study(self, devel=True):
        if devel:
            from importlib import reload
            import CharFunc.CfOptimizationStudy
            reload(CharFunc.CfOptimizationStudy)
        from CharFunc.CfOptimizationStudy import cf_optimization_study_impl
        self.execute_and_monitor(cf_optimization_study_impl, 20)

    def test_lognormalpore(self, devel=True):
        if devel:
            from importlib import reload
            import Models.Stochastic.LognormalPore
            reload(Models.Stochastic.LognormalPore)
        from molass_legacy.Models.Stochastic.LognormalPore import lognormalpore_test_impl
        self.execute_and_monitor(lognormalpore_test_impl, 20)

    def test_tripore(self, devel=True):
        if devel:
            from importlib import reload
            import Models.Stochastic.Tripore
            reload(Models.Stochastic.Tripore)
        from molass_legacy.Models.Stochastic.Tripore import tripore_test_impl
        self.execute_and_monitor(tripore_test_impl, 20)

    def test_it(self):
        from importlib import reload
        import Models.Stochastic.Tripore
        reload(Models.Stochastic.Tripore)
        from molass_legacy.Models.Stochastic.Tripore import test_it_from_editor_frame
        print("test it")
        test_it_from_editor_frame(self.editor)

if __name__ == '__main__':
    import sys
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

    import seaborn
    seaborn.set()
    import molass_legacy.KekLib

    context = {}
    func_code = ("def proportions(p):\n"
                 "    return p, 1-p")
    exec(func_code, context)
    proportions = context['proportions']
    print(proportions(0.2))

    root = Tk.Tk()

    class DummyModel:
        def __init__(self):
            pass
        def is_traditional(self):
            return False

        def get_name(self):
            return "EGHA"

    class DummyEditor:
        def __init__(self):
            pass
        def get_current_params_array(self):
            return np.zeros((2,3))
        def bind(self, *args):
            pass
        def get_current_model(self):
            return DummyModel()

    frame = AdvancedFrame(root, DummyEditor(), bd=3, relief=Tk.RIDGE)
    frame.pack()

    root.mainloop()
