"""
    Optimizer.OptStrategyDialog.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy.Trimming import seem_to_be_manually_trimmed
from molass_legacy.KekLib.BraceWidgets import ClosingBrace
from .OptConstants import MIN_NUM_PEAKS, MAX_NUM_PEAKS
from .DefaultNumPeaks import get_default_num_peaks
from molass_legacy.Baseline.BaselineUtils import get_default_baseline_type
from .ElutionComposer import COMPOSER_CB_TEXT
from molass_legacy._MOLASS.Version import is_developing_version

BETA_RELEASE = get_setting("beta_release")
ENABLE_SEPARATE_FOULING = get_setting("enable_separate_fouling")
SHOW_RG_CURVE_ENTRY = False
SHOW_OTHER_OPTIONS = False
SIMPLE_EOII_OPTIONS = True
SHOW_FULL_OPTIONS = False
ENABLE_ADVANCED_SETTINGS = False
SHOW_OPTIMIZATION_STRATEGY_OPTION = False

DEFAULT_FUNC_ITEM = {
    0 : 'default_func_egh',
    1 : 'default_func_sdm',
    2 : 'default_func_lj_egh',
    3 : 'default_func_fd_emg',
    4 : 'default_func_rt_egh',
    5 : 'default_func_edm',
    }

MODEL_LIST = [  (0, "[%d] %s - EGH - Exponential-Gaussian Hybrid"),
                (2, "[%d] %s - Constrained EGH (Plate Theory)"),
                (3, "[%d] %s - Constrained EMG (Plate Theory)"),
                # (4, "[%d] %s - Constrained EMG (Approximation from Rate Theory Moments)"),
                (1, "[%d] %s - SDM - Stochastic DIspersive Model"),
                (5, "[%d] %s - EDM - Equilibrium Dispersive Model"),
                ]

MODEL_TO_FUNC = {
    0 : 'G0346',
    1 : 'G1100',
    5 : 'G2010',
}

class OptStrategyDialog(Dialog):
    def __init__(self, parent, sd, pre_recog=None, test_mode=False):
        self.parent = parent
        self.sd = sd
        self.pre_recog = pre_recog
        self.manually_trimmed = seem_to_be_manually_trimmed()
        self.default_num_peaks = get_default_num_peaks(sd)
        self.major_peak_index = None    # let the user give this info because there is no reliable index yet
        self.applied = False
        location = None if test_mode else "lower center"
        Dialog.__init__(self, parent, "Optimization Strategy", visible=False, location=location)

    def show(self):
        self._show()

    def body(self, body_frame, devel=True):
        if devel:
            from importlib import reload
            import molass_legacy.GuiParts.ColumnTypeSelector
            reload(molass_legacy.GuiParts.ColumnTypeSelector)
        from molass_legacy.Experiment.DataUtils import get_columntype
        from molass_legacy.GuiParts.ColumnTypeSelector import ColumnTypeSelector

        HasA_debug = False

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=50, pady=10)

        indent_width = 50

        # Exclusion Limit
        grid_row = 0
        label = Tk.Label(iframe, text="Exclusion Limit (which implies the SEC pore size)")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        grid_row += 1
        option_frame = Tk.Frame(iframe)
        option_frame.grid(row=grid_row, column=0, padx=indent_width, pady=5, sticky=Tk.W)

        columntype = get_columntype()
        self.default_excl_limit = columntype.excl_limit
        column_name = columntype.name
        numplates_pm = columntype.num_plates
        if numplates_pm is None:
            numplates_pm = round(get_setting('num_plates_pc')/0.3)

        self.el_option = Tk.IntVar()
        if column_name == 'Unknown':
            el_option = 1
        else:
            el_option = 0
        self.el_option.set(el_option)
        self.excl_limit = Tk.DoubleVar()
        self.excl_limit.set(self.default_excl_limit)
        self.column_name = Tk.StringVar()

        for k, cname in enumerate([ "use the default column value, i.e., %s (%g kDa)" % (column_name, self.default_excl_limit),
                                    "select from the column products list",
                                    "use the specified limit in the next entry: "]):
            button_frame = Tk.Frame(option_frame)
            rb = Tk.Radiobutton(button_frame, text=cname,
                        variable=self.el_option, value=k,
                        )
            rb.pack(side=Tk.LEFT)
            if k == 1:
                self.column_selector = ColumnTypeSelector(button_frame,
                                                          column_name_var=self.column_name,
                                                          excl_limit_var=self.excl_limit)
                self.column_selector.pack(side=Tk.LEFT)
            elif k == 2:
                self.el_entry  = Tk.Entry(button_frame, textvariable=self.excl_limit,
                                  justify=Tk.CENTER, width=6)
                self.el_entry.pack(side=Tk.LEFT)
                unit_label = Tk.Label(button_frame, text=" (kDa)")
                unit_label.pack(side=Tk.LEFT)
                space = Tk.Frame(button_frame, width=60)
                space.pack(side=Tk.LEFT)

            button_frame.grid(row=k, column=1, sticky=Tk.W )

        # Selected Column Display
        selected_column_frame = Tk.Frame(option_frame)
        selected_column_frame.grid(row=0, column=2, rowspan=3)

        brace = ClosingBrace(selected_column_frame, width=20, height=75)
        brace.grid(row=0, column=0)

        excl_limit_disp = Tk.Label(selected_column_frame, textvariable=self.excl_limit)
        excl_limit_disp.grid(row=0, column=1)
        label = Tk.Label(selected_column_frame, text="kDa")
        label.grid(row=0, column=2)
        mwrg_fig_button = Tk.Button(selected_column_frame, text="Show MwRg figure", command=self.show_mwrg_figure)
        mwrg_fig_button.grid(row=0, column=3, padx=5)

        # Tracers
        self.el_option.trace_add("write", self.el_option_tracer)
        self.el_option_tracer()
        self.excl_limit.trace_add("write", self.excl_limit_tracer)

        # Rg Curve Input
        if SHOW_RG_CURVE_ENTRY:
            grid_row += 1
            label = Tk.Label(iframe, text="Rg Curve Input")
            label.grid(row=grid_row, column=0, sticky=Tk.W)

            grid_row += 1
            button_frame = Tk.Frame(iframe)
            button_frame.grid(row=grid_row, column=0, padx=indent_width, pady=10, sticky=Tk.W)

            rg_curve_folder = get_setting("rg_curve_folder")
            self.rg_curve_input = Tk.IntVar()
            self.rg_curve_input.set(0 if rg_curve_folder is None else 1)
            btn_row = 0
            for k, cname in enumerate([ "calculate",
                                        "copy from the Rg curve folder below"]):
                rb = Tk.Radiobutton(button_frame, text=cname,
                            variable=self.rg_curve_input, value=k,
                            )
                rb.grid(row=k, column=0, sticky=Tk.W)

            grid_row += 1
            self.rg_curve_folder = Tk.StringVar()
            if rg_curve_folder is not None:
                self.rg_curve_folder.set(rg_curve_folder)
            entry_frame = Tk.Frame(iframe)
            entry_frame.grid(row=grid_row, column=0)
            space = Tk.Frame(entry_frame, width=indent_width + 30)
            space.grid(row=0, column=0)
            self.rg_folder_entry = FolderEntry(entry_frame, textvariable=self.rg_curve_folder, width=70,
                                                slimbutton=True,
                                                on_entry_cb=self.on_entry_rg_folder )
            self.rg_folder_entry.grid(row=0, column=1, sticky=Tk.W)

            space = Tk.Frame(entry_frame, height=10)
            space.grid(row=1, column=0)
        else:
            rg_curve_folder = get_setting("rg_curve_folder")
            self.rg_curve_input = Tk.IntVar()
            self.rg_curve_input.set(0 if rg_curve_folder is None else 1)
            self.rg_curve_folder = Tk.StringVar()
            if rg_curve_folder is not None:
                self.rg_curve_folder.set(rg_curve_folder)

        # Number of Components
        grid_row += 1
        label = Tk.Label(iframe, text="Number of Components")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        grid_row += 1
        option_frame = Tk.Frame(iframe)
        option_frame.grid(row=grid_row, column=0, padx=indent_width, pady=5, sticky=Tk.W)

        self.nc_option = Tk.IntVar()
        self.nc_option.set(0)
        self.num_peaks = Tk.IntVar()
        set_num_peaks = self.default_num_peaks
        self.num_peaks.set(set_num_peaks)

        btn_row = 0
        for k, cname in enumerate([ "use an automatically determined number",
                                    "use the specified number in the next box: "]):
            button_frame = Tk.Frame(option_frame)
            rb = Tk.Radiobutton(button_frame, text=cname,
                        variable=self.nc_option, value=k,
                        )
            rb.pack(side=Tk.LEFT)
            if k == 1:
                self.sbox  = Tk.Spinbox(button_frame, textvariable=self.num_peaks,
                                  from_=MIN_NUM_PEAKS, to=MAX_NUM_PEAKS, increment=1,
                                  justify=Tk.CENTER, width=6)
                self.sbox.pack(side=Tk.LEFT)

            button_frame.grid(row=btn_row+k, column=1, sticky=Tk.W )

        self.nc_option.trace_add("write", self.nc_option_tracer)
        self.nc_option_tracer()
        self.num_peaks.trace_add("write", self.num_peaks_tracer)

        # Elution Model
        grid_row += 1
        label = Tk.Label(iframe, text="Elution Model")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        grid_row += 1
        option_frame = Tk.Frame(iframe)
        option_frame.grid(row=grid_row, column=0, padx=indent_width, pady=5, sticky=Tk.W)

        elution_model = get_setting("elution_model")

        self.elution_model_gui = Tk.IntVar()
        self.elution_model_gui.set(elution_model)

        self.number_of_plates = Tk.DoubleVar()
        self.number_of_plates.set(numplates_pm)

        i = 0
        if SHOW_FULL_OPTIONS:
            model_list = MODEL_LIST
        else:
            model_list = [MODEL_LIST[0], MODEL_LIST[3], MODEL_LIST[4]]
        for k, cname_fmt in model_list:
            state = Tk.DISABLED if k == 4 else Tk.NORMAL
            # state = Tk.NORMAL
            cname = cname_fmt % (k, get_setting(DEFAULT_FUNC_ITEM[k]))
            rb = Tk.Radiobutton(option_frame, text=cname, variable=self.elution_model_gui, value=k, state=state)
            span = 3 if k == 4 else 2
            rb.grid(row=i, column=0, columnspan=span, sticky=Tk.W )

            i += 1

        if SHOW_FULL_OPTIONS:
            # Palte Number
            space = Tk.Frame(option_frame, width=20)
            space.grid(row=1, column=2)
            brace = ClosingBrace(option_frame, width=20, height=50)
            brace.grid(row=1, column=3, rowspan=2)

            pn_frame = Tk.Frame(option_frame, width=20)
            pn_frame.grid(row=1, column=4, rowspan=2)
            label = Tk.Label(pn_frame, text="Number of Plates: ")
            label.grid(row=0, column=0)
            self.num_plates_entry = Tk.Entry(pn_frame, textvariable=self.number_of_plates, width=8, justify=Tk.CENTER)
            self.num_plates_entry.grid(row=0, column=1)
            percol = Tk.Label(pn_frame, text="/m")
            percol.grid(row=0, column=2)

        # Trimming Strategy
        grid_row += 1
        label = Tk.Label(iframe, text="Trimming Strategy")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        grid_row += 1
        trimming_frame = Tk.Frame(iframe)
        trimming_frame.grid(row=grid_row, column=0, columnspan=2, padx=indent_width, pady=5, sticky=Tk.W)

        self.trimming_strategy = Tk.IntVar()
        self.trimming_strategy.set(1 if self.manually_trimmed else 2)

        for k, (strategy_text, v) in enumerate([
                                        ("apply automatic trimming", 2),
                                        ("apply traditional (or manually specified) trimming", 1),
                                        ]):
            rb = Tk.Radiobutton(trimming_frame, text=strategy_text,
                        variable=self.trimming_strategy, value=v,
                        )
            rb.grid(row=k, column=0, sticky=Tk.W )

        self.trimming_strategy.trace_add("write", self.trimming_strategy_tracer)

        state = Tk.DISABLED if BETA_RELEASE else Tk.NORMAL

        # Trimming Check Figure
        space = Tk.Frame(trimming_frame, width=10)
        space.grid(row=0, column=1)
        check_fig_frame = Tk.Frame(trimming_frame)
        check_fig_frame.grid(row=0, column=2, rowspan=2)
        brace = ClosingBrace(check_fig_frame, width=20, height=50)
        brace.grid(row=0, column=0)
        button  = Tk.Button(check_fig_frame, text="Check Figure", command=self.show_trimming_figure)
        button.grid(row=0, column=1, padx=50)

        # Option Grid Frame
        grid_row += 1
        grid_frame = Tk.Frame(iframe)
        grid_frame.grid(row=grid_row, column=0, padx=0, pady=5, sticky=Tk.W)
        for k in range(5):
            grid_frame.columnconfigure(k, weight=1)

        # Baseline Options
        gf_row = 0
        label = Tk.Label(grid_frame, text="Baseline Options")
        label.grid(row=gf_row, column=0, sticky=Tk.W)

        gf_row += 1
        space = Tk.Frame(grid_frame)
        space.grid(row=gf_row, column=0)

        self.unified_baseline_type = Tk.IntVar()
        baseline_type = get_default_baseline_type()
        # baseline_type = get_setting("unified_baseline_type")
        self.unified_baseline_type.set(baseline_type)

        if ENABLE_SEPARATE_FOULING:
            option_specs =  [  # (0, "No baseline separation"),
                            (1, "Linear"),
                            (2, "Linear + Uniform Fouling"),
                            (3, "Linear + Separate Fouling"),
                            ]
        else:
            option_specs = [  # (0, "No baseline separation"),
                            (1, "Linear"), 
                            (2, "Linear + Fouling")
                            ]
            
        for k, cname in (option_specs):
            state = Tk.DISABLED if k == 3 else Tk.NORMAL
            # option_frame.columnconfigure(k, weight=1)
            rb = Tk.Radiobutton(grid_frame, text=cname, variable=self.unified_baseline_type, value=k, state=state)
            rb.grid(row=gf_row, column=k, sticky=Tk.W )

        self.optimization_method = Tk.IntVar()
        self.optimization_method.set(get_setting("optimization_method"))

        gf_row += 1
        label = Tk.Label(grid_frame, text="Optimization Method")
        label.grid(row=gf_row, column=0, sticky=Tk.W)

        gf_row += 1
        option_specs = [
                        (0, "Basin-Hopping"),
                        (1, "Nested Sampling"),
                        (2, "Alternate BH → NS"),
                        (3, "Alternate NS → BH"),
                        # (2, "MCMC (emcee)        "),
                        # (3, "SMC (pyABC)"),
                        ]
        # activate = slice(None, None) if is_developing_version() else slice(0,2)
        activate = slice(0,2)
        for k, cname in (option_specs[activate]):
            # method_frame.columnconfigure(k, weight=1)
            rb = Tk.Radiobutton(grid_frame, text=cname, variable=self.optimization_method, value=k)
            rb.grid(row=gf_row, column=k+1, sticky=Tk.W)

        if SHOW_OPTIMIZATION_STRATEGY_OPTION:
            gf_row += 1
            label = Tk.Label(grid_frame, text="Optimization Strategy")
            label.grid(row=gf_row, column=0, sticky=Tk.W)

            self.optimization_strategy = Tk.IntVar()
            self.optimization_strategy.set(get_setting("optimization_strategy"))

            gf_row += 1

            option_specs = [
                            (0, "Standard     "),
                            (1, "Custom       "),
                            ]
            for k, cname in (option_specs[activate]):
                # method_frame.columnconfigure(k, weight=1)
                rb = Tk.Radiobutton(grid_frame, text=cname, variable=self.optimization_strategy, value=k)
                rb.grid(row=gf_row, column=k+1, sticky=Tk.W)

            self.editor_button = Tk.Button(grid_frame, text="Strategy Editor", command=self.show_strategy_editor, state=Tk.DISABLED)
            self.editor_button.grid(row=gf_row, column=3, sticky=Tk.W)

            self.optimization_strategy.trace_add("write", self.optimization_strategy_tracer)

        # option variables
        self.ratio_interpretation = Tk.IntVar()
        self.ratio_interpretation.set(get_setting("ratio_interpretation"))
        self.try_model_composing = Tk.IntVar()
        self.try_model_composing.set(get_setting('try_model_composing'))
        self.identification_allowance = Tk.DoubleVar()
        self.identification_allowance.set(get_setting("identification_allowance"))
        self.separate_eoii = Tk.IntVar()
        self.separate_eoii.set(get_setting("separate_eoii"))
        self.separate_eoii_type = Tk.IntVar()
        self.apply_sf_bounds = Tk.IntVar()
        self.apply_sf_bounds.set(get_setting("apply_sf_bounds"))
        self.sf_bound_ratio = Tk.DoubleVar()
        self.sf_bound_ratio.set(get_setting("sf_bound_ratio"))
        self.avoid_peak_fronting = Tk.IntVar()
        self.avoid_peak_fronting.set(get_setting("avoid_peak_fronting"))
        self.apply_rg_discreteness = Tk.IntVar()
        self.apply_rg_discreteness.set(get_setting("apply_rg_discreteness"))
        self.rg_discreteness_unit = Tk.DoubleVar()
        self.rg_discreteness_unit.set(get_setting("rg_discreteness_unit"))
        self.apply_mw_integrity = Tk.IntVar()
        self.apply_mw_integrity.set(get_setting("apply_mw_integrity"))
        self.mw_integer_ratios = Tk.StringVar()
        ratios = get_setting("mw_integer_ratios")
        if ratios is None:
            ratios = ""
        self.mw_integer_ratios.set(ratios)
        self.avoid_peak_fronting_cb = None
        self.try_model_composing_cb = None
        self.separate_eoii_flags = []
        for k in range(self.num_peaks.get()):
            self.separate_eoii_flags.append(Tk.IntVar())

        if ENABLE_ADVANCED_SETTINGS:
            # Advanced Settings
            grid_row += 1
            label = Tk.Label(iframe, text="Advanced Settings")
            label.grid(row=grid_row, column=0, sticky=Tk.W)

            grid_row += 1
            advanced_frame = Tk.Frame(iframe)
            advanced_frame.grid(row=grid_row, column=0, padx=indent_width, pady=5, sticky=Tk.W)

            frame_row = 0
            cb = Tk.Checkbutton(advanced_frame, text="apply ratio interpretation", variable=self.ratio_interpretation)
            cb.grid(row=frame_row, column=0, sticky=Tk.W)

        if SHOW_FULL_OPTIONS:
            frame_row += 1
            compose_frame = Tk.Frame(advanced_frame)
            compose_frame.grid(row=frame_row, column=0, columnspan=5, sticky=Tk.W)
            cb = Tk.Checkbutton(compose_frame, text=COMPOSER_CB_TEXT, variable=self.try_model_composing)
            cb.grid(row=0, column=0, sticky=Tk.W)
            self.try_model_composing_cb = cb
            space = Tk.Frame(compose_frame, width=25)
            space.grid(row=0, column=1)
            state = Tk.DISABLED
            label = Tk.Label(compose_frame, text="identification allowance: ", state=state)
            label.grid(row=0, column=2, sticky=Tk.W)
            self.identification_allowance_label = label
            entry = Tk.Entry(compose_frame, textvariable=self.identification_allowance, width=6, justify=Tk.CENTER, state=state)
            entry.grid(row=0, column=3, sticky=Tk.W)
            self.identification_allowance_entry = entry
            self.try_model_composing.trace_add("write", self.try_model_composing_tracer)

            frame_row += 1
            separate_eoii_frame = Tk.Frame(advanced_frame)
            separate_eoii_frame.grid(row=frame_row, column=0, columnspan=5, sticky=Tk.W)

            if SIMPLE_EOII_OPTIONS:
                cb = Tk.Checkbutton(separate_eoii_frame, text="separate effect of interparticle interactions", variable=self.separate_eoii)
                cb.grid(row=0, column=0, sticky=Tk.W)
                self.separate_eoii.trace_add("write", self.update_separate_eoii_simple)
                frame = Tk.Frame(separate_eoii_frame)
                frame.grid(row=1, column=0,  sticky=Tk.W, padx=40)
                cb = Tk.Checkbutton(frame, text="applying bounds", variable=self.apply_sf_bounds, state=state)
                cb.grid(row=0, column=0,  sticky=Tk.W)
                self.apply_sf_bounds_cb = cb
                label = Tk.Label(frame, text="with parameter: ", state=state)
                label.grid(row=0, column=1)
                entry = Tk.Entry(frame, textvariable=self.sf_bound_ratio, width=6, justify=Tk.CENTER, state=state)
                entry.grid(row=0, column=2, sticky=Tk.W)

                self.sf_bounds_dependents = [cb, label, entry]
                self.update_separate_eoii_simple()

            else:
                cb = Tk.Checkbutton(separate_eoii_frame, text="separate effect of interparticle interactions in either of the following ways", variable=self.separate_eoii)
                cb.grid(row=0, column=0, sticky=Tk.W)
                self.separate_eoii.trace_add("write", self.update_separate_eoii_flags)
                cb = Tk.Checkbutton(separate_eoii_frame, text="applying bounds", variable=self.apply_sf_bounds, state=state)
                cb.grid(row=0, column=1,  sticky=Tk.W)
                self.apply_sf_bounds_cb = cb

                frame_row += 1
                radio_button_frame = Tk.Frame(advanced_frame)
                radio_button_frame.grid(row=frame_row, column=0, columnspan=5, sticky=Tk.W, padx=40)
                self.separate_eoii_rbs = []
                rb = Tk.Radiobutton(radio_button_frame, text="treating the total sum of components as a single cause elution", variable=self.separate_eoii_type, value=1)
                rb.grid(row=0, column=0, sticky=Tk.W)
                self.separate_eoii_rbs.append(rb)
                self.separate_eoii_type.trace_add("write", self.separate_eoii_type_tracer)

                label_flags_frame = Tk.Frame(radio_button_frame)
                label_flags_frame.grid(row=1, column=0, sticky=Tk.W)
                self.label_flags_frame = label_flags_frame
                rb = Tk.Radiobutton(label_flags_frame, text="separately in the order of component peak positions:", variable=self.separate_eoii_type, value=2)
                rb.grid(row=0, column=0)
                self.separate_eoii_rbs.append(rb)

                # construct flags_frame separately to avoid strange behavior in cases with less than three checkbuttons
                flags_frame = Tk.Frame(label_flags_frame)
                flags_frame.grid(row=0, column=1)
                self.flags_frame = flags_frame

                self.separate_eoii_cbs = []         # needed for the destroying
                self.update_separate_eoii_flags()

            frame_row += 1
            cb = Tk.Checkbutton(advanced_frame, text="avoid peak fronting", variable=self.avoid_peak_fronting)
            cb.grid(row=frame_row, column=0, sticky=Tk.W)
            self.avoid_peak_fronting_cb = cb

            frame_row += 1
            cb = Tk.Checkbutton(advanced_frame, text="apply Rg-discreteness unit", variable=self.apply_rg_discreteness)
            cb.grid(row=frame_row, column=0, sticky=Tk.W)
            entry_frame = Tk.Frame(advanced_frame)
            entry_frame.grid(row=frame_row, column=1, sticky=Tk.W)
            self.discreteness_unit_entry = Tk.Entry(entry_frame, textvariable=self.rg_discreteness_unit, width=5, justify=Tk.CENTER)
            self.discreteness_unit_entry.grid(row=0, column=0, sticky=Tk.W)
            unit_label = Tk.Label(entry_frame, text="Å")
            unit_label.grid(row=0, column=1, sticky=Tk.W)
            self.apply_rg_discreteness_tracer()
            self.apply_rg_discreteness.trace_add("write", self.apply_rg_discreteness_tracer)

            frame_row += 1
            cb = Tk.Checkbutton(advanced_frame, text="apply MW integer ratios", variable=self.apply_mw_integrity)
            cb.grid(row=frame_row, column=0, sticky=Tk.W)
            entry_frame = Tk.Frame(advanced_frame)
            entry_frame.grid(row=frame_row, column=1, sticky=Tk.W)
            self.mw_integer_ratios_entry = Tk.Entry(entry_frame, textvariable=self.mw_integer_ratios, width=10, justify=Tk.CENTER)
            self.mw_integer_ratios_entry.grid(row=0, column=0, sticky=Tk.W)
            guide_message = Tk.Label(entry_frame, text="specify the ratios like [2,1] if you are confident, otherwise leave it empty")
            guide_message.grid(row=0, column=1, sticky=Tk.W, padx=2)
            self.apply_mw_integrity_tracer()
            self.apply_mw_integrity.trace_add("write", self.apply_mw_integrity_tracer)

        # Other Options
        self.strict_sec_penalty = Tk.IntVar()
        self.strict_sec_penalty.set(1)
        self.correction = Tk.IntVar()
        self.correction.set(1)
        self.uv_basemodel = Tk.IntVar()
        self.uv_basemodel.set(1)

        if SHOW_OTHER_OPTIONS or SHOW_FULL_OPTIONS:
            grid_row += 1
            label = Tk.Label(iframe, text="Other Options")
            label.grid(row=grid_row, column=0, sticky=Tk.W)

            grid_row += 1
            cb = Tk.Checkbutton( iframe, text="apply strict size exclusion penalty in the optimization",
                                    variable=self.strict_sec_penalty, state=state )
            cb.grid(row=grid_row, column=0, sticky=Tk.W, padx=indent_width)

            grid_row += 1
            cb = Tk.Checkbutton( iframe, text="apply baseline correction to X-ray data before the optimization",
                                    variable=self.correction, state=state )
            cb.grid(row=grid_row, column=0, sticky=Tk.W, padx=indent_width)

            grid_row += 1
            cb = Tk.Checkbutton( iframe, text="use the Advanced UV Baseline Model",
                                    variable=self.uv_basemodel, state=Tk.DISABLED )     # don't change uv_basemodel == 1 until the non-advanced baseline gets completed
            cb.grid(row=grid_row, column=0, sticky=Tk.W, padx=indent_width)
            self.uv_basemodel_cb = cb

        # Tracers which concers multiple parts
        self.elution_model_gui_tracer()
        self.elution_model_gui.trace_add("write", self.elution_model_gui_tracer)

    def get_model_name(self):
        from molass_legacy.Optimizer.OptimizerUtils import get_model_name
        elution_model = self.elution_model_gui.get()
        func = MODEL_TO_FUNC[elution_model]
        return get_model_name(func)

    def on_entry_rg_folder(self):
        pass

    def el_option_tracer(self, *args):
        el_option = self.el_option.get()
        if el_option == 0:
            self.excl_limit.set(self.default_excl_limit)

        def get_state(state):
            return Tk.NORMAL if state else Tk.DISABLED

        self.column_selector.config(state=get_state(el_option == 1))
        self.el_entry.config(state=get_state(el_option == 2))

    def excl_limit_tracer(self, *args):
        el_option = self.el_option.get()
        try:
            self.excl_limit_disp_var.set("%g kDa" % self.excl_limit.get())
        except:
            pass

    def elution_model_gui_tracer(self, *args):
        elution_model_gui = self.elution_model_gui.get()

        if SHOW_FULL_OPTIONS:
            state = Tk.NORMAL if elution_model_gui in [2, 3] else Tk.DISABLED
            for w in [self.num_plates_entry]:
                w.config(state=state)

        if self.avoid_peak_fronting_cb is not None:
            if elution_model_gui in [0, 2, 3]:
                self.avoid_peak_fronting_cb.config(state=Tk.NORMAL)
            else:
                self.avoid_peak_fronting.set(0)
                self.avoid_peak_fronting_cb.config(state=Tk.DISABLED)

        if self.try_model_composing_cb is not None:
            if elution_model_gui in [0, 2]:
                state = Tk.NORMAL
            else:
                state = Tk.DISABLED
                self.try_model_composing.set(0)
            self.try_model_composing_cb.config(state=state)

    def trimming_strategy_tracer(self, *args):
        strategy = self.trimming_strategy.get()
        if self.manually_trimmed and strategy == 2:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox

            MessageBox.showwarning( "Manual Trimming Cleared Warning",
                    "Manual trimming info seems to exist.\n"
                    "Changing to automatic trimming will clear the info.\n"
                    "Revert this option if it is not desired.",
                    parent=self )

        if False:
            if strategy == 2:
                state = Tk.NORMAL
                advanced = 1
            else:
                state = Tk.DISABLED
                advanced = 0
            self.uv_basemodel_cb.config(state=state)
            self.uv_basemodel.set(advanced)

    def nc_option_tracer(self, *args):
        nc_option = self.nc_option.get()
        state = Tk.DISABLED if nc_option == 0 else Tk.NORMAL
        self.sbox.config(state=state)

    def num_peaks_tracer(self, *args):
        if not SIMPLE_EOII_OPTIONS:
            self.update_separate_eoii_flags()

    def try_model_composing_tracer(self, *args):
        try_model_composing = self.try_model_composing.get()
        state = Tk.NORMAL if try_model_composing else Tk.DISABLED
        for w in (  self.identification_allowance_label,
                    self.identification_allowance_entry,
                    ):
            w.config(state=state)

    def update_separate_eoii_simple(self, *args):
        if self.separate_eoii.get():
            state = Tk.NORMAL
            type_value = 1
        else:
            state = Tk.DISABLED
            type_value = 0
        self.separate_eoii_type.set(type_value)
        for w in self.sf_bounds_dependents:
            w.config(state=state)

    def update_separate_eoii_flags(self, *args):
        num_peaks = self.num_peaks.get()
        for cb in self.separate_eoii_cbs:
            cb.destroy()

        if self.separate_eoii.get() == 1:
            state = Tk.NORMAL
            self.separate_eoii_type.set(1)
        else:
            state = Tk.DISABLED
            self.separate_eoii_type.set(0)

        for k, rb in enumerate(self.separate_eoii_rbs):
            if k == 0:
                rb.config(state=state)
            else:
                # not yet supported
                rb.config(state=Tk.DISABLED)

        # self.flags_label.config(state=state)
        self.separate_eoii_flags = []
        self.separate_eoii_cbs = []
        for k in range(num_peaks):
            print([k], k)
            flag_value = Tk.IntVar()
            flag_value.set(1 if k == self.major_peak_index else 0)      # note that self.major_peak_index may be None
            self.separate_eoii_flags.append(flag_value)
            cb = Tk.Checkbutton(self.flags_frame, text="", variable=flag_value, state=Tk.DISABLED)  # not yet supported
            cb.grid(row=0, column=k)
            self.separate_eoii_cbs.append(cb)

    def separate_eoii_type_tracer(self, *args):
        separate_eoii_type = self.separate_eoii_type.get()
        state = Tk.NORMAL if separate_eoii_type > 0 else Tk.DISABLED
        self.apply_sf_bounds_cb.config(state=state)
        if separate_eoii_type < 2:
            return

        state = Tk.NORMAL if separate_eoii_type == 2 else Tk.DISABLED
        for cb in self.separate_eoii_cbs:
            cb.config(state=state)

    def apply_rg_discreteness_tracer(self, *args):
        apply_rg_discreteness = self.apply_rg_discreteness.get()
        state = Tk.DISABLED if apply_rg_discreteness == 0 else Tk.NORMAL
        self.discreteness_unit_entry.config(state=state)

    def apply_mw_integrity_tracer(self, *args):
        apply_mw_integrity = self.apply_mw_integrity.get()
        state = Tk.DISABLED if apply_mw_integrity == 0 else Tk.NORMAL
        self.mw_integer_ratios_entry.config(state=state)

    def optimization_strategy_tracer(self, *args):
        strategy = self.optimization_strategy.get()
        state = Tk.NORMAL if strategy == 1 else Tk.DISABLED
        self.editor_button.config(state=state)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X, padx=20)

        for j in range(2):
            box.columnconfigure(j, weight=1)

        w = Tk.Button(box, text="◀ Cancel", width=10, command=self.cancel)
        col = 0
        w.grid(row=0, column=col, pady=10)
        self.cancel_btn = w

        col += 1
        w = Tk.Button(box, text="▶ Proceed", width=12, command=self.ok)
        w.grid(row=0, column=col, pady=10)
        self.proceed_btn = w

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def validate(self):
        ret = 1
        if self.apply_mw_integrity.get():
            ratios = self.mw_integer_ratios.get()
            if ratios == "":
                pass
            else:
                try:
                    self.mw_integer_ratios_eval = eval(ratios)
                    assert type(self.mw_integer_ratios_eval) is list
                except Exception as exc:
                    import molass_legacy.KekLib.CustomMessageBox as MessageBox
                    MessageBox.showerror( "Value Error", "mw_integer_ratios error:" + str(exc), parent=self)
                    ret = 0
        return ret

    def apply(self):
        from molass_legacy._MOLASS.SerialSettings import set_setting
        from molass_legacy.Global.V2Init import update_sec_settings

        self.applied = True

        set_setting("uv_basemodel", self.uv_basemodel.get())
        elution_model = self.elution_model_gui.get()
        set_setting("elution_model", elution_model)
        item_name = DEFAULT_FUNC_ITEM[elution_model]
        func_code = get_setting(item_name)
        set_setting("default_objective_func", func_code)

        el_option = self.el_option.get()
        column_name = self.column_name.get()
        excl_limit = self.excl_limit.get()
        num_plates_pm = self.number_of_plates.get()
        update_sec_settings(el_option, column_name, excl_limit, num_plates_pm)

        set_setting("unified_baseline_type", self.unified_baseline_type.get())

        # Advanced Settings
        set_setting("ratio_interpretation", self.ratio_interpretation.get())
        set_setting("optimization_method",self.optimization_method.get())
        set_setting("try_model_composing", self.try_model_composing.get())
        set_setting("separate_eoii", self.separate_eoii.get())
        set_setting("separate_eoii_type", self.separate_eoii_type.get())
        separate_eoii_flags = []
        for flag in self.separate_eoii_flags:
            separate_eoii_flags.append(flag.get())
        set_setting("apply_sf_bounds", self.apply_sf_bounds.get())
        set_setting("sf_bound_ratio", self.sf_bound_ratio.get())
        set_setting("separate_eoii_flags", separate_eoii_flags)
        set_setting("avoid_peak_fronting", self.avoid_peak_fronting.get())
        set_setting("apply_rg_discreteness", self.apply_rg_discreteness.get())
        set_setting("rg_discreteness_unit", self.rg_discreteness_unit.get())
        set_setting("apply_mw_integrity", self.apply_mw_integrity.get())
        ratios = self.mw_integer_ratios.get()
        if ratios == "":
            ratios = None
        else:
            ratios = self.mw_integer_ratios_eval   # this value has been validated
        set_setting("mw_integer_ratios", ratios)
        set_setting("identification_allowance", self.identification_allowance.get())

        if self.trimming_strategy.get() == 2:
            set_setting("uv_restrict_list", get_setting("uv_restrict_copy"))
            set_setting("xr_restrict_list", get_setting("xr_restrict_copy"))
            set_setting("manually_trimmed", False)

    def get_num_peaks(self):
        if self.nc_option.get() == 1:
            num_peaks = self.num_peaks.get()
        else:
            # num_peaks = None      # revive this if a better estimation using Rg curve become availabe
            num_peaks = self.num_peaks.get()
        return num_peaks

    def get_options(self):
        strict_sec_penalty = self.strict_sec_penalty.get()
        correction = self.correction.get()
        trimming = self.trimming_strategy.get()
        uv_basemodel = self.uv_basemodel.get()
        unified_baseline_type = self.unified_baseline_type.get()
        return strict_sec_penalty==1, correction, trimming, unified_baseline_type

    def show_mwrg_figure(self, debug=True):
        if debug:
            from importlib import reload
            import SecTheory.MwRgFigure
            reload(SecTheory.MwRgFigure)
        from SecTheory.MwRgFigure import MwRgFigure
        fig = MwRgFigure(self, self.excl_limit.get())
        fig.show()

    def show_trimming_figure(self, debug=True):
        if debug:
            from importlib import reload
            import Trimming.TrimmingResult
            reload(Trimming.TrimmingResult)
        from molass_legacy.Trimming.TrimmingResult import TrimmingResultDialog
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition

        if self.pre_recog is None:
            # doing this twice causes an error. why?
            self.pre_recog = PreliminaryRecognition(self.sd)

        dialog = TrimmingResultDialog(self, self.pre_recog)
        dialog.show()

    def show_strategy_editor(self, debug=True):
        if debug:
            from importlib import reload
            import Optimizer.StrategyEditor
            reload(Optimizer.StrategyEditor)
        from molass_legacy.Optimizer.StrategyEditor import StrategyEditor

        editor = StrategyEditor(self)
        editor.show()