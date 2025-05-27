"""
    PreviewButtonFrame.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import sys
import logging
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from .PreviewData import PreviewData, PreviewOptions
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.LRF.PnoScdMap import PnoScdMap
DEBUG_SD = False
if DEBUG_SD:
    from molass_legacy.SerialAnalyzer.SdDebugger import SdDebugger

ADDITIONAL_CONSTRAINTS = False
DEVELOP_MODE = sys.executable.find("pythonw") < 0
# DEVELOP_MODE = False
ELUTION_DATATYPE_NAMES = ["Xray elution model", "Xray measured data", "UV elution model", "mapped UV data"]

class PreviewButtonFrame(Tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.dialog = kwargs.pop('dialog', None)
        self.use_elution_models =  kwargs.pop('use_elution_models', None)
        self.judge_holder =  kwargs.pop('judge_holder', None)
        conc_opts = kwargs.pop('conc_opts', False)
        self.enable_conc_opts = conc_opts and get_setting('enable_conc_opts')
        self.enable_conctype_change =  get_setting('enable_conctype_change')
        self.editor =  kwargs.pop('editor', None)
        self.pre_proc =  kwargs.pop('pre_proc', None)
        self.post_proc =  kwargs.pop('post_proc', None)
        self.layout_check_only = kwargs.pop('layout_check_only', False)
        self.mapper = self.dialog.mapper
        self.is_microfluidic = self.mapper is None
        self.allow_rank_variation = get_setting('allow_rank_variation')
        self.extended_conc_dep = get_setting('extended_conc_dep')

        Tk.LabelFrame.__init__(self, parent, text="Preview Options", labelanchor=Tk.N, *args, **kwargs)

        # concentration_datatype
        conc_datatype = get_setting("concentration_datatype")
        self.concentration_datatype = Tk.IntVar()
        if self.use_elution_models:
            rb_dtypes = [0, 1, 2, 3]
            if conc_datatype is None:
                datatype = 2
            else:
                if self.enable_conctype_change:
                    datatype = conc_datatype
                else:
                    datatype = 2
        else:
            rb_dtypes = [1, 3]
            if conc_datatype is None:
                datatype = 3
            else:
                if self.enable_conctype_change:
                    if conc_datatype in rb_dtypes:
                        datatype = conc_datatype
                    else:
                        datatype = 1 if conc_datatype < 2 else 3
                else:
                    datatype = 3
        self.concentration_datatype.set(datatype)

        if self.layout_check_only:
            self.main_peak_rank = 1
        else:
            cnv_ranges = self.get_cnv_ranges()
            self.pno_map = PnoScdMap(self.dialog.sd, cnv_ranges)
            self.main_peak_rank = self.pno_map.get_main_peak_rank()

        # 
        iframe = Tk.Frame(self)
        iframe.pack(padx=0, pady=10)

        self.widgets = []

        grid_row = 0
        grid_col = 0

        self.rank_control = Tk.IntVar()
        rank_control = get_setting( 'rank_control' )
        self.rank_control.set(rank_control)
        cb = Tk.Checkbutton(iframe, text="Automatic Rank Control", variable=self.rank_control )
        cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=10 )

        self.rc_dependents = []
        manual_state = Tk.DISABLED if rank_control else Tk.NORMAL

        grid_row += 1
        self.conc_dependence = Tk.IntVar()
        self.set_conc_depend()
        dependence_frame = Tk.Frame(iframe)
        label = Tk.Label(dependence_frame, text="Conc. Dependency", state=manual_state)
        label.pack(side=Tk.LEFT)
        self.rc_dependents.append(label)
        max_dependence = 6 if self.extended_conc_dep else 2
        spinbox = Tk.Spinbox( dependence_frame, textvariable=self.conc_dependence,
                                from_=1, to=max_dependence, increment=1,
                                justify=Tk.CENTER, width=6, state=manual_state)

        spinbox.pack(side=Tk.LEFT)
        self.rc_dependents.append(spinbox)
        dependence_frame.grid(row=grid_row, column=grid_col, sticky=Tk.W, padx=10 )
        self.widgets.append(spinbox)

        if self.is_microfluidic:
            grid_col += 1
            self.denat_dependent = Tk.IntVar()
            self.denat_dependent.set(get_setting('denat_dependent'))
            cb = Tk.Checkbutton(iframe, text="denat. dependent", variable=self.denat_dependent, state=manual_state)
            cb.grid( row=grid_row, column=grid_col, sticky=Tk.W, padx=10 )
            self.rc_dependents.append(cb)
            self.denat_dependent_cb = cb
            self.widgets.append(cb)

        if self.enable_conc_opts:
            grid_col += 1
            self.conc_opts_button = btn = Tk.Button( iframe, text='Conc. Options', command=self.show_conc_opts_dialog)
            btn.grid( row=grid_row, column=grid_col, sticky=Tk.W, padx=20 )
            self.widgets.append(btn)

        grid_col += 1
        rowspan_ = 7 if self.allow_rank_variation else 6
        self.show_zx_preview_button = btn = Tk.Button( iframe, text='Preview', command=self.show_zx_preview )
        btn.grid( row=0, rowspan=rowspan_, column=2, padx=30, sticky=Tk.E )
        self.widgets.append(btn)

        if False:
            self.test_button = btn = Tk.Button(iframe, text='Test', command=self.do_devel_test)
            btn.grid(row=0, rowspan=rowspan_, column=3, padx=10)

        grid_row += 1
        self.svd_reconstruct_dependents = []
        if self.allow_rank_variation:
            svd_frame = Tk.Frame(iframe)
            svd_frame.grid( row=grid_row, column=0, columnspan=2, sticky=Tk.W)
            self.svd_reconstruct = Tk.IntVar()
            self.svd_reconstruct.set( get_setting( 'svd_reconstruct' ) )
            cb = Tk.Checkbutton(svd_frame, text="SVD Reconstruction", variable=self.svd_reconstruct, state=manual_state )
            cb.grid( row=0, column=0, sticky=Tk.W, padx=10 )
            self.rc_dependents.append(cb)
            self.widgets.append(cb)

            self.rank_increment = Tk.IntVar()
            self.rank_increment.set( get_setting( 'rank_increment' ) )
            additional_rank_frame = Tk.Frame(svd_frame)
            label = Tk.Label(additional_rank_frame, text="Rank Increment", state=manual_state)
            label.pack(side=Tk.LEFT)
            self.rc_dependents.append(label)
            self.svd_reconstruct_dependents.append(label)
            self.ar_spinbox = spinbox = Tk.Spinbox(additional_rank_frame, textvariable=self.rank_increment,
                                    from_=0, to=5, increment=1,
                                    justify=Tk.CENTER, width=6, state=manual_state )

            spinbox.pack(side=Tk.LEFT)
            self.rc_dependents.append(spinbox)
            additional_rank_frame.grid(row=0, column=1, sticky=Tk.W, padx=10 )
            self.svd_reconstruct_dependents.append(spinbox)

        enable_new_features = get_setting('enable_new_features')

        # LRF Bound Correction
        self.lrf_bound_correction = Tk.IntVar()
        self.lrf_bound_correction.set(get_setting('lrf_bound_correction'))
        if enable_new_features:
            grid_row += 1
            cb = Tk.Checkbutton( iframe, text="LRF Bound Correction", variable=self.lrf_bound_correction )
            cb.grid( row=grid_row, column=0, sticky=Tk.W, padx=10 )
            self.lrf_bound_correction_cb = cb

        grid_row += 1
        self.force_proportions = Tk.IntVar()
        self.proportions_text = Tk.StringVar()
        self.proportions_text.set("0.2, 0.8")

        proportions_frame = Tk.Frame(iframe)
        proportions_frame.grid(row=grid_row, column=0, columnspan=2, sticky=Tk.W)
        state = Tk.DISABLED
        cb = Tk.Checkbutton(proportions_frame, text="Force Proportions to", variable=self.force_proportions, state=state)
        cb.grid(row=0, column=0, sticky=Tk.W, padx=10)
        self.proportions_entry = Tk.Entry(proportions_frame, textvariable=self.proportions_text, width=10, justify=Tk.CENTER, state=state)
        self.proportions_entry.grid(row=0, column=1, sticky=Tk.W, padx=10)

        # Data Selection Buttons
        # note that concentration_datatype has been created earlier above

        if self.enable_conctype_change:
            grid_row += 1
            label = Tk.Label(iframe, text="Concetration Datatype Selection:")
            label.grid(row=grid_row, column=0, padx=10, sticky=Tk.W)

            grid_row += 1
            frame = Tk.Frame(iframe)
            frame.grid(row=grid_row, column=0, columnspan=4, padx=20, sticky=Tk.W)

            for k, value in enumerate(rb_dtypes):
                text = ELUTION_DATATYPE_NAMES[value]
                rb = Tk.Radiobutton(frame, variable=self.concentration_datatype, text=text, value=value)
                d, r = divmod(k, 2)
                rb.grid(row=d, column=r, padx=10, sticky=Tk.W)

        if ADDITIONAL_CONSTRAINTS:
            grid_row += 1
            grid_col = 0
            self.aq_smoothness = Tk.IntVar()
            # this value is set in self.update_aq_smoothness
            self.aq_smoothness.set( get_setting( 'aq_smoothness' ) )
            cb = Tk.Checkbutton( iframe, text="Smoothness", variable=self.aq_smoothness, state=manual_state )
            cb.grid( row=grid_row, column=grid_col, sticky=Tk.W, padx=10 )
            self.rc_dependents.append(cb)
            self.widgets.append(cb)

            grid_col += 1
            self.aq_positivity = Tk.IntVar()
            self.aq_positivity.set( get_setting( 'aq_positivity' ) )
            cb = Tk.Checkbutton( iframe, text="A(q) Positivity", variable=self.aq_positivity, state=manual_state )
            cb.grid( row=grid_row, column=grid_col, sticky=Tk.W, padx=20 )
            self.rc_dependents.append(cb)
            self.widgets.append(cb)

        self.rank_control_tracer()
        self.svd_reconstruct.trace("w", self.rank_control_tracer)
        self.rank_control.trace("w", self.rank_control_tracer)

    def update_modelframe(self):
        self.editor = self.dialog.get_current_frame()

    def set_conc_depend(self, revive_auto_rank=False):
        if self.is_microfluidic:
            cd = 1
        else:
            cd = get_setting('conc_dependence')
            if cd is None or revive_auto_rank:
                cd = self.main_peak_rank
        self.conc_dependence.set(cd)

    def rank_control_tracer(self, *args):
        rank_control = self.rank_control.get()
        state = Tk.DISABLED if rank_control else Tk.NORMAL
        for w in self.rc_dependents:
            w.config(state=state)
        state = Tk.NORMAL if not rank_control and self.svd_reconstruct.get() else Tk.DISABLED
        for w in self.svd_reconstruct_dependents:
            w.config(state=state)
        if rank_control:
            # reset to default which has been established before the init of this frame
            self.set_conc_depend(revive_auto_rank=True)

    def get_cnv_ranges(self):
        # task: reduce redundancy with get_preview_data
        from molass_legacy.DataStructure.AnalysisRangeInfo import convert_to_paired_ranges

        paired_ranges = self.dialog.make_range_info(self.concentration_datatype.get())
        ret = convert_to_paired_ranges(paired_ranges)
        return ret[0]

    def get_preview_data(self, with_update=True):
        use_mtd_conc = get_setting('use_mtd_conc')

        if with_update:
            self.update_settings(update_memorized=False)    # False to avoid recursive calls to update_settings

        if use_mtd_conc:
            decomp_info = self.dialog.get_decomp_info()
            pdata = PreviewData(xdata=self.dialog.xdata, decomp_info=decomp_info, slice_=self.dialog.slice_)
        else:
            mapper = self.mapper
            sd_ = self.get_desired_serial_data(mapper)    # 
            # sd_ = self.dialog.sd
            paired_ranges = self.dialog.make_range_info(self.concentration_datatype.get())
            pdata = PreviewData(sd=sd_, mapper=mapper, judge_holder=self.judge_holder, paired_ranges=paired_ranges)

        """
        TODO: other params should be handled in the same way
        """
        if ADDITIONAL_CONSTRAINTS:
            aq_smoothness = self.aq_smoothness.get() == 1
            aq_positivity = self.aq_positivity.get() == 1
        else:
            aq_smoothness = False
            aq_positivity = False

        concentration_datatype = self.concentration_datatype.get()
        use_elution_models = concentration_datatype in [0, 2]
        conc_depend = self.get_conc_depend()
        popts = PreviewOptions(conc_depend, aq_smoothness, aq_positivity, use_elution_models)

        return pdata, popts

    def memorize_preview_data(self):
        pdata, popts = self.get_preview_data(with_update=False)
        set_setting('preview_params', (pdata, popts))

    def show_zx_preview(self, use_pool=True):
        if self.pre_proc is not None:
            self.pre_proc()

        self.update_settings()
        pdata, popts = get_setting('preview_params')

        if use_pool:
            self.show_zx_preview_using_pool(pdata, popts)
        else:
            from PreviewController import PreviewController
            self.preview_ctl = PreviewController(dialog=self.dialog, editor=self.editor)
            self.preview_ctl.run_solver(self.dialog, pdata, popts)
            if self.preview_ctl.ok():
                self.preview_ctl.show_dialog()
                self.update_widgets()
                if self.preview_ctl.dialog.applied:
                    self.update_settings()
            else:
                # TODO
                pass

        if self.post_proc is not None:
            self.post_proc()

    def get_last_change_id_from_advanced_frame(self):
        return self.dialog.advanced_frame.get_last_change_id()

    def show_zx_preview_using_pool(self, pdata, popts, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.LRF.LrfResultPool
            reload(molass_legacy.LRF.LrfResultPool)       
        from molass_legacy.LRF.LrfResultPool import LrfResultPool
        self.pool = LrfResultPool(pdata, popts)
        self.pool.run_solver(self.dialog)
        last_change_id = self.get_last_change_id_from_advanced_frame()
        self.pool.show_dialog(self.dialog.parent, last_change_id=last_change_id)

    def get_desired_serial_data(self, mapper):
        try:
            # try: to avoid errors like
            # _tkinter.TclError: invalid command name ".!decompeditordialog9.!frame2.!frame2.!previewbuttonframe"
            self.config(cursor='wait')
            self.update()
        except:
            pass

        sd = self.dialog.sd
        sd_copy = sd.get_exec_copy( mapper )

        if DEBUG_SD:
            debugger =SdDebugger()
            debugger.save_info(sd_copy)
        mapped_info = mapper.get_mapped_info()
        if mapped_info.needs_sd_corrcetion():
            sd_copy.apply_baseline_correction(mapped_info)
            if DEBUG_SD:
                debugger.save_info(sd_copy)
        sd_copy.absorbance = sd.absorbance    # exec_copy does not have absorbance

        try:
            self.config(cursor='')
        except:
            pass

        return sd_copy

    def config_states_alternative( self, state ):
        ignore_bq_list = get_setting('ignore_bq_list')
        for k, w in enumerate(self.widgets):
            if k == 0 and ignore_bq_list is not None:
                state_ = Tk.DISABLED
            else:
                state_ = state
            w.config( state=state_ )

    def config_states( self, state ):
        ignore_bq_list = get_setting('ignore_bq_list')
        for k, w in enumerate(self.widgets):
            w.config( state=state )

    def get_conc_depend(self):
        # 
        return self.conc_dependence.get()

    def update_widgets( self ):
        # TODO: 

        if ADDITIONAL_CONSTRAINTS:
            self.aq_smoothness.set( get_setting( 'aq_smoothness' ) )
            self.aq_positivity.set( get_setting( 'aq_positivity' ) )
        self.conc_dependence.set( get_setting( 'conc_dependence' ) )
        self.update()

    def update_settings(self, update_memorized=True):

        if hasattr(self.editor, "model_name"):
            preview_model = self.editor.model_name
        else:
            preview_model = None
        set_setting('preview_model', preview_model)

        rank_control = self.rank_control.get()
        set_setting('rank_control', rank_control)

        if ADDITIONAL_CONSTRAINTS:
            set_setting('aq_smoothness', self.aq_smoothness.get())
            set_setting('aq_positivity', self.aq_positivity.get())

        conc_dependence = self.conc_dependence.get()
        set_setting('conc_dependence', conc_dependence )
        self.logger.info('conc_dependence: %d' % conc_dependence)

        if self.is_microfluidic:
            denat_dependent = self.denat_dependent.get()
            set_setting('denat_dependent', denat_dependent)
            ignore_all_bqs = 1 - denat_dependent
        else:
            if rank_control:
                ignore_all_bqs = 0
            else:
                ignore_all_bqs = 1 if conc_dependence == 1 else 0
        set_setting('ignore_all_bqs', ignore_all_bqs)

        if self.allow_rank_variation:
            set_setting('svd_reconstruct',  self.svd_reconstruct.get())
            set_setting('rank_increment',   self.rank_increment.get())

        set_setting('lrf_bound_correction', self.lrf_bound_correction.get())
        set_setting('concentration_datatype', self.concentration_datatype.get())

        if update_memorized:
            self.memorize_preview_data()

    def show_conc_opts_dialog(self):
        from ConcOptsDialog import ConcOptsDialog
        pranges = self.editor.make_restorable_ranges()
        dialog = ConcOptsDialog(self.dialog, self.dialog, pranges)
        dialog.show()

    def set_guard_for_three_state_model(self):
        self.denat_dependent.set(0)
        self.denat_dependent_cb.config(state=Tk.DISABLED)

    def remove_guard_for_three_state_model(self):
        self.denat_dependent.set(1)
        self.denat_dependent_cb.config(state=Tk.NORMAL)

    def do_devel_test(self):
        from importlib import reload
        import Tools.EmbedCushion
        reload(Tools.EmbedCushion)
        from molass_legacy.Tools.EmbedCushion import embed_cushion

        embed_cushion(self)
