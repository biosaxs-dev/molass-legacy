"""
    DecompEditorDialog.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import os
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy.Decomposer.DecompUtils import CorrectedBaseline
from .DecompDummyFrame import DecompDummyFrame
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy._MOLASS.Version import is_developing_version
from molass_legacy._MOLASS.Version import is_developing_version
from molass_legacy.Selective.ModelSelectFrame import enable_edm_model, MODEL_NAMES

class DecompEditorDialog(Dialog):
    def __init__(self, parent, title, sd, mapper, judge_holder, scrollable=False, debug=False):
        self.logger  = logging.getLogger( __name__ )

        if debug:
            from molass_legacy.Mapping.MappingUtils import save_mapped_curves
            save_mapped_curves(mapper, plot=True)

        self.parent = parent
        self.title_ = title
        self.sd     = sd
        self.serial_data = sd   # for compatibility with AnalyzerDialog for Global Fitting
        self.mapper = mapper
        self.judge_holder = judge_holder
        self.ready = False
        self.scrollable = scrollable or is_low_resolution()
        self.canceled = False
        self.applied = False
        self.init_use_elution_models = get_setting('use_elution_models')
        set_setting('use_elution_models', 1)    # required to preview in this dialog
        self.global_fit = Tk.IntVar()
        self.global_fit.set(0)
        self.global_fit.trace_add("write", self.global_fit_tracer)
        self.global_frames = [None, None]

    def get_model_list(self):
        if is_developing_version():
            from importlib import reload
            import molass_legacy.Models.ElutionCurveModels
            import molass_legacy.Models.RateTheory.EDM
            # import molass_legacy.Models.Stochastic.Tripore
            reload(molass_legacy.Models.ElutionCurveModels)
            reload(molass_legacy.Models.RateTheory.EDM)
            # reload(molass_legacy.Models.Stochastic.Tripore)
        from molass_legacy.Models.ElutionCurveModels import EGHA, EMGA
        if enable_edm_model:
            from molass_legacy.Models.RateTheory.EDM import EDM
            from molass_legacy.Models.Stochastic.Monopore import Monopore
            MODEL_LIST = [EMGA(), EGHA(), EDM(delayed=True), Monopore(delayed=True)]
        else:
            MODEL_LIST = [EMGA(), EGHA()]
        return MODEL_LIST 

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        for frame in self.frames:
            frame.close_figs()
        print("DecompEditorDialog: closed figs")
        Dialog.cancel(self)

    def user_cancel(self):
        # this method is provided aside from "cancel" to avoid confusion.
        set_setting('use_elution_models', self.init_use_elution_models)
        self.cancel()

    def is_ready( self ):
        return self.ready

    def prepare( self, counter=None):
        self.parent.config( cursor='wait' )
        self.counter = counter
        self.parent.update()
        Dialog.__init__( self, self.parent, self.title_, visible=False, block=False )
        self.parent.config( cursor='' )
        self.ready = True
        self.advanced_frame.enable_buttons()

    def debug_sd(self):
        from molass_legacy.DataStructure.MdViewer import MdViewer
        sd = self.sd
        M = sd.intensity_array[:,:,1].T
        mdv = MdViewer(self, M, xvector=sd.qvector)
        mdv.show()

    def show(self, counter=None):
        if not self.ready:
            # for backward compatibility
            self.prepare(counter=counter)

        # self.after(1000, self.debug_sd)
        self._show()

    def body(self, default_body_frame, devel=True):
        if devel:
            from importlib import reload
            import molass_legacy.RangeEditors.DecompEditorFrame
            reload(molass_legacy.RangeEditors.DecompEditorFrame)
        from .DecompEditorFrame import DecompEditorFrame
        set_icon( self )

        if self.scrollable:
            from molass_legacy.KekLib.ScrolledFrame import ScrolledFrame
            self.scrolled_frame = ScrolledFrame(default_body_frame)
            # scrolled_frame.pack( fill=Tk.BOTH, expand=1 )
            self.scrolled_frame.pack(anchor=Tk.N)
            body_frame = self.scrolled_frame.interior
        else:
            self.scrolled_frame = None
            body_frame = default_body_frame

        self.body_frame = body_frame    # for use in create_new_frame

        self.corbase_info = CorrectedBaseline(self.sd, self.mapper)
        if self.counter is not None:
            self.counter[0] += 1
            self.parent.update()

        model_list = self.get_model_list()
        memorized_model = get_setting('editor_model')
        if memorized_model is None:
            selected_index = None
        else:
            selected_index = MODEL_NAMES.index(memorized_model)
            selected_model = model_list[selected_index]
            self.logger.info('previously applied %s has been selected.' % selected_model.get_name())

        min_score = None
        the_error = None
        min_index = None
        min_model = None
        self.frames = []
        scores = []
        errors = []
        for k, model in enumerate(model_list):
            if model.is_delayed() and (selected_index is None or selected_index != k):
                frame = DecompDummyFrame(body_frame, model)
            else:
                self.min_index = min_index      # for reference in delayed (i.e., non-traditional,) frames
                frame = DecompEditorFrame(body_frame, self, self.sd, self.mapper, self.corbase_info, model)
                if model.is_delayed():
                    model.set_delayed_off()
            frame.pack()
            if self.counter is not None:
                self.counter[0] += 1
                self.parent.update()
            self.frames.append(frame)
            score = frame.get_decomp_score()
            scores.append(score)
            error = frame.get_fit_error()
            errors.append(error)
            # print([k], error)
            if min_score is None or score < min_score:
                min_score = score
                the_error = error
                min_index = k
                min_model = model

        if selected_index is None:
            selected_index = min_index

        self.current = selected_index
        self.min_index = min_index      # for reference in delayed frames
        for k in range(len(model_list)):
            if k != self.current:
                self.frames[k].pack_forget()

    def buttonbox(self, devel=False):
        from molass_legacy.Selective.ModelSelectFrame import ModelSelectFrame
        from molass_legacy.Selective.AdvancedFrame import AdvancedFrame
        from molass_legacy.Extrapolation.PreviewButtonFrame import PreviewButtonFrame

        box = Tk.Frame(self)
        box.pack(fill=Tk.X)

        lower_frame = Tk.Frame(box)
        lower_frame.pack(fill=Tk.X)

        additional_box = Tk.Frame(lower_frame)
        additional_box.pack(side=Tk.RIGHT, anchor=Tk.N + Tk.E)
        standard_box = Tk.Frame(lower_frame)
        standard_box.pack(anchor=Tk.CENTER, padx=10)

        self.guide_frame = Tk.Frame(box)
        self.guide_frame.pack(fill=Tk.X)
        ok_cancel_frame = Tk.Frame(box)
        ok_cancel_frame.pack()

        sb_col = 0
        self.select_frame = ModelSelectFrame(standard_box, self, text="Model and Column Selection", labelanchor=Tk.N)
        self.select_frame.grid(row=0, column=0, padx=10)

        sb_col += 1
        self.advanced_frame = AdvancedFrame(standard_box, self)
        self.advanced_frame.grid(row=0, column=sb_col, pady=5)

        sb_col += 1
        self.preview_frame = PreviewButtonFrame(standard_box,
                                    dialog=self, use_elution_models=True, judge_holder=self.judge_holder,
                                    editor=self.get_current_frame())
        self.preview_frame.grid(row=0, column=sb_col, padx=10, pady=5, ipady=12)

        from molass_legacy.QuickAnalysis.AnalysisGuide import get_data_correction_state_text
        text = "Showing data processed with " + get_data_correction_state_text()
        guide_label = Tk.Label(self.guide_frame, text=text, bg='white')
        guide_label.pack(fill=Tk.X, pady=10)
        self.guide_message = guide_label

        w = Tk.Button(ok_cancel_frame, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.grid(row=0, column=0, pady=5)
        self.ok_button = w

        space = Tk.Frame(ok_cancel_frame, width=160)
        space.grid(row=0, column=1)

        w = Tk.Button(ok_cancel_frame, text="Cancel", width=10, command=self.user_cancel)
        w.grid(row=0, column=2, pady=5)
        self.cancel_button = w

        if False:
            w = Tk.Button(additional_box, text="Reset from Log", width=12, command=self.show_reset_from_log_dialog)
            w.pack(padx=10, pady=5)
            self.restore_button = w

        # self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.user_cancel)
        self.after(100, self.update_selection_state)

    def update_selection_state(self):
        self.select_frame.update_selection_state()

    def use_column_constraints_tracer(self, *args):
        state = Tk.DISABLED if self.use_column_constraints.get() == 0 else Tk.NORMAL
        self.selector.config(state=state)

    def change_model(self, selected_name):
        from .DecompEditorFrame import DecompEditorFrame
        current = self.current
        next_ = MODEL_NAMES.index(selected_name)
        if next_ == current:
            return

        self.advanced_frame.disable_buttons()
        self.update()   # to make the newly selected name appear timely

        next_frame = self.frames[next_]
        if next_frame.is_delayed():
            next_model = next_frame.model
            next_model.set_delayed_off()
            next_frame.destroy()
            new_frame = DecompEditorFrame(self.body_frame, self, self.sd, self.mapper, self.corbase_info, next_model)
            new_frame.pack()
            new_frame.pack_forget()
            self.frames[next_] = new_frame

        global_mode = self.global_fit.get()
        if global_mode:
            self.global_frames[current].pack_forget()
            global_frame = self.get_global_frame(next_)
            global_frame.pack()
        else:
            current_fig_type = self.frames[current].elution_fig_type
            self.frames[current].pack_forget()
            self.frames[next_].pack()
            self.frames[next_].refresh_frame(elution_fig_type=current_fig_type)
        self.current = next_
        self.preview_frame.update_modelframe()
        self.advanced_frame.update_mode_dependent_state()
        self.update_selection_state()

    def get_conc_depend(self):
        return self.preview_frame.get_conc_depend()

    def apply( self ):
        current_frame = self.frames[self.current]
        current_frame.apply()
        self.preview_frame.update_settings()        # concentration_datatype will be set_setting-ed here
        self.applied = True
        set_setting('use_elution_models', 1)
        set_setting('has_elution_models', True)
        set_setting('matrix_formulation', 1)
        self.try_image_save()

    def try_image_save( self ):
        # TODO: move this code to Tester.py

        test_pattern = get_setting('test_pattern')
        if test_pattern is None or test_pattern < 7:
            return

        fig_type = self.get_current_frame().elution_fig_type
        data_type = '_xray' if fig_type == 0 else '_uv'
        decomp_image_folder = get_setting('decomp_image_folder' + data_type )
        if decomp_image_folder is not None and os.path.exists(decomp_image_folder):
            self.save_the_figure( decomp_image_folder, get_setting('analysis_name') )

    def save_the_figure(self, folder, analysis_name):
        from pyautogui import screenshot
        from molass_legacy.KekLib.TkUtils import split_geometry

        self.focus_force()
        self.update()
        filename = analysis_name.replace( 'analysis', 'figure' ) + '.png'
        path = os.path.join( folder, filename )
        w, h, x, y = split_geometry(self.geometry())
        # print( path, w, h, x, y  )
        screenshot( path, region=(x, y, w, h) ) # doesn't support multi-screen

    def get_current_frame(self):
        return self.frames[self.current]

    def get_current_model(self):
        return self.get_current_frame().model

    def get_current_params_array(self):
        import numpy as np
        frame = self.get_current_frame()
        params_list = []
        for rec in frame.opt_recs:
            params = rec.get_params()
            params_list.append(params)
        return np.array(params_list)       

    def get_current_modelname(self):
        return self.get_current_frame().model_name

    def make_range_info(self, concentration_datatype):
        current_frame = self.get_current_frame()        # note that current frame will never be a dummy
        return current_frame.make_range_info(concentration_datatype)

    def get_applied_model(self):
        return self.frames[self.current].model

    def get_model_fit_errors(self):
        fit_errors = []
        for frame in self.frames:
            fit_errors.append( frame.get_fit_error() )
        return fit_errors

    def show_reset_from_log_dialog(self):
        from molass_legacy.Decomposer.DecompResetDialog import DecompResetDialog
        self.get_current_frame().log_info_for_reset(modification="currently active")
        dialog = DecompResetDialog(self)
        dialog.show()

    def global_fit_tracer(self, *args):
        global_mode = self.global_fit.get()
        current = self.current
        current_frame = self.get_current_frame()
        global_frame = self.get_global_frame(current)
        if global_mode:
            current_frame.pack_forget()
            global_frame.pack()
        else:
            global_frame.pack_forget()
            current_frame.pack()

    def get_global_frame(self, i):
        global_frame = self.global_frames[i]
        if global_frame is None:
            self.config(cursor='wait')
            self.update()
            decomp_result = self.frames[i].compute_global_fitting()
            global_frame = self.create_new_frame(i, decomp_result)
            self.global_frames[i] = global_frame
            self.config(cursor='')
        return global_frame

    def create_new_frame(self, i, decomp_result):
        from .DecompEditorFrame import DecompEditorFrame
        model = self.get_model_list()[i]
        frame = DecompEditorFrame(self.body_frame, self, self.sd, self.mapper, self.corbase_info, model, decomp_result=decomp_result, global_flag=True)
        return frame

    def update_global_btn_state(self, fig_type):
        # task: remove this method
        pass

    def update_current_frame_with_result(self, decomp_result, debug=False):
        self.get_current_frame().update_with_result(decomp_result, debug=debug)

    def reset_current_frame(self):
        self.get_current_frame().reset()
