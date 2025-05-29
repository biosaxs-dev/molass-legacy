"""
    RangeEditorDialog.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import os
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.TkUtils import is_low_resolution
from molass_legacy.Decomposer.DecompUtils import CorrectedBaseline
from .RangeEditorFrame import RangeEditorFrame
from molass_legacy.Models.ElutionCurveModels     import EGHA, EMGA
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting

class RangeEditorDialog(Dialog):
    def __init__( self, parent, title, sd, mapper, judge_holder, scrollable=False ):
        self.logger  = logging.getLogger( __name__ )
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

    def is_ready( self ):
        return self.ready

    def prepare( self, counter=None):
        self.parent.config( cursor='wait' )
        self.counter = counter
        self.parent.update()
        Dialog.__init__( self, self.parent, self.title_, visible=False, block=False )
        self.parent.config( cursor='' )
        self.ready = True

    def cancel(self):
        # overiding cancel to cleanup self.fig
        # because the call to the destructor __del__ seems to be delayed
        self.editor_frame.close_figs()
        print("RangeEditorDialog: closed figs")
        Dialog.cancel(self)

    def user_cancel(self):
        # this method is provided aside from "cancel" to avoid confusion.
        set_setting('use_elution_models', self.init_use_elution_models)
        self.cancel()

    def show(self, counter=None):
        if not self.ready:
            # for backward compatibility
            self.prepare(counter=counter)

        self._show()

    def body( self, default_body_frame ):   # overrides parent class method
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

        corbase_info = CorrectedBaseline(self.sd, self.mapper)      # necessary?
        self.editor_frame = RangeEditorFrame(body_frame, self, self.sd, self.mapper, corbase_info)
        self.editor_frame.pack()

    def buttonbox(self, devel=False):
        if devel:
            from importlib import reload
            import molass_legacy.Extrapolation.PreviewButtonFrame
            reload(molass_legacy.Extrapolation.PreviewButtonFrame)
        from molass_legacy.Extrapolation.PreviewButtonFrame import PreviewButtonFrame

        box = Tk.Frame(self)
        box.pack(fill=Tk.X)

        lower_frame = Tk.Frame(box)
        lower_frame.pack(fill=Tk.X)

        additional_box = Tk.Frame(lower_frame)
        additional_box.pack(side=Tk.RIGHT, anchor=Tk.N + Tk.E)
        standard_box = Tk.Frame(lower_frame)
        standard_box.pack(anchor=Tk.CENTER)

        self.guide_frame = Tk.Frame(box)
        self.guide_frame.pack(fill=Tk.X)
        ok_cancel_frame = Tk.Frame(box)
        ok_cancel_frame.pack()

        self.preview_frame = PreviewButtonFrame(standard_box,
                                    dialog=self, use_elution_models=False, judge_holder=self.judge_holder,
                                    conc_opts=True, editor=self.editor_frame)
        self.preview_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        from molass_legacy.QuickAnalysis.AnalysisGuide import get_data_correction_state_text
        text = "Showing data processed with " + get_data_correction_state_text()
        guide_label = Tk.Label(self.guide_frame, text=text, bg='white')
        guide_label.pack(fill=Tk.X, pady=10)
        self.guide_message = guide_label

        w = Tk.Button(ok_cancel_frame, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.grid(row=1, column=0, padx=5, pady=5)
        self.ok_button = w

        space = Tk.Frame(ok_cancel_frame, width=160)
        space.grid(row=0, column=1)

        w = Tk.Button(ok_cancel_frame, text="Cancel", width=10, command=self.user_cancel)
        w.grid(row=1, column=2, padx=5, pady=5)
        self.cancel_button = w

        if False:
            w = Tk.Button(additional_box, text="Reset from Log", width=12, command=self.show_reset_from_log_dialog, state=Tk.DISABLED)
            w.pack(padx=10, pady=5)
            self.restore_button = w

        # self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.user_cancel)

    def get_conc_depend(self):
        return self.preview_frame.get_conc_depend()

    def apply( self ):
        self.editor_frame.apply()
        self.preview_frame.update_settings()        # concentration_datatype will be set_setting-ed here
        self.applied = True
        set_setting('use_elution_models', 0)
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

        filename = analysis_name.replace( 'analysis', 'figure' ) + '.png'
        path = os.path.join( folder, filename )
        w, h, x, y = split_geometry(self.geometry())
        # print( path, w, h, x, y  )
        screenshot( path, region=(x, y, w, h) ) # doesn't support multi-screen

    def make_range_info(self, concentration_datatype):
        set_setting('concentration_datatype', concentration_datatype)
        return self.editor_frame.make_range_info(concentration_datatype)

    def show_reset_from_log_dialog(self):
        # TODO:
        from molass_legacy.Decomposer.DecompResetDialog import DecompResetDialog
        self.get_current_frame().log_info_for_reset(modification="currently active")
        dialog = DecompResetDialog(self)
        dialog.show()
