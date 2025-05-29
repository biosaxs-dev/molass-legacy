"""
    ExtrapolationControlPanel.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.OurTtk import ttk, TtkScale
from molass_legacy.KekLib.TkSupplements import BlinkingFrame
from ExtrapolationAnimation import ExtrapolationAnimationDialog, TOO_SMALL_TO_PLOT
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from molass_legacy.KekLib.TkCustomWidgets import FileEntry
from molass_legacy.SerialAnalyzer.SerialDataUtils import serial_np_loadtxt

SCALE_MAX       = 0.2
SCALE_DIGITS    = 2
ENABLE_ANIMATE  = False
ENABLE_KNOWN_INPUT = False

class ControlPanel( Tk.Frame ):
    def __init__(self, parent, dialog, pno, ad, paired_range, row, selected, weights, ignore_bq, editor=None):
        Tk.Frame.__init__( self, parent )
        self.known = Tk.IntVar()

        if ENABLE_KNOWN_INPUT:
            cb= Tk.Checkbutton(self, text="Known", variable=self.known)
            cb.pack(anchor=Tk.W)

        self.unknown_panel = UnknownControlPanel(self, dialog, pno, ad, paired_range, row, selected, weights, ignore_bq, editor)
        self.unknown_panel.pack(anchor=Tk.W, padx=0)

        if ENABLE_KNOWN_INPUT:
            self.known_panel = KnownControlPanel(self, dialog, pno, ad, row, selected, self.unknown_panel)
            self.known_panel.pack(anchor=Tk.W, fill=Tk.X, expand=1)
            self.known_panel.pack_forget()

            self.known.trace("w", self.known_tracer)

    def known_tracer(self, *args):
        known = self.known.get()
        if known:
            self.unknown_panel.pack_forget()
            self.known_panel.pack(anchor=Tk.W)
        else:
            self.known_panel.pack_forget()
            self.unknown_panel.pack(anchor=Tk.W)

class UnknownControlPanel( Tk.Frame ):
    def __init__(self, parent, dialog, pno, ad, paired_range, row, selected, weights, ignore_bq, editor):
        Tk.Frame.__init__(self, parent)

        self.dialog = dialog
        self.editor = editor
        self.pno = pno
        self.ad = ad
        self.paired_range = paired_range
        self.row = row
        self.selected = selected
        # print( 'ControlPanel', [pno, ad, paired_range], selected )
        self.weights = weights
        self.solving_required = False
        self.add_conc_const = get_dev_setting('add_conc_const')
        self.individual_bq_ingore = get_dev_setting('individual_bq_ingore')

        scale_frame = Tk.Frame( self )
        scale_frame.pack()
        self.button_frame = BlinkingFrame( self )
        self.button_frame.pack()

        grid_row = 0
        base_drift_label = Tk.Label( scale_frame, text="Baseline Drift Penality" )
        base_drift_label.grid( row=grid_row, column=0, sticky=Tk.W + Tk.S )

        self.base_drift_scale = TtkScale( scale_frame, from_=0, to=SCALE_MAX,
                                        # resolution=0.1, sliderlength=10,
                                        length=160, orient=Tk.HORIZONTAL,
                                        showvalue=True, showvalue_y=14, digits=SCALE_DIGITS,
                                        style='my.Horizontal.TScale', slim=True )
        self.base_drift_scale.grid( row=grid_row, column=1 )
        self.base_drift_scale.set( self.weights[1] )

        if self.add_conc_const:
            self.base_drift_scale.variable.trace('w', self.parameter_tracer)
        else:
            base_drift_label.grid_forget()
            self.base_drift_scale.grid_forget()

        grid_row += 1
        a_smoothness_label = Tk.Label( scale_frame, text="A(q) Smoothness Penality" )
        a_smoothness_label.grid( row=grid_row, column=0, sticky=Tk.W + Tk.S )

        self.a_smoothness_scale = TtkScale( scale_frame, from_=0, to=SCALE_MAX,
                                        # resolution=0.1, sliderlength=10,
                                        length=160, orient=Tk.HORIZONTAL,
                                        showvalue=True, showvalue_y=14, digits=SCALE_DIGITS,
                                        style='my.Horizontal.TScale', slim=True )
        self.a_smoothness_scale.grid( row=grid_row, column=1 )
        self.a_smoothness_scale.set( self.weights[2] )

        grid_row += 1
        self.b_smoothness_label = Tk.Label( scale_frame, text="B(q) Smoothness Penality" )
        self.b_smoothness_row = grid_row

        self.b_smoothness_scale = TtkScale( scale_frame, from_=0, to=SCALE_MAX,
                                        # resolution=0.1, sliderlength=10,
                                        length=160, orient=Tk.HORIZONTAL,
                                        showvalue=True, showvalue_y=14, digits=SCALE_DIGITS,
                                        style='my.Horizontal.TScale', slim=True )
        self.b_smoothness_scale.set( self.weights[3] )

        self.ignore_bq = Tk.IntVar()
        self.ignore_bq.set(ignore_bq)
        self.bq_widgets_update()

        if dialog.aq_smoothness:
            self.a_smoothness_scale.variable.trace('w', self.parameter_tracer)
            self.b_smoothness_scale.variable.trace('w', self.parameter_tracer)
        else:
            a_smoothness_label.grid_forget()
            self.a_smoothness_scale.grid_forget()
            self.b_smoothness_label.grid_forget()
            self.b_smoothness_scale.grid_forget()

        grid_row += 1
        a_neg_label = Tk.Label( scale_frame, text="A(q) Positivity Penality" )
        a_neg_label.grid( row=grid_row, column=0, sticky=Tk.W + Tk.S )

        style = ttk.Style(self)
        style.configure('my.Horizontal.TScale', sliderlength=10)

        self.a_neg_scale = TtkScale( scale_frame, from_=0, to=SCALE_MAX,
                                        # resolution=0.1, sliderlength=10,
                                        length=160, orient=Tk.HORIZONTAL,
                                        showvalue=True, showvalue_y=14, digits=SCALE_DIGITS,
                                        style='my.Horizontal.TScale', slim=True )
        self.a_neg_scale.grid( row=grid_row, column=1 )
        self.a_neg_scale.set( self.weights[0] )

        if dialog.aq_positivity:
            self.a_neg_scale.variable.trace('w', self.parameter_tracer)
        else:
            a_neg_label.grid_forget()
            self.a_neg_scale.grid_forget()

        if self.individual_bq_ingore:
            ignore_bq_cb = Tk.Checkbutton( self.button_frame, text="Ignore B(q)", variable=self.ignore_bq, state=Tk.NORMAL )
            ignore_bq_cb.pack( side=Tk.LEFT, padx=5, pady=5 )
            self.ignore_bq.trace('w', self.ignore_bq_tracer)

        self.button_frame.objects = []

        if ENABLE_KNOWN_INPUT:
            solve_button = Tk.Button( self.button_frame, text="Solve", command=self.solve_plot )
            solve_button.pack(padx=5, pady=5)
            self.button_frame.objects.append(solve_button)

        if ENABLE_ANIMATE:
            amin_button = Tk.Button( self.button_frame, text="Animate", command=self.make_animation )
            amin_button.pack(padx=5, pady=5)
            self.button_frame.objects.append(amin_button)

        range_button = Tk.Button( self.button_frame, text="Range", command=self.show_range_inspector )
        range_button.pack(padx=5, pady=5 )

        rank_button = Tk.Button( self.button_frame, text="Rank", command=self.show_cdi_dialog )
        rank_button.pack(padx=5, pady=5 )

        denss_button = Tk.Button( self.button_frame, text="DENSS", command=self.show_denss_dialog )
        denss_button.pack(padx=5, pady=5 )

    def set_C_matrix( self, C ):
        self.C = C

    def bq_widgets_update( self ):
        if self.dialog.aq_smoothness:
            ignore_bq = self.ignore_bq.get()
            if ignore_bq:
                self.b_smoothness_label.grid_forget()
                self.b_smoothness_scale.grid_forget()
            else:
                self.b_smoothness_label.grid( row=self.b_smoothness_row, column=0, sticky=Tk.W + Tk.S )
                self.b_smoothness_scale.grid( row=self.b_smoothness_row, column=1 )

    def ignore_bq_tracer( self, *args ):
        if self.solving_required:
            self.button_frame.stop()
            self.solving_required = False
        else:
            self.button_frame.start()
            self.solving_required = True
        self.bq_widgets_update()

    def parameter_tracer( self, *args ):
        self.solving_required = True
        self.button_frame.start()

    def get_penalty_weights( self ):
        # always return all weights
        return [    self.a_neg_scale.get(),
                    self.base_drift_scale.get(),
                    self.a_smoothness_scale.get(),
                    self.b_smoothness_scale.get() ]

    def solve_plot( self ):
        self.button_frame.stop()
        self.solving_required = False
        self.dialog.config(cursor='wait')
        self.dialog.update()
        penalty_weights = self.get_penalty_weights()
        self.dialog.solve_plot( self.row, self.selected, penalty_weights=penalty_weights, ignore_bq=self.ignore_bq.get() )
        self.dialog.canvas_draw()
        self.bq_widgets_update()
        self.dialog.config(cursor='')
        self.dialog.update()

    def make_animation( self ):
        self.button_frame.stop()
        self.solving_required = False
        self.dialog.config(cursor='wait')
        self.dialog.update()
        penalty_weights = self.get_penalty_weights()
        anim_data = self.dialog.solve_plot( self.row, self.selected, penalty_weights=penalty_weights, ignore_bq=self.ignore_bq.get(), animation=True )
        self.dialog.canvas_draw()
        self.dialog.config(cursor='')
        self.dialog.update()

        # ridge = self.dialog.data[]
        # print( 'cnv_ranges=', self.dialog.cnv_ranges[self.row] )
        top_x = self.paired_range.top_x
        ridge = self.dialog.data[:,top_x] / self.dialog.mc_vector[top_x]
        ridge[ridge < TOO_SMALL_TO_PLOT] = 0    # to avoid these values from deforming the figure
        dialog = ExtrapolationAnimationDialog( self.dialog, self.row, self.dialog.q, ridge, anim_data, penalty_weights, ignore_bq=self.ignore_bq.get() )
        dialog.show()

    def show_range_inspector( self ):
        from .RangeInspector import RangeInspectorDialog
        dialog = self.dialog
        try:
            solver = dialog.solver.impl
        except:
            solver = dialog.solver
        init_result = dialog.solver_results[self.row]
        conc_depend = dialog.popts.conc_depend
        inspector = RangeInspectorDialog(dialog.parent, solver, dialog.j0, self.paired_range, self.selected, self.ad,
                    conc_depend=conc_depend,
                    init_result=init_result)
        ret = inspector.show()
        if ret:
            new_range = inspector.get_range(shifted=False)
            self.aplly_new_range(*new_range)
            new_shifted_range = inspector.get_range(shifted=True)
            self.editor.update_range(self.pno, self.ad, *new_shifted_range)
            self.solve_plot()

    def show_cdi_dialog(self):
        from molass_legacy.Conc.CdInspection import CdInspectionDailog
        print( self.pno, self.ad, self.paired_range )
        print( self.dialog.q.shape, self.dialog.data.shape )

        q = self.dialog.q

        f, t = self.paired_range[self.ad]
        eslice = slice(f, t+1)
        M = self.dialog.data[:, eslice]
        E = self.dialog.error[:, eslice]
        from_ax = self.dialog.axis_array[self.row][0]
        xray_scale = self.dialog.sd.get_xray_scale()
        dialog = CdInspectionDailog(self.dialog.parent, self.dialog, M, E, self.C, q, eslice, from_ax=from_ax, xray_scale=xray_scale)
        dialog.show()

    def aplly_new_range(self, f, t):
        print('aplly_new_range', (f, t))
        self.paired_range.update_range(self.ad, f, t)
        self.dialog.update_to_solve_range(self.row, f, t)
        """
        TODO:
            refactoring: unifiy these range infos
        """

    def get_file_name(self):
        print('show_denss_dialog', self.pno, self.ad, self.paired_range, self.row)
        if len(self.paired_range.get_fromto_list()) == 1:
            ad_str = 'bth'
        else:
            ad_str = 'asc' if self.ad == 0 else 'dsc'
        file_name = 'pk%d_%s_A.dat' % (self.pno+1, ad_str)
        return file_name

    def show_denss_dialog( self ):
        from molass_legacy._MOLASS.Version import is_developing_version
        if is_developing_version():
            from importlib import reload
            import molass_legacy.DENSS.DenssGui
            reload(molass_legacy.DENSS.DenssGui)
        from molass_legacy.DENSS.DenssGui import DenssGuiDialog
        A, B, Z, E, _, C = self.dialog.solver_results[self.row]
        dialog = DenssGuiDialog( self.dialog.parent, self.dialog.q, A, E[0], self.get_file_name() )
        dialog.show()

class KnownControlPanel( Tk.Frame ):
    def __init__(self, parent, dialog, pno, ad, row, selected, unknown_panel):
        width = unknown_panel.cget('width')
        height = unknown_panel.cget('height')
        Tk.Frame.__init__( self, parent, width=width, height=height)

        self.dialog = dialog
        self.row = row
        self.selected = selected

        label = Tk.Label(self, text="Enter a file path for your known scattering curve.")
        label.pack(anchor=Tk.W)

        self.known_data_file = Tk.StringVar()
        entry = FileEntry(self, textvariable=self.known_data_file, width=45, on_entry_cb=self.on_entry_file)
        entry.pack()

        num_spaces = 3      # actually variable
        for k in range(num_spaces):
            space = Tk.Label(self)
            space.pack()

    def on_entry_file(self):
        file_path = self.known_data_file.get()
        self.data, _ = serial_np_loadtxt(file_path)
        self.dialog.plot_known(self.row, self.selected, self.data)
        self.dialog.su_btn_blink.start()
