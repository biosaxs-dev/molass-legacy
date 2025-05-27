"""
    RangeSpecFrame.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import os
import copy
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.gridspec    import GridSpec
from matplotlib import colors
import matplotlib
import seaborn as sns
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.TkUtils                import split_geometry
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar, get_color, get_hex_color
from molass_legacy.KekLib.DisguinsingWidgets     import CheckonLabel
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting
from molass_legacy.DataStructure.PeakInfo               import PeakInfo
from molass_legacy.DataStructure.AnalysisRangeInfo      import PairedRange

RANGE_FRAME_HEIGHT      = 60

class RangeSpecFrame(Tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        self.parent = parent

        self.editor = kwargs.pop('editor', None)
        self.j_min = kwargs.pop('j_min', 0)
        self.j_max = kwargs.pop('j_max', 1000)
        self.peak_no = kwargs.pop('peak_no', None)
        self.active_peak_no = kwargs.pop('active_peak_no', None)
        self.row_no_base =  kwargs.pop('row_no_base', None)
        self.col_widths = col_widths = kwargs.pop('col_widths', None)
        self.select_list = kwargs.pop('select_list', None)
        self.range_list = kwargs.pop('range_list', None)
        """
            self.range_list == self.editor.editor_ranges[k]
        """
        self.peak = kwargs.pop('peak', None)
        self.peak_info = PeakInfo(self.peak_no-1, self.peak.top_x)

        num_ranges_ = len(self.range_list)

        Tk.LabelFrame.__init__(self, parent, *args, **kwargs)

        # Peak No
        frame1 = Tk.Frame(self, width=col_widths[1], height=20)
        frame1.pack_propagate(0)
        frame1.pack(side=Tk.LEFT)
        label = Tk.Label(frame1, text=str(self.peak_no))
        label.pack()

        # Number of Ranges
        frame3 = Tk.Frame(self, width=col_widths[2], height=20)
        frame3.pack_propagate(0)
        frame3.pack(side=Tk.LEFT)
        self.num_ranges = Tk.IntVar()
        self.num_ranges.set(num_ranges_)
        min_num_ranges = 1
        # state = Tk.NORMAL if self.peak.sign > 0 else Tk.DISABLED
        state = Tk.NORMAL
        self.spinbox = Tk.Spinbox( frame3, textvariable=self.num_ranges,
                                from_=min_num_ranges, to=2, increment=1,
                                justify=Tk.CENTER, width=6, state=state )

        self.num_ranges.trace('w', self.num_ranges_tracer)
        self.spinbox.pack()

        # Range(s)
        self.frame34 = frame34 = Tk.Frame(self, width=col_widths[3]+col_widths[4], height=RANGE_FRAME_HEIGHT)
        frame34.pack_propagate(0)
        frame34.pack(side=Tk.LEFT)

        # Ignore
        frame6 = Tk.Frame(self, width=col_widths[5], height=20)
        frame6.pack_propagate(0)
        frame6.pack(side=Tk.LEFT)
        self.ignore = Tk.IntVar()
        self.ignore.set(int(self.editor.ignorable_flags[self.peak_no-1]))

        self.range_widgets = []
        self.update_ranges()    # uses self.ignore

        if self.peak.sign > 0:
            cb = Tk.Checkbutton(frame6, variable=self.ignore)
            self.ignore.trace('w', self.ignore_tracer)
        else:
            cb = CheckonLabel(frame6)
        cb.pack()

    def update_ranges(self, *args):
        for w in self.range_widgets:
            w.destroy()

        peak = self.peak
        num_ranges = self.num_ranges.get()
        range_list = self.range_list

        self.range_widgets = []
        """
            gridding directly from self.frame34 seems to break self.frame34.pack_propagate(0)
            therefore, create an intermediate frame and do gridding from it
        """
        frame = Tk.Frame(self.frame34)
        frame.pack(fill=Tk.BOTH, expand=1)
        self.range_widgets.append(frame)

        self.range_var_list = []
        self.edit_buttons = []
        for k in range(num_ranges):
            frame.grid_rowconfigure(k, weight=1)
            range_frame = Tk.Frame(frame)
            range_frame.grid(row=k, column=0)

            label_frame = Tk.Frame(range_frame, width=self.col_widths[3], height=20)
            label_frame.pack_propagate(0)
            label_frame.pack(side=Tk.LEFT)
            text = '' if self.active_peak_no is None else ('%d-%d' % (self.active_peak_no, k+1))
            label = Tk.Label(label_frame, text=text)
            label.pack()

            spinbox_frame = Tk.Frame(range_frame, width=self.col_widths[4], height=30)
            spinbox_frame.pack_propagate(0)
            spinbox_frame.pack(side=Tk.LEFT, padx=5)

            fv, tv = range_list[k]

            f = Tk.IntVar()
            f.set(fv)
            t = Tk.IntVar()
            t.set(tv)
            self.range_var_list.append( [f, t] )
            spinbox1 = Tk.Spinbox( spinbox_frame, textvariable=f,
                                from_=self.j_min, to=self.j_max, increment=1,
                                justify=Tk.CENTER, width=6 )
            spinbox1.pack(side=Tk.LEFT, padx=5, pady=5)
            f.trace( 'w',  lambda *args, k_=k: self.spinbox_tracer(k_, 0) )

            spinbox2 = Tk.Spinbox( spinbox_frame, textvariable=t,
                                from_=self.j_min, to=self.j_max, increment=1,
                                justify=Tk.CENTER, width=6 )
            spinbox2.pack(side=Tk.LEFT, padx=5, pady=5)
            t.trace( 'w',  lambda *args, k_=k: self.spinbox_tracer(k_, 1) )
            self.spinbox_trace = True

            state = Tk.DISABLED if self.ignore.get() else Tk.NORMAL
            edit_button = Tk.Button(spinbox_frame, text="🖉", state=state,
                                command=lambda k_=k: self.show_range_inspector(k_))
            edit_button.pack(side=Tk.LEFT, padx=5)
            self.edit_buttons.append(edit_button)

    def spinbox_tracer(self, i, j):
        if not self.spinbox_trace:
            return

        try:
            v = self.range_var_list[i][j].get()
            # print( 'spinbox_tracer: ', (i, j), v )
            self.range_list[i][j] = v
            if False:
                self.editor.update_figs(peak_no=self.peak_no)
            else:
                self.editor.refresh_figs()
        except:
            pass

    def get_conc_depend(self):
        return self.editor.dialog.get_conc_depend()

    def show_range_inspector(self, k):
        from molass_legacy.DataStructure.AnalysisRangeInfo import PairedRange
        from molass_legacy.PeaksetSelector import PeakSetSelector
        from RangeInspector import RangeInspectorDialog
        row = self.row_no_base + k
        print('show_range_inspector: (row, k)=', (row, k))
        print('select_list=', self.select_list)
        print('range_list=', self.range_list)
        solver = self.editor.get_extrapolation_solver()
        j0 = self.editor.xr_j0
        ranges = [[j-j0 for j in fromto] for fromto in self.range_list]
        paired_range = PairedRange(self.peak_info, *ranges)
        selector = PeakSetSelector(solver.cnv_ranges, self.editor.mapper.x_curve)
        peakset_info = selector.select_peakset(row)
        ad = k
        conc_depend = self.get_conc_depend()
        inspector = RangeInspectorDialog(self.editor.dialog.parent, solver, j0,
                        paired_range, peakset_info, ad, conc_depend=conc_depend)
        ret = inspector.show()
        if ret:
            new_shifted_range = inspector.get_range(shifted=True)
            self.editor.update_range(self.peak_no-1, ad, *new_shifted_range)

    def update_range(self, ad, f, t):
        self.spinbox_trace = False
        vars_ = self.range_var_list[ad]
        vars_[0].set(f)
        vars_[1].set(t)
        self.update()
        self.spinbox_trace = True

    def num_ranges_tracer(self, *args):
        try:
            num_ranges = self.num_ranges.get()
        except:
            return

        assert num_ranges in [0, 1, 2]

        prev_range_list = copy.deepcopy(self.range_list)
        self.range_list.clear()
        """
            the above also means self.editor.editor_ranges[k].clear()
            because self.range_list == self.editor.editor_ranges[k]
        """

        if num_ranges > 0:
            if len(prev_range_list) > 0:
                f = prev_range_list[0][0]
                t = prev_range_list[-1][1]
            else:
                range_list = self.opt_rec.get_range_list(self.editor.x)
                f = range_list[0][0]
                t = range_list[-1][-1]

            if num_ranges == 1:
                self.range_list.append( [f, t] )
            elif num_ranges == 2:
                top_x = self.j_min + self.peak.top_x
                self.range_list.append( [f, top_x] )
                self.range_list.append( [top_x, t] )

        self.update_ranges()

        # specifying one peak_no is not appropriate, it is better to update all figgures
        self.editor.refresh_figs()
        self.editor.refresh_spec_panels()

    def update_eebars(self):
        for panel in self.editor.specpanel_list:
            panel.eebar.refresh()

    def ignore_tracer(self, *args):
        ignore = bool(self.ignore.get())
        self.editor.ignorable_flags[self.peak_no-1] = ignore
        state = Tk.DISABLED if ignore else Tk.NORMAL
        for w in self.edit_buttons:
            w.config(state=state)
        self.editor.refresh_figs()
        self.editor.refresh_spec_panels()

    def get_info_for_reset(self):
        num_ranges = self.num_ranges.get()
        range_info = [ (v[0].get(), v[1].get()) for v in self.range_var_list]
        ignore = self.ignore.get()
        return (num_ranges, range_info, ignore)

    def get_paired_range(self):
        # note that elm_recs is omitted here
        # consider to refactor to unify PeakInfo and EmgPeak
        range_list = [ [f - self.j_min, t - self.j_min] for f, t  in self.range_list]
        return PairedRange(self.peak_info, *range_list)
