# coding: utf-8
"""
    PreviewController.py

    Copyright (c) 2018-2021, SAXS Team, KEK-PF
"""
import logging
from molass_legacy.PeaksetSelector import PeakSetSelector
from ExtrapolationSolver import ExtrapolationSolver, run_extrapolation_solver
from ExtrapolSolverDialog import ExtrapolSolverDialog
from molass_legacy._MOLASS.SerialSettings import get_setting

class PreviewController:
    def __init__(self, dialog=None, editor=None):
        self.logger = logging.getLogger(__name__)
        self.parent_dialog = dialog
        self.editor = editor
        self.solver_results = None

    def run_solver( self, parent, pdata, popts,
                    known_info_list=None,
                    weighted=True, use_gui=True ):

        self.parent = parent
        self.mapper = mapper = pdata.mapper
        self.pdata = pdata
        self.popts = popts
        self.logger.info('cnv_ranges=%s, num_ranges=%s', str(pdata.cnv_ranges), str(pdata.num_ranges))

        self.arg_paired_ranges = pdata.paired_ranges    # keep this to call resursively in ExtrapolSolverDialog.solve_unknowns

        self.solver = ExtrapolationSolver(pdata, popts)

        self.selector = PeakSetSelector(pdata.cnv_ranges, mapper.x_curve)
        if known_info_list is None:
            known_info_list = get_setting('known_info_list')
        if known_info_list is not None:
            self.selector.update_known_peak_info_list(known_info_list)

        # parent_ = parent if aq_smoothness else None
        parent_ = parent

        ret_tuple = run_extrapolation_solver(
            self.solver, self.selector, parent=parent_, use_gui=use_gui,
            )
        self.solver_results = ret_tuple[0]
        self.peak_range_infos = ret_tuple[1]

        # assert use_elution_models
        self.use_elution_models = popts.use_elution_models

    def ok( self ):
        return self.solver_results is not None

    def show_dialog( self ):
        self.dialog = ExtrapolSolverDialog( self.parent_dialog.parent, self )
        self.dialog.show()

    def get_dialog( self ):
        return self.dialog

    def get_solver_ret_tuple( self ):
        return self.solver_results, self.peak_range_infos

    def get_better_results(self):
        # for compaitbility with LrfResultPool
        return self.solver_results
