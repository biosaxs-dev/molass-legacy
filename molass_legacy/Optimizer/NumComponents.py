"""
    Optimizer.NumComponents.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
from molass_legacy.Optimizer.TheUtils import guess_n_compnents_trial
from molass_legacy.Models.ElutionCurveModels import EGHA

MAXNUM_COMPONENTS = 5

class NumComponents:
    def __init__(self, fullopt_input):
        self.logger = logging.getLogger(__name__)
        self.fullopt_input = fullopt_input
        self.dsets = fullopt_input.get_dsets()
        self.decomp_result = None
        self.n_components = None

    def get_decomp_result(self, fullopt_input, mapped_params):
        from DecompUtils import CorrectedBaseline, decompose_elution_better

        A, B, num_peaks_mapping = mapped_params
        sd = fullopt_input.sd
        mapper = fullopt_input.mapper

        self.logger.info("guessing init_params with v1decomposer")

        corbase_info = CorrectedBaseline(sd, mapper)
        decomp_result = decompose_elution_better(corbase_info, mapper, model=EGHA(), logger=self.logger)

        self.decomp_result = decomp_result
        return decomp_result

    def guess(self):
        if self.decomp_result is None:
            # this case is not yet tested
            mapped_params = self.fullopt_input.get_mapped_params()
            self.get_decomp_result(self.fullopt_input, mapped_params)

        # self.n_components = guess_n_compnents_trial(self.dsets, self.logger)
        self.n_components = self.guess_from_decomp_opt_recs()

    def get(self):
        if self.n_components is None:
            self.guess()
        return self.n_components

    def guess_from_decomp_opt_recs(self, debug=False):
        from DecompUtils import debug_elution_plot

        xr_curve, xrD = self.dsets[0]
        x = xr_curve.x
        y = xr_curve.y
        opt_recs = self.decomp_result.opt_recs

        if debug:
            debug_elution_plot("guess_from_decomp_opt_recs", EGHA(), x, y, opt_recs)

        nc = 0
        for k, rec in enumerate(opt_recs):
            ev = rec.evaluator
            h = ev.get_param_value(0)
            if debug:
                print([k], ev.sign, h)
            if ev.sign > 0:
                nc += 1

        return nc
