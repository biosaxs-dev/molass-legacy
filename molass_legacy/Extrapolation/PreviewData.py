"""
    PreviewData.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy.DataStructure.AnalysisRangeInfo import convert_to_paired_ranges, shift_paired_ranges
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util

class PreviewData:
    def __init__(self, sd=None, mapper=None, judge_holder=None, paired_ranges=None,
                    xdata=None, decomp_info=None, slice_=None,
                    conc_factor=None,
                    ):
        self.logger = logging.getLogger(__name__)
        self.is_for_sec = mapper is not None
        self.mapper = mapper
        self.judge_holder = judge_holder
        self.paired_ranges = paired_ranges
        self.xdata = xdata
        self.decomp_info = decomp_info
        self.slice_ = slice_

        if self.is_for_sec:
            self.sd = sd
            self.mc_vector = sd.mc_vector
            if conc_factor is None:
                conc_factor = compute_conc_factor_util()
            conc = None
            conc_curves = None
            if paired_ranges is None:
                ranges_ = mapper.x_ranges
            else:
                ranges_ = paired_ranges
        else:
            from molass_legacy.KekLib.BasicUtils import Struct
            decomp_info = self.decomp_info
            ranges_ = decomp_info.ranges
            ft_list = ranges_[0].get_fromto_list()
            xr_j0 = ft_list[0][0]
            self.sd = Struct(xr_j0=xr_j0)   # TODO: consider to set serial_data
            self.mc_vector = np.ones(len(xdata.e_y))
            if conc_factor is None:
                conc_factor = get_setting('conc_factor')
            conc = decomp_info.conc
            conc_curves = decomp_info.curves

        ret = convert_to_paired_ranges(ranges_)
        self.cnv_ranges = ret[0]
        self.num_ranges = ret[1]

        self.logger.info("cnv_ranges=%s from ranges_=%s", self.cnv_ranges, ranges_)

        assert conc_factor is not None
        self.conc_factor = conc_factor
        self.conc = conc
        self.conc_curves = conc_curves

    def get_analysis_ranges(self):
        """
        this shift is required in MCT to make it compatible with SEC.
        """
        assert not self.is_for_sec
        return shift_paired_ranges(self.sd.xr_j0, self.cnv_ranges)

    def make_conc_vector(self):
        if self.is_for_sec:
            # conc_factor ?
            y_ = self.mapper.make_uniformly_scaled_vector(scale=1)
        else:
            y_ = self.xdata.e_curve.y
        return y_

    def __repr__(self):
        from molass_legacy._MOLASS.DummyClasses import DummySd, DummyMapper, DummyJudgeHolder
        return '%s(%s, %s, %s, %s, %s, %s, %s)' % (self.__class__.__qualname__,
                                                   DummySd(),
                                                   DummyMapper(),
                                                   DummyJudgeHolder(),
                                                   str(self.paired_ranges),
                                                   "'xdata'",
                                                   str(self.decomp_info),
                                                   str(self.slice_))

class PreviewOptions:
    def __init__(self, conc_depend=None, aq_smoothness=False, aq_positivity=False, use_elution_models=True):
        self.conc_depend = conc_depend
        self.aq_smoothness = aq_smoothness
        self.aq_positivity = aq_positivity
        self.use_elution_models = use_elution_models
    def __repr__(self):
        return 'PreviewOptions(%s, %s, %s)' % (str(self.aq_smoothness), str(self.aq_positivity), str(self.use_elution_models))
