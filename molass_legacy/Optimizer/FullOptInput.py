"""
    Optimizer.FullOptInput.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF

    First note that FullOptInput is constructed in the both processes, namely
        ・ monitor process and
        ・ optimizer process,
    as detailed below.

    (1) in the monitor process

        ・ OptimizerUtils.show_peak_editor_impl
            ・ PeakEditor.prepare_rg_curve → FullOptInput.get_dsets()
            ・ InitialInfo.__init__,
                ・ dsets = pe.get_dsets()
                ・ dsets.relocate_rg_folder()
        ・ show_result_folder_selector_impl → FullOptInput.get_dsets()

    (2) in the optimizer process

        ・ OptimizerMain → FullOptInput.get_dsets()
"""
import os
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

class FullOptInput:
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.update_attributes(**kwargs)
        if self.trimming_txt is not None:
            self.restore_trimming_info()

    def update_attributes(self, **kwargs):
        in_folder = kwargs.pop('in_folder', None)
        if in_folder is None:
            in_folder = get_setting('in_folder')
        self.in_folder = in_folder
        self.dsets = kwargs.pop('dsets', None)
        self.rg_folder = kwargs.pop('rg_folder', None)
        self.trimming_txt = kwargs.pop('trimming_txt', None)
        self.sd = kwargs.pop('sd', None)
        self.corrected_sd = kwargs.pop('corrected_sd', None)
        self.mapper = kwargs.pop('mapper', None)
        self.mapped_params = kwargs.pop('mapped_params', None)
        self.peak_params = kwargs.pop('peak_params', None)

    def get_sd(self, in_folder=None):
        if in_folder is None:
            in_folder = self.in_folder

        sd = self.sd
        if sd is None :
            if in_folder is None:
                # i.e., no input data info
                sd = None
            else:
                sd, corrected_sd = self.get_sd_from_folder(in_folder)
                self.sd = sd
                self.corrected_sd = corrected_sd
        return sd

    def get_dsets(self, in_folder=None, progress_cb=None, compute_rg=False, possibly_relocated=False, debug=False):
        sd = self.get_sd(in_folder=in_folder)
        if sd is None:
            return None

        dsets = self.dsets
        if dsets is None:
            if debug:
                from importlib import reload
                import Optimizer.OptDataSets
                reload(Optimizer.OptDataSets)
            from .OptDataSets import OptDataSets
            corrected_sd = self.corrected_sd
            self.dsets = dsets = OptDataSets(sd, corrected_sd,
                                    rg_folder=self.rg_folder,
                                    progress_cb=progress_cb,
                                    compute_rg=compute_rg,
                                    possibly_relocated=possibly_relocated,
                                    )

        return dsets

    def get_sd_from_folder(self, in_folder):
        from .TheUtils import get_sd_from_folder_impl
        sd, corrected_sd, self.treat = get_sd_from_folder_impl(in_folder, self.logger)
        return sd, corrected_sd

    def get_base_curve(self):
        return self.treat.get_base_curve()

    def get_mapped_params(self):
        if self.mapped_params is not None:
            return self.mapped_params

        if self.mapper is None:
            self.sp.get_corrected_sd(proxy=False)
            self.mapper = self.sp.mapper
        mapper = self.mapper
        mapped_info = mapper.get_mapped_info()
        opt_results = mapped_info.opt_results
        A, B, _ = np.average(np.array(opt_results), axis=0)
        num_peaks_mapping = len(opt_results)
        self.logger.info("num_peaks_mapping=%d", num_peaks_mapping)
        return A, B, num_peaks_mapping

    def restore_trimming_info(self):
        from molass_legacy.Trimming import restore_trimming_info_impl
        restore_trimming_info_impl(self.trimming_txt, self.logger)
