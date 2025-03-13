"""
    StandardProcedure.py

    Copyright (c) 2020-2024, SAXS Team, KEK-PF
"""
import os
import logging
import copy
import numpy as np
from time import time
from molass_legacy.DataStructure.MeasuredData import MeasuredData
from molass_legacy._MOLASS.SerialSettings import set_setting
from molass_legacy.SerialAnalyzer.AbnormalityCheck import bubble_check
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.Decomposer.DecompUtils import CorrectedBaseline
from molass_legacy.Models.ElutionCurveModels import EGHA, EMGA
from molass_legacy.Decomposer.DecompUtils import decompose_elution_better, make_range_info_impl
from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util
from molass_legacy.Mapping.CurveSimilarity import CurveSimilarity

class MockEditor:
    def __init__(self):
        pass

    def update_range(self, *args):
        print('update_range:', *args)

class StandardProcedure:
    def __init__(self):
        self.logger = logging.getLogger()

    def load(self, folder, debug=False):
        self.folder = folder
        self.md = MeasuredData(folder)
        self.sd = None
        self.old_way = False
        if debug:
            self.md.plot()

    def load_old_way(self, folder):
        from molass_legacy.SerialAnalyzer.SerialDataLoader import SerialDataLoader
        from molass_legacy.SerialAnalyzer.SerialDataUtils import get_uv_filename
        self.folder = folder
        loader = SerialDataLoader()
        uv_file = get_uv_filename(folder)
        print("uv_file=", uv_file)
        if uv_file is None:
            set_setting('use_xray_conc', 1)
            loader.load_xray_data_only(folder)
        else:
            set_setting('use_xray_conc', 0)
            loader.load_from_folders(folder, uv_folder=folder, uv_file=uv_file )
        loader.wait_until_ready()
        self.sd = loader.get_current_object()
        print("sd loaded as %s in the old way" % type(self.sd))
        exclude = bubble_check(self.sd)
        if len(exclude) > 0:
            self.sd.exclude_intensities(exclude)
        self.old_way = True
        return self.sd

    def get_sd(self, whole=False, debug=False, set_offsets=True):
        from molass_legacy.SerialAnalyzer.SerialData import SerialData
        # datafiles, xray_array, uv_array, lvector, col_header, mtd_elution

        if self.old_way and self.sd is not None:
            return self.sd

        xr = self.md.xr

        if whole:
            i_slice = slice(0, None)
            j_slice = slice(0, None)
        else:
            i_slice = xr.i_slice
            j_slice = xr.j_slice
        datafiles = xr.files[j_slice]

        q = xr.vector[i_slice]
        shape = (len(datafiles), len(q), 3)
        xray_array = np.zeros(shape)
        xray_array[:,:,0] = np.array([ q for k in range(shape[0])])
        xray_array[:,:,1] = xr.data[i_slice, j_slice].T
        xray_array[:,:,2] = xr.error.data[i_slice, j_slice].T

        uv = self.md.uv
        if whole:
            i_slice = slice(None, None)
            j_slice = slice(None, None)
        else:
            i_slice = uv.i_slice
            j_slice = uv.j_slice
        uv_array = uv.data[i_slice, j_slice]
        lvector = uv.vector[i_slice]
        col_header = None   # implement if needed
        mtd_elution = None  # temp

        # restricted data
        data_info = copy.deepcopy([datafiles, xray_array, uv_array, lvector, col_header, mtd_elution])
        sd = SerialData(self.folder, self.folder, data_info=data_info)
        if set_offsets:
            sd.set_offsets_from_md(self.md)

        t0 = time()
        exclude = bubble_check(sd)
        print("took %.3g" % (time() - t0))
        if len(exclude) > 0:
            sd.exclude_intensities(exclude)

        if debug:
            md = MeasuredData(None, sd=sd)
            md.plot()

        self.sd = sd
        return sd

    def add_pre_recog(sp, sd):
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        pre_recog = PreliminaryRecognition(sd)
        sd.pre_recog = pre_recog
        return pre_recog

    def get_curve_similarity(self):
        uv_curve = self.sd.get_uv_curve()
        xr_curve = self.sd.get_xray_curve()
        cs = CurveSimilarity(uv_curve, xr_curve)
        return cs

    def get_corrected_sd(self, proxy=True, debug=False):
        from molass_legacy.Mapping.MapperConstructor import create_mapper
        sd_orig = self.get_sd(debug=debug)
        if proxy:
            sd_orig.set_prerecog_proxy(self.md)
            pre_recog = sd_orig.pre_recog
            sd = sd_orig
        else:
            from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
            pre_recog = PreliminaryRecognition(sd_orig)
            sd = sd_orig._get_analysis_copy_impl(pre_recog)

        """
        approximate uv-xr mapping has been already done in MeasuredData.
        however, no baseline correction yet.
        """

        self.mapper = create_mapper(None, sd, sd_orig, pre_recog)
        sd.apply_baseline_correction(self.mapper.get_mapped_info())

        if debug:
            md = MeasuredData(None, sd=sd)
            md.plot()

        conc_factor = compute_conc_factor_util()
        sd.set_mc_vector(self.mapper, conc_factor)
        return sd

    def get_lrf_args(self, sd):
        from Extrapolation.PreviewData import PreviewData, PreviewOptions

        decomp_needed = self.get_decomp_necessity(sd)
        print('decomp_needed=', decomp_needed)
        if decomp_needed:
            use_elution_models = True
            decomp_info = self.get_decomp_info(sd)
            control_info = decomp_info.get_range_edit_info(logger=self.logger)
            """
            note that ranges are not shifted.
            consider refactoring which moves make_range_info_impl into ret
            """
            paired_ranges = make_range_info_impl(decomp_info.opt_recs_uv, control_info)
        else:
            use_elution_models = False
            decomp_info = None
            x_curve = sd.get_xray_curve()
            paired_ranges = x_curve.get_default_paired_ranges()
        conc_factor = compute_conc_factor_util()
        pdata = PreviewData(sd=sd, mapper=self.mapper, paired_ranges=paired_ranges,
                                decomp_info=decomp_info, conc_factor=conc_factor)
        popts = popts = PreviewOptions(use_elution_models=use_elution_models)
        return pdata, popts

    def get_preview_results(self, sd, gui=False, parent=None, return_ctrl=False, use_pool=True):
        pdata, popts = self.get_lrf_args(sd)

        editor = MockEditor()
        if use_pool:
            from LRF.LrfResultPool import LrfResultPool
            pool = LrfResultPool(pdata, popts)
            pool.run_solver(parent)
            if gui:
                pool.show_dialog(parent)
            return pool.get_better_results()
        else:
            from Extrapolation.PreviewController import PreviewController
            ctrl = PreviewController(dialog=parent, editor=editor)
            ctrl.run_solver(parent, pdata, popts)
            if gui:
                ctrl.show_dialog()

            if return_ctrl:
                return ctrl
            else:
                return ctrl.solver_results

    def get_decomp_necessity(self, sd):
        self.get_guide_info(sd)
        return self.guide_info.decomp_proc_needed()

    def get_guide_info(self, sd):
        from molass_legacy.QuickAnalysis.AnalysisGuide import get_analysis_guide_info
        self.discomfort = sd.compute_scattering_baseline_discomfort()
        self.guide_info = get_analysis_guide_info(self.mapper, None, self.discomfort)

    def get_decomp_info(self, sd):
        corbase_info = CorrectedBaseline(sd, self.mapper)
        min_score = None
        min_ret = None
        for k, model in enumerate([EMGA(), EGHA()]):
            ret = decompose_elution_better(corbase_info, self.mapper, model,
                                    logger=self.logger)
            score = ret.compute_synthetic_score()
            print([k], 'score=', score)
            if min_score is None or score < min_score:
                min_score = score
                min_ret = ret

        return min_ret
