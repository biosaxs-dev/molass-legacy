"""
    Optimizer.FullOptResult.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import os
import re
import logging
import numpy as np
from molass_legacy.Trimming import restore_trimming_info_impl
from molass_legacy.Baseline.BaselineUtils import create_xr_baseline_object
from .StateSequence import read_callback_txt_impl, read_seed_txt
from .FuncImporter import import_objective_function
from molass_legacy.SecSaxs.DataTreatment import DataTreatment
from .OptJobInfo import OptJobInfo
from .OptDataSets import OptDataSets
from .OptLogFile import OptLogFile
from .TheUtils import FILES

OPTIONAL_FILES = [
    "composite_info.txt",
    ]


class FullOptResult:
    def __init__(self, orig_sd, pre_recog, folder, set_folder_setting=False):
        self.logger = logging.getLogger(__name__)
        self.logger.info("constructing a result object from %s", folder)
        self.orig_sd = orig_sd          # not yet trimmed
        self.pre_recog = pre_recog
        # self.folder = folder.replace("/", "\\") # internal path delimiter should be \?
        self.folder = folder
        self.optimizer_folder = self.get_optimizer_folder_impl()
        if set_folder_setting:
            # temporary fix for Batch.BatchUtils.load_v2_result
            from molass_legacy._MOLASS.SerialSettings import set_setting
            set_setting('optimizer_folder', self.optimizer_folder)
        trimming_txt = os.path.join(folder, FILES[6])
        restore_trimming_info_impl(trimming_txt, self.logger)
        self.load_info()
        self.best_index = None

    def load_composite(self):
        composite_info_txt = os.path.join(self.folder, OPTIONAL_FILES[0])
        if os.path.exists(composite_info_txt):
            from .CompositeInfo import CompositeInfo
            composite = CompositeInfo()
            composite.load(composite_info_txt)
        else:
            composite = None
        return composite

    def get_log_file(self):
        optimizer_log = os.path.join(self.folder, FILES[3])
        return OptLogFile(optimizer_log)

    def get_class_code(self, log_file):
        self.class_code =  log_file.class_code
        return self.class_code

    def load_callback_info(self):
        callback_txt = os.path.join(self.folder, FILES[0])
        fv_list, x_list = read_callback_txt_impl(callback_txt)
        fv_array = np.array([rec[1] for rec in fv_list])
        return fv_list, x_list, fv_array
    
    def load_init_params(self):
        init_params_txt = os.path.join(self.folder, FILES[2])
        return np.loadtxt(init_params_txt)

    def load_info(self, debug=False):
        from .OptimizerSettings import OptimizerSettings

        optimizer_folder = self.get_optimizer_folder()

        self.settings = settings = OptimizerSettings()
        settings.load(optimizer_folder=optimizer_folder)    # must be loaded before fullopt construction

        self.treat = treat = DataTreatment()
        treat.load(optimizer_folder=optimizer_folder)
        # self.sd_copy = treat.get_treated_sd(self.orig_sd, self.pre_recog)
        self.sd_copy = treat.get_trimmed_sd(self.orig_sd, self.pre_recog)
        corrected_sd = treat.get_corrected_sd(self.orig_sd, self.pre_recog, self.sd_copy)
        uv_base_curve = treat.get_base_curve()

        self.dsets = OptDataSets(self.sd_copy, corrected_sd, possibly_relocated=True, current_folder=self.folder)
        composite = self.load_composite()
        log_file = self.get_log_file()
        class_code = self.get_class_code(log_file)
        self.fullopt_class = import_objective_function(class_code, self.logger)
        self.drift_type = "linear"

        info = OptJobInfo()
        info.load(self.folder)
        self.n_components = info.nc

        xr_base_curve = create_xr_baseline_object()
        self.fullopt = self.fullopt_class(self.dsets, self.n_components,
                            uv_base_curve=uv_base_curve,
                            xr_base_curve=xr_base_curve,
                            qvector=self.sd_copy.qvector,   # sd_copy: trimmed_sd
                            wvector=self.sd_copy.lvector,
                            composite=composite
                            )

        if self.fullopt.is_stochastic():
            from molass_legacy.Estimators.SdmEstimatorProxy import SdmEstimatorProxy
            estimator = SdmEstimatorProxy(self.folder)
            self.fullopt.params_type.set_estimator(estimator)      # reconsider the neccesity of this line

        fv_list, x_list, fv_array= self.load_callback_info()
        init_params = self.load_init_params()

        if log_file.version_date < "2022-09-20":

            if not debug:
                import molass_legacy.KekLib.CustomMessageBox as MessageBox
                MessageBox.showerror("Not Supported Error",
                    "View of results made with versions earlier than 2022-09-20\n"
                    "is not supported!",
                    parent=None,
                    )
                assert False

            adjuster = self.fullopt.params_type.get_adjuster()
            adjuster.fit(self.fullopt, fv_array, x_list)
            init_params = adjuster.convert(init_params)
            x_list = adjuster.convert(x_list)

        self.init_params = init_params
        self.fv_list = fv_list
        self.fv_array = fv_array
        self.x_list = x_list
        self.n_iterations = len(fv_list) - 2

        seed_txt = os.path.join(self.folder, FILES[5])
        self.seed = read_seed_txt(seed_txt)
        self.composite = composite

        bounds_txt = os.path.join(self.folder, FILES[7])
        if os.path.exists(bounds_txt):
            real_bounds = np.loadtxt(bounds_txt)
        else:
            real_bounds = None

        self.fullopt.prepare_for_optimization(self.init_params, real_bounds=real_bounds)

        self.logger.info("load_info: n_components=%d, len(init_params)=%d, composite=%s", self.n_components, len(self.init_params), str(composite))

    def get_optimizer(self):
        return self.fullopt

    def get_optimizer_folder_impl(self):
        return os.path.dirname(os.path.dirname(self.folder))

    def get_optimizer_folder(self):
        return self.optimizer_folder

    def get_optimizer_jobs_folder(self):
        return os.path.join(self.optimizer_folder, "jobs")

    def get_jobnames(self):
        nodes = os.listdir(self.get_optimizer_jobs_folder())
        return nodes

    def get_num_jobs(self):
        nodes = self.get_jobnames()
        return len(nodes)

    def get_rg_folder(self):
        return os.path.join(self.optimizer_folder, "rg-curve")

    def get_init_info(self):
        from molass_legacy.Optimizer.InitialInfo import InitialInfo
        return InitialInfo(result=self)

    def get_demo_index(self):
        return self.n_iterations

    def get_job_name(self):
        import re
        job_name_re = re.compile(r"(\d+)$")
        m = job_name_re.search(self.folder)
        if m:
            name = m.group(1)
        else:
            name = "???"
        return name

    def get_job_list(self):
        jobs_folder = self.get_optimizer_jobs_folder()
        job_list = []
        for name in self.get_jobnames():
            try:
                folder = os.path.join(jobs_folder, name)
                info = OptJobInfo()
                info.load(folder)
                job_list.append(info)
            except:
                # may be the folder is not complete: missing "optimizer.log", etc.
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(self.logger, "get_job_list: ")
        return job_list

    def get_best_params(self):
        if self.best_index is None:
            self.best_index = np.argmin(self.fv_array)
        return self.x_list[self.best_index]

    def get_result_iterator(self, all=False):
        from molass_legacy.Optimizer.ParamsIterator import create_iterator
        if all:
            for i, params in enumerate(self.x_list):
                yield i, params
        else:
            for i in create_iterator(self.fv_array):
                yield i, self.x_list[i]

    def debug_plot(self, params):
        optimizer = self.get_optimizer()
        # def objective_func(self, p, plot=False, debug=False, fig_info=None, axis_info=None, return_full=False):
        optimizer.objective_func(params, plot=True)

def get_result_from_workfolder(workfolder, replace_specs=None):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.LiteBatch import LiteBatch
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition

    in_data_path = os.path.join(workfolder, FILES[1])
    with open(in_data_path, "r") as f:
        in_folder = f.read().strip().split('=')[1]

    if replace_specs is not None:
        for k, v in replace_specs:
            in_folder = in_folder.replace(k, v)

    analysis_folder = '\\'.join(workfolder.split('\\')[:-3])

    print("analysis_folder=", analysis_folder)
    print("workfolder=", workfolder)
    print("in_folder=", in_folder)
    set_setting("in_folder", in_folder)
    set_setting("analysis_folder", analysis_folder)

    batch = LiteBatch()
    sd = batch.load_data(in_folder)
    pre_recog = PreliminaryRecognition(sd)
    result = FullOptResult(sd, pre_recog, workfolder)
    return result