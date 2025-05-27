"""

    Test.TestAllData.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
from molass_legacy._MOLASS.SerialSettings import set_setting, clear_temporary_settings
from DataUtils import get_pytools_folder, serial_folder_walk
from molass_legacy.Batch.StandardProcedure import StandardProcedure

counter = 0
def test_all_data(callback_all, analysis_copy=False, logger=None, restart_folder=None, index_csv_fh=None):
    set_setting("test_pattern", 0)

    if restart_folder is not None:
        restart_folder = restart_folder.replace("\\", "/")

    sp = StandardProcedure()

    def walk_callback(in_folder, uv_folder=None, plot=False):
        global counter
        if restart_folder is not None:
            if in_folder < restart_folder:
                counter += 1
                print("skipping", in_folder)
                return True, None

        clear_temporary_settings()
        if analysis_copy:
            from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
            sd_ = sp.load_old_way(in_folder)
            try:
                pre_recog = PreliminaryRecognition(sd_)
                sd = sd_._get_analysis_copy_impl(pre_recog)
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(logger, "failed to get an analysis_copy: ", n=10)
                sd = sd_
        else:
            sd = sp.load_old_way(in_folder)

        callback_all(in_folder, sd, counter, index_csv_fh)
        counter += 1
        return True, None

    data_folder = get_pytools_folder() + '/Data'
    serial_folder_walk(data_folder, walk_callback)
