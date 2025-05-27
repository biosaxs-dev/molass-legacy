"""
    Batch/DataWalk.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
from molass_legacy.SerialAnalyzer.DataUtils import get_pytools_folder, serial_folder_walk
from molass_legacy.Batch.LiteBatch import LiteBatch
from molass_legacy.KekLib.ExceptionTracebacker import log_exception
from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings, set_setting

class CallbackWrapper:
    """
    This wrapper adjust the arguments of callback function
    so as to match the signature of the old-style callback function.
    """
    def __init__(self, callback, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.batch = LiteBatch()
        self.callback = callback
        self.ignore_error = kwargs.get("ignore_error", True)
        self.start_folder = kwargs.get("start_folder", None)
        self.counters = kwargs.get("counters", None)
        self.started = False

    def __call__(self, xr_folder, uv_folder=None, plot=False):
        if self.counters is None:
            count = None
        else:
            count = self.counters[0]
            self.counters[0] += 1
        if self.started:
            pass
        else:
            if self.start_folder is not None:
                if xr_folder.find(self.start_folder) < 0:
                    print("skipping", xr_folder)
                    return True, None
            self.started = True

        clear_temporary_settings()
        set_setting("in_folder", xr_folder)
        set_setting("test_pattern", 0)  # this is to avoid the popping up of the dialog box in irregular cases
        try:
            ret = self.callback(xr_folder, self.batch, count)
        except:
            log_exception(self.logger, "Error occurred during processing of %s: " % xr_folder, n=10)
            ret = self.ignore_error
        return ret, None

def do_a_folder(callback, batch, in_folder, count):
    set_setting("in_folder", in_folder)
    callback(in_folder, batch, count)
    
def test_data_walk(callback, **kwargs):
    exact_folder = kwargs.get("exact_folder", None)
    in_folder_file = kwargs.get("in_folder_file", None)
    if exact_folder is None and in_folder_file is None:
        root_folder = kwargs.get("root_folder", None)
        if root_folder is None:
            root_folder = get_pytools_folder()

        wrapper = CallbackWrapper(callback, **kwargs)
        serial_folder_walk(root_folder, wrapper)
    else:
        batch = LiteBatch()       
        if in_folder_file is None:
            # i.e., exact_folder is not None
            import os
            import sys
            this_dir = os.path.dirname( os.path.abspath( __file__ ) )
            sys.path.append( this_dir + '/../../..' )
            from TestEnv import get_data_folder    
            in_folder = get_data_folder(exact_folder)
            do_a_folder(callback, batch, in_folder, 0)
        else:
            with open(in_folder_file) as fh:
                for k, line in enumerate(fh):
                    in_folder = line[:-1].split(',')[0]
                    clear_temporary_settings()
                    set_setting("test_pattern", 0)
                    do_a_folder(callback, batch, in_folder, k)