"""
    Optimizer.FuncImporter.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import os
import glob
import re
from importlib import import_module, reload

OBJFUNC_DIRNAME = "ObjectiveFunctions"

def import_objective_function(class_code, logger=None):
    try:
        import molass_legacy.Optimizer.BasicOptimizer
        reload(molass_legacy.Optimizer.BasicOptimizer)

        module = import_module("molass_legacy.%s.%s" % (OBJFUNC_DIRNAME, class_code))
        module = reload(module)
        class_ = getattr(module, class_code)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(logger, "importing ObjectiveFunctions: ", n=5)
        class_ = None
    return class_

def get_objective_function_info(logger=None, default_func_code=None, debug=False):
    # note that default_objective_func can be changed depending on elution model
    # so, making this to a singleton needs careful streatment of such changes

    if debug:
        import molass_legacy.Optimizer.BasicOptimizer as base_opt
        reload(base_opt)

    from molass_legacy._MOLASS.SerialSettings import get_setting
    from molass_legacy.KekLib.BasicUtils import Struct

    elution_model = get_setting('elution_model')
    if default_func_code is None:
        # this should have been set in OptStrategyDialog.py
        default_func_code = get_setting('default_objective_func')
    func_dict = {}
    key_list = []
    default_index = None
    file_re = re.compile(r'\W(\w\d+)\.py')
    upper_dir = os.path.dirname(os.path.dirname(__file__))
    if debug:
        print("upper_dir:", upper_dir)
    for k, file in enumerate(sorted(glob.glob(upper_dir + r"\%s\*.py" % OBJFUNC_DIRNAME))):
        if debug:
            print("importing objective function %d: %s" % (k, file))
        m = file_re.search(file)
        if m:
            class_code = m.group(1)
            if debug:
                print("class_code:", class_code)
            if elution_model == 0:
                if class_code >= "G0500":
                    continue
            elif elution_model == 1:
                if class_code < "G1000":
                    continue
            elif elution_model == 2:
                if class_code < "G0500" or class_code >= "G0600":
                    continue
            elif elution_model == 3:
                if class_code < "G0600" or class_code >= "G0700":
                    continue
            elif elution_model == 4:
                if class_code < "G0700" or class_code >= "G0800":
                    continue
            elif elution_model == 5:
                if class_code < "G2000" or class_code >= "G3000":
                    continue
            else:
                assert False

            if class_code == default_func_code:
                default_index = len(key_list)
            class_ = import_objective_function(class_code, logger)
            docstr = class_.__doc__
            key_str = ' : '.join([class_code, docstr])
            func_dict[key_str] = class_
            key_list.append(key_str)
            logger.info("function %s appended", class_code)

    func_info = Struct(func_dict=func_dict, key_list=key_list, default_index=default_index)

    return func_info
