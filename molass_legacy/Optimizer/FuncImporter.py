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
    """Import (hot-reloading) the ObjectiveFunctions class for ``class_code``.

    Returns ``None`` only when the model is genuinely unsupported, i.e. there
    is no ``ObjectiveFunctions/<class_code>.py`` module on disk -- the one
    case callers (e.g. ``construct_legacy_optimizer``) are meant to turn into
    a "model not supported" error.

    Any other failure (BasicOptimizer or its dependencies failing to
    (re)load, the module existing but raising on import, a missing/renamed
    class attribute, ...) is a real bug, not an unsupported model, and is
    re-raised after logging so the true cause is visible instead of being
    silently swallowed into a misleading "not supported" message.

    One such real bug (molass-legacy#102): editing ``BasicOptimizer.py`` (or
    a module it imports, e.g. ``NumericalUtils.py``) while a long-running
    process has it loaded can leave a stale, already-imported dependency
    module in ``sys.modules`` -- ``reload()`` only re-executes the named
    module, not its transitive imports -- so a newly added
    ``from .NumericalUtils import compute_uv_domain_mask`` can raise
    ``ImportError`` here even though the code is correct on disk.
    """
    from molass_legacy.KekLib.ExceptionTracebacker import log_exception

    try:
        import molass_legacy.Optimizer.BasicOptimizer
        reload(molass_legacy.Optimizer.BasicOptimizer)
    except Exception:
        # BasicOptimizer is shared infrastructure for every model -- a
        # failure here is never "model not supported".
        log_exception(logger, "reloading BasicOptimizer: ", n=5)
        raise

    module_name = "molass_legacy.%s.%s" % (OBJFUNC_DIRNAME, class_code)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            # No such ObjectiveFunctions module for this model -- genuinely
            # unsupported.
            log_exception(logger, "importing ObjectiveFunctions: ", n=5)
            return None
        # The module exists but one of ITS imports is missing -- a real bug.
        log_exception(logger, "importing ObjectiveFunctions: ", n=5)
        raise

    try:
        module = reload(module)
        class_ = getattr(module, class_code)
    except Exception:
        log_exception(logger, "importing ObjectiveFunctions: ", n=5)
        raise
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
                if not ("G1000" <= class_code < "G1400"):
                    continue
            elif elution_model == 5:
                if class_code != "G2020":
                    continue
            elif elution_model == 6:
                if class_code != "G1400":
                    continue
            elif elution_model == 7:
                if class_code != "G1500":
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
