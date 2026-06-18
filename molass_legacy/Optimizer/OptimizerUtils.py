"""
    OptimizerUtils.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
MODEL_NAME_DICT = {
    "G0346" : "EGH",
    "G0367" : "EGH",
    "G1100" : "SDM(exp)",
    "G1200" : "SDM(mono)",
    "G1300" : "SDM(lognormal)",
    "G1400" : "LKM",
    "G1500" : "GRM",
    "G2010" : "NEDM",
    "G2020" : "EDM",
}

def get_model_name(class_code):
    return MODEL_NAME_DICT.get(class_code, str(class_code))

def get_function_code(model_name):
    model_name = model_name.upper()
    for code, name in MODEL_NAME_DICT.items():
        if name == model_name:
            return code
    return None

METHOD_NAMES      = None   # populated below from Registry
IMPL_METHOD_NAMES = None   # populated below from Registry

def _load_registry():
    global METHOD_NAMES, IMPL_METHOD_NAMES
    try:
        from molass_legacy.Solvers.Registry import METHOD_NAMES as _mn, IMPL_METHOD_NAMES as _imn
        METHOD_NAMES      = _mn
        IMPL_METHOD_NAMES = _imn
    except ImportError:
        # Fallback for environments without Registry (should not happen)
        METHOD_NAMES      = ["BH", "NS", "MCMC", "SMC", "PYMC", "CMA", "DE", "NSGA2"]
        IMPL_METHOD_NAMES = ["bh", "ultranest", "emcee", "pyabc", "pymc", "cma", "de", "nsga2"]

_load_registry()

def get_method_name():
    from molass_legacy._MOLASS.SerialSettings import get_setting
    return METHOD_NAMES[get_setting("optimization_method")]

def get_impl_method_name(nnn, method=None):
    if method is None:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        method = get_setting("optimization_method")
    if 2 <= method <= 3:
        # MCMC (2) and SMC (3) alternate between sub-implementations
        # based on the job index (nnn).  BH/NS/CMA map directly.
        r = method % 2
        method = (nnn + r) % 2
    return IMPL_METHOD_NAMES[method]

def _apply_library_recommendation(corrected_sd, fallback_num_peaks):
    """Call molass-library recommend_decomposition_options() and update settings.

    Used by show_peak_editor_impl() when "Automatic" peak recognition is
    selected.  Falls back silently to *fallback_num_peaks* on any error.

    Returns the recommended (or fallback) number of components.
    """
    from molass_legacy._MOLASS.SerialSettings import set_setting
    try:
        from molass.Bridge.SdAdapter import make_ssd_from_corrected_sd
        from molass.Decompose.Recommend import recommend_decomposition_options
        lib_ssd = make_ssd_from_corrected_sd(corrected_sd)
        opts = recommend_decomposition_options(lib_ssd.xr)
        n = opts['num_components']
        if 'proportions' in opts:
            ratios = opts['proportions']
            set_setting('proportional_peaks',
                        ','.join(str(int(r)) for r in ratios))
        else:
            set_setting('proportional_peaks', None)
        set_setting('interparticle_ranks', opts.get('ranks', None))
        return n
    except Exception:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, '_apply_library_recommendation failed: ', n=5)
        return fallback_num_peaks


def show_peak_editor_impl(strategy_dialog, dialog, pe_proxy=None, pe_ready_cb=None, apply_cb=None, debug=True):
    from molass_legacy._MOLASS.SerialSettings import get_setting
    if debug:
        from importlib import reload
        import molass_legacy.SecSaxs.DataTreatment
        reload(molass_legacy.SecSaxs.DataTreatment)
        import molass_legacy.Peaks.PeakEditor
        reload(molass_legacy.Peaks.PeakEditor)
    from molass_legacy.SecSaxs.DataTreatment import DataTreatment
    from molass_legacy.Peaks.PeakEditor import PeakEditor

    parent = dialog.parent

    if pe_proxy is None:
        exact_num_peaks = strategy_dialog.get_num_peaks()
        strict_sec_penalty, correction, trimming, unified_baseline_type = strategy_dialog.get_options()

        treat = DataTreatment(route="v2", trimming=trimming, correction=correction, unified_baseline_type=unified_baseline_type)
        sd = dialog.serial_data
        pre_recog = dialog.pre_recog
        trimmed_sd = treat.get_trimmed_sd(sd, pre_recog)
        corrected_sd = treat.get_corrected_sd(sd, pre_recog, trimmed_sd)

        # --- Library recommendation override for "Automatic" recognition ---
        # When the user selects method=0 (Automatic), delegate ncomp detection
        # and proportional/xr_peakpositions decision to recommend_decomposition_options().
        # This also detects interparticle effects (ranks=[2]) via the Guinier test.
        if strategy_dialog is not None and strategy_dialog.peak_recog_method.get() == 0:
            exact_num_peaks = _apply_library_recommendation(corrected_sd, exact_num_peaks)
        # -------------------------------------------------------------------

        dialog.grab_set()   # temporary fix to the grab_release problem

        pe = PeakEditor(parent, trimmed_sd, treat.pre_recog, corrected_sd, treat, exact_num_peaks=exact_num_peaks, strict_sec_penalty=strict_sec_penalty)
        if pe_ready_cb is not None:
            pe_ready_cb(pe)

        pe.show()
        if not pe.applied:
            return

        settings = None
    else:
        pe_proxy.load_settings()
        trimmed_sd = pe_proxy.get_sd()
        treat = pe_proxy.get_treat()
        pe = pe_proxy

    from molass_legacy.Optimizer.InitialInfo import InitialInfo
    if debug:
        from importlib import reload
        import molass_legacy.Optimizer.FullOptDialog
        reload(molass_legacy.Optimizer.FullOptDialog)
    from molass_legacy.Optimizer.FullOptDialog import FullOptDialog

    optinit_info = InitialInfo(trimmed_sd, treat=treat, pe=pe)
    dialog.grab_set()   # temporary fix to the grab_release problem, to be inspected to eventually remove this
    dialog.fullopt_dialog = FullOptDialog(parent, dialog, optinit_info)
    if apply_cb is not None:
        apply_cb(dialog.fullopt_dialog)
    dialog.fullopt_dialog.show()

class OptimizerResult:
    def __init__(self, x=None, nit=None, nfev=None):
        self.x = x
        self.nit = nit
        self.nfev = nfev

if __name__ == '__main__':
    for nnn, method in [(0, 0), (0, 1), (5, 2), (5, 3)]:
        if method < 2:
            print((nnn, method), get_impl_method_name(nnn, method=method))
        else:
            for n in range(nnn):
                print((n, method), get_impl_method_name(n, method=method))
