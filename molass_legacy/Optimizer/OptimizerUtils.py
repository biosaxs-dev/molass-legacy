"""
    OptimizerUtils.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
MODEL_NAME_DICT = {
    "G0346" : "EGH",
    "G1100" : "SDM",
    "G2010" : "EDM",
}
def get_model_name(class_code):
    return MODEL_NAME_DICT[class_code]

METHOD_NAMES = ["BH", "NS", "MCMC", "SMC"]
def get_method_name():
    from molass_legacy._MOLASS.SerialSettings import get_setting
    return METHOD_NAMES[get_setting("optimization_method")]

IMPL_METHOD_NAMES = ["bh", "ultranest", "emcee", "pyabc", "pymc"]
def get_impl_method_name(nnn, method=None):
    if method is None:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        method = get_setting("optimization_method")
    if method >= 2:
        r = method % 2
        method = (nnn + r) % 2
    return IMPL_METHOD_NAMES[method]

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
