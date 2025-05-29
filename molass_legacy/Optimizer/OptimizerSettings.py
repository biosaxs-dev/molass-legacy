"""
    OptimizerSettings.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import logging
import os
from collections import OrderedDict
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.Experiment.ColumnTypes import PORESIZE_ALLOWANCE
from .TheUtils import get_optimizer_folder

def get_settings_path(optimizer_folder=None):
    if optimizer_folder is None:
        optimizer_folder = get_optimizer_folder()
    return os.path.join(optimizer_folder, "opt_settings.txt")

OPT_DEFAULT_SETTINGS = None

# task: unify these values with those in SerialSettings
def delayed_settings_init():
    from molass_legacy.Experiment.DataUtils import get_columntype, get_default_num_plates
    global OPT_DEFAULT_SETTINGS, OPT_DEFAULT_DICT

    default_columntype = get_columntype()
    poresize = default_columntype.poresize

    OPT_DEFAULT_SETTINGS = [
        ("param_init_type", 0),
        ("bounds_type", 0),
        ("elution_model", 0),       # not get_setting("elution_model") for backward compatibility
        ("t0_upper_bound", None),
        ("uv_basemodel", 0),
        ("poreexponent", 0),        # set_setting("poreexponent", None) doesn't seem to work. why?
        ("ratio_interpretation", 0),
        ("separate_eoii", 0),
        ("separate_eoii_type", 0),
        ("separate_eoii_flags", []),
        ("apply_sf_bounds", 0),
        ("sf_bound_ratio", 1.0),
        ("apply_rg_discreteness", 0),
        ("rg_discreteness_unit", 1.0),
        ("apply_mw_integrity", 0),
        ("mw_integer_ratios", None),
        ("avoid_peak_fronting", 0),
        ("optimization_method", 0),

        # SEC parameters
        ("exclusion_limit", default_columntype.excl_limit),
        ("poresize", poresize),
        ("poresize_bounds", (poresize - PORESIZE_ALLOWANCE, poresize + PORESIZE_ALLOWANCE)),
        ("num_plates_pc", int(round(get_default_num_plates()*0.3)))
        ]

    OPT_DEFAULT_DICT = dict(OPT_DEFAULT_SETTINGS)

class OptimizerSettings:
    def __init__(self, **kwargs):
        if OPT_DEFAULT_SETTINGS is None:
            delayed_settings_init()

        self.logger = logging.getLogger(__name__)
        self._dict = OrderedDict()

        if len(kwargs) > 0:
            for key, value in OPT_DEFAULT_SETTINGS[0:2]:
                self._dict[key] =  kwargs.pop(key, value)
            assert len(kwargs) == 0
            for key, _ in OPT_DEFAULT_SETTINGS[2:]:
                self._dict[key] = get_setting(key)

    def save(self, path=None):
        if path is None:
            path = get_settings_path()
        with open(path, "w") as fh:
            fh.write(self.__repr__())

    def load(self, path=None, optimizer_folder=None):
        from molass_legacy.KekLib.EvalUtils import eval_file
        if path is None:
            path = get_settings_path(optimizer_folder=optimizer_folder)

        def replacer(code):
            return code.replace("OptimizerSettings", "OrderedDict")

        self._dict = eval_file(path,
                               locals_=globals(),   # for OrderedDict
                               replacer=replacer)

        for key, _ in OPT_DEFAULT_SETTINGS[2:]:
            value = self.get(key)
            # print("set_setting(%s, %s)" % (key, str(value)))
            set_setting(key, value)

        separate_eoii = self.get("separate_eoii")
        if separate_eoii:
            key = "separate_eoii_type"
            separate_eoii_type = self.get(key)
            if separate_eoii_type == 0:
                value = 1
                self._dict[key] = value
                set_setting(key, value)
                self.logger.info("%s has been set to %d over %d to make it backward compatible.", key, value, separate_eoii_type)

        self.logger.info("optimizer settings have been restored as %s", self.__repr__())

    def keys(self):
        return [k for k, v in OPT_DEFAULT_SETTINGS]

    def get(self, key):
        return self._dict.get(key, OPT_DEFAULT_DICT[key])  #

    def __repr__(self):
        return str(self._dict).replace("OrderedDict", "OptimizerSettings")

def get_advanced_settings_text():
    # task: make this complete
    ratio_interpretation = get_setting("ratio_interpretation")
    separate_eoii = get_setting("separate_eoii")
    separate_eoii_type = get_setting("separate_eoii_type")
    separate_eoii_flags = get_setting("separate_eoii_flags")
    apply_sf_bounds = get_setting("apply_sf_bounds")
    sf_bound_ratio = get_setting("sf_bound_ratio")
    avoid_peak_fronting = get_setting("avoid_peak_fronting")

    if separate_eoii_type == 2:
        flags_text = ": %s" % str(separate_eoii_flags)
    else:
        flags_text = ""

    if separate_eoii_type > 0:
        flags_text += ", bounded with %g" % sf_bound_ratio if apply_sf_bounds else ""

    ret_text = ""
    for flag, text in [ (ratio_interpretation, "rational interpretation"),
                        (separate_eoii, "separating effect of interparticle interactions" + flags_text),
                        (avoid_peak_fronting, "avoiding peak fronting"),
                        ]:
        if flag:
            if ret_text > "":
                ret_text += ", "
            ret_text += text

    return ret_text
