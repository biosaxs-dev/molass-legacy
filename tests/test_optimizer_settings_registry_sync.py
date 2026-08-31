"""Regression test for issue #254: SOLVER_REGISTRY.settings_keys vs
OptimizerSettings.OPT_DEFAULT_SETTINGS schema drift.

Solvers/Registry.py's settings_keys (per-solver kwargs read from
SerialSettings at solve time) and OptimizerSettings.OPT_DEFAULT_SETTINGS (the
fixed schema serialized to opt_settings.txt for the subprocess to reload) are
two independently-maintained lists. A key present in the former but missing
from the latter is silently dropped for every in_process=False (subprocess,
default) run -- exactly what happened with 'de_tol' (molass-library#253).

Round-trips every SOLVER_REGISTRY settings_keys entry through
OptimizerSettings.save()/load() with a sentinel value, catching future
schema drift for any solver automatically instead of just the one instance
already fixed.
"""
import pytest


def _all_settings_keys():
    from molass_legacy.Solvers.Registry import SOLVER_REGISTRY
    keys = set()
    for entry in SOLVER_REGISTRY.values():
        keys.update(entry.settings_keys)
    return sorted(keys)


@pytest.mark.parametrize("key", _all_settings_keys())
def test_solver_settings_key_survives_save_load(tmp_path, key):
    from molass_legacy._MOLASS.SerialSettings import set_setting, get_setting
    from molass_legacy.Optimizer.OptimizerSettings import OptimizerSettings
    import molass_legacy.Optimizer.OptimizerSettings as opt_settings_mod

    if opt_settings_mod.OPT_DEFAULT_SETTINGS is None:
        opt_settings_mod.delayed_settings_init()

    assert key in opt_settings_mod.OPT_DEFAULT_DICT, (
        f"{key!r} is declared in SOLVER_REGISTRY[...].settings_keys but missing "
        "from OptimizerSettings.OPT_DEFAULT_SETTINGS -- it will be silently "
        "dropped from opt_settings.txt for every in_process=False run. "
        "Add it to OPT_DEFAULT_SETTINGS in Optimizer/OptimizerSettings.py "
        "(see molass-library#253/#254)."
    )

    sentinel = "__sentinel_254__"
    set_setting(key, sentinel)

    settings = OptimizerSettings(
        param_init_type=0, bounds_type=0, elution_model=0,
        optimization_method=0, separate_eoii_flags=[],
        ns_narrow_bounds=True, ns_adaptive_nsteps=False, ns_nsteps=None,
    )
    path = tmp_path / "opt_settings.txt"
    settings.save(path=str(path))

    # Reset before reload so a pass can only mean the value truly came from disk.
    set_setting(key, None)

    loaded = OptimizerSettings()
    loaded.load(path=str(path))

    assert get_setting(key) == sentinel, (
        f"{key!r} did not survive the OptimizerSettings.save()/load() round-trip "
        "-- schema drift between SOLVER_REGISTRY.settings_keys and "
        "OPT_DEFAULT_SETTINGS (molass-library#254)."
    )
