"""
    Simulative.SimulationCushion.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.Version import is_developing_version

def lnp_model_interactive(lrf_src, guess_info, parent):
    from importlib import reload
    import Simulative.SimulationMain
    reload(Simulative.SimulationMain)
    from Simulative.SimulationMain import lognormalpore_model_interactive    
    lognormalpore_model_interactive(lrf_src, guess_info, parent)

def approx_boundary(lrf_src, guess_info, parent):
    from importlib import reload
    import Simulative.ApproxBoundary
    reload(Simulative.ApproxBoundary)
    from Simulative.ApproxBoundary import demo_impl
    x = lrf_src.xr_x
    y = lrf_src.xr_y
    demo_impl(x, y, *guess_info, parent)

def several_studies(lrf_src, guess_info, parent):
    from importlib import reload
    import Simulative.SeveralStudies
    reload(Simulative.SeveralStudies)
    from Simulative.SeveralStudies import several_studies_impl    
    several_studies_impl(lrf_src, guess_info, parent)

def temporary_trial(lrf_src, guess_info, parent):
    from importlib import reload
    import Simulative.TemporaryTrial
    reload(Simulative.TemporaryTrial)
    from Simulative.TemporaryTrial import temporary_trial_impl    
    temporary_trial_impl(lrf_src, guess_info, parent)

def demo_cushion(lrf_src, guess_info, parent=None):
    if is_developing_version():
        extra_button_specs = [
            ("LNP Interactive", lambda: lnp_model_interactive(lrf_src, guess_info, parent)),
            ("Approx Boundary", lambda: approx_boundary(lrf_src, guess_info, parent)),
            ("Several Studies", lambda: several_studies(lrf_src, guess_info, parent)),
            ("Temporary Trial", lambda: temporary_trial(lrf_src, guess_info, parent)),
            ]
        lrf_src.draw(extra_button_specs=extra_button_specs)
    else:
        lnp_model_interactive(lrf_src, guess_info, parent)