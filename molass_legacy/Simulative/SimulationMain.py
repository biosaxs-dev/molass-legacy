"""
    Simulative.SimulationMain.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def lognormalpore_model_interactive(lrf_src, guess_info, parent=None):
    import molass_legacy.KekLib.DebugPlot as plt
    from importlib import reload
    import Simulative.LognormalPsd
    reload(Simulative.LognormalPsd)
    from Simulative.LognormalPsd import lognormalpore_model_interactive_impl

    x = lrf_src.xr_x
    y = lrf_src.xr_y
    lognormalpore_model_interactive_impl(x, y, *guess_info, use_ty_as_data=False, plot_mnp=True,
                                         window_title="Demo", parent=parent)