"""
    Simulative/PsdSimulation.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from importlib import reload
import Batch.DataBridgeUtils
reload(Batch.DataBridgeUtils)
from molass_legacy.Batch.DataBridgeUtils import get_lrf_source_impl

def pds_simulation_impl(editor, show_fixed_demo=False):
    print("psd_simulation_impl")
    if show_fixed_demo:
        from importlib import reload
        import Simulative.SimulativeDemo
        reload(Simulative.SimulativeDemo)
        from Simulative.SimulativeDemo import demo
        # showing temporarily the fixed demo
        demo(parent=editor.parent)
    else:
        parent = editor
        parent.config(cursor="wait")
        parent.update()
        lrf_src = get_lrf_source_impl(editor)
        lrf_src.run_simulation_process(editor.parent)
        parent.config(cursor="")