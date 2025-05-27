"""
    Simulative.DataBridgeUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def get_databridge(decomp_editor, devel=True):
    from importlib import reload
    import Batch.DataBridge
    reload(Batch.DataBridge)
    from molass_legacy.Batch.DataBridge import DataBridge
    sd = decomp_editor.judge_holder.sd_orig
    pre_recog = decomp_editor.judge_holder.pre_recog
    params_array = decomp_editor.get_current_params_array()
    num_components = len(params_array)
    print("num_components =", num_components)
    bridge = DataBridge(sd, pre_recog)
    bridge.prepare_for_lrf_source(num_components=num_components)
    return bridge

def get_lrf_source_impl(decomp_editor):
    bridge = get_databridge(decomp_editor)
    return bridge.get_lrf_source()