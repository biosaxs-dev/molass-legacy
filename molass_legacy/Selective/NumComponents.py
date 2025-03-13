"""
    Selective.NumComponents.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import Struct
from Selective.V1ParamsAdapter import make_decomp_result_impl

def change_num_components_impl(advanced_frame, target_props=None, debug=False):
    if debug:    
        import CFSD.SimpleDecompose
        reload(CFSD.SimpleDecompose)
    from CFSD.SimpleDecompose import decompose_elution_simply
    if debug:
        print("change_num_components_impl")
    editor = advanced_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    if debug:
        print("edm_text_from_editor_frame: ", model.get_name(), model.__class__)
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    if debug:
        with plt.Dp(window_title='Change Num Components', button_spec=["Close"]):
            fig, ax = plt.subplots()
            ax.set_title("before change", fontsize=16)
            ax.plot(x, y)
            for params in params_array:
                cy = model(fx, params)
                ax.plot(x, cy, ":")
            fig.tight_layout()
            plt.show()

    num_peaks = advanced_frame.num_components.get()

    if target_props is None:
        import Selective.BridgeUtils
        reload(Selective.BridgeUtils)
        from Selective.BridgeUtils import decompose_by_bridge
        traditional_info = Struct(num_peaks=num_peaks, mapper=editor_frame.mapper)
        decomp_result = decompose_by_bridge(fx, y, uv_y, model, traditional_info, debug=debug)
        if decomp_result is None:
            return
    else:
        if num_peaks < 3:
            traditional_info = Struct(num_peaks=num_peaks, mapper=editor_frame.mapper)
            decomp_result = decompose_elution_simply(fx, y, uv_y, model, traditional_info, props=target_props, debug=False)
        else:
            import Elution.Proportional
            reload(Elution.Proportional)
            from molass_legacy.Elution.Proportional import decompose_by_proportions
            params_result = decompose_by_proportions(fx, y, uv_y, model, target_props, xr_only=True, debug=debug)
            if params_result is None:
                return
            decomp_result = make_decomp_result_impl(editor, params_result, debug=False)

    with plt.Dp(window_title='Change Num Components', button_spec=["Accept", "Cancel"]):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
        fig.suptitle("after change", fontsize=20)
        ax1.set_title("UV", fontsize=16)
        ax1.plot(x, uv_y)
        cy_list = []
        for rec in decomp_result.opt_recs_uv:
            cy = rec.evaluator(fx)
            cy_list.append(cy)
            ax1.plot(x, cy, ":")
        ty = np.sum(cy_list, axis=0)
        ax1.plot(x, ty, ":", color="red")
        ax2.set_title("XR", fontsize=16)
        ax2.plot(x, y)
        cy_list = []
        for rec in decomp_result.opt_recs:
            cy = rec.evaluator(fx)
            cy_list.append(cy)
            ax2.plot(x, cy, ":")
        ty = np.sum(cy_list, axis=0)
        ax2.plot(x, ty, ":", color="red")
        fig.tight_layout()
        ret = plt.show()

    print("ret=", ret)
    if ret:
        advanced_frame.update_button_status(change_id="CNC")
        editor.update_current_frame_with_result(decomp_result, debug=True)