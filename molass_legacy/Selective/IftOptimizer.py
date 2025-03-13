"""
    Selective/IftOptimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data

def try_ift_optimization(button_frame, devel=True):
    if devel:
        from importlib import reload
        import IFT.IftOptimizeImpl
        reload(IFT.IftOptimizeImpl)
    from IFT.IftOptimizeImpl import ift_optimize   
    print("try_ift_optimization")
    editor = button_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    D, E, qv, ecurve = editor.sd.get_xr_data_separate_ly()
    num_components = len(params_array)
    M = get_denoised_data(D, rank=num_components)

    opt_params_array = ift_optimize(M, E, qv, fx, y, model, params_array, debug=True)

