"""
    Optimizer.GuinierDeviationTester.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def test_update_guinier_region_impl(js_canvas):
    print("test_update_guinier_region_impl")
    optimizer = js_canvas.fullopt
    params = js_canvas.get_current_params()
    optimizer.update_guinier_region(params=params, debug=True)

def show_guinier_deviation(js_canvas):
    from ..Optimizer.FvScoreConverter import convert_score
    print("show_guinier_deviation")
    optimizer = js_canvas.fullopt
    params = js_canvas.get_current_params()
    lrf_info = optimizer.objective_func(params, return_lrf_info=True)
    Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices
    rg_params = optimizer.separate_params[2]
    deviation = optimizer.get_guinier_deviation(Pxr, rg_params, debug=True)
    print("deviation=", deviation, convert_score(deviation))
