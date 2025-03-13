"""
    GuinierTools.CompareGdevMeasures.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def compare_gdev_measures_impl(gdev, Cxr, rg_params, valid_components):
    from importlib import reload
    import GuinierTools.RgCurveUtils
    reload(GuinierTools.RgCurveUtils)
    from .RgCurveUtils import get_connected_curve_info, convert_to_milder_qualities, get_reconstructed_curve
    from Distance.FrobeniusXdiffmax import frobenius_xdiffmax
    print("compare_gdev_measures_impl")
    rg_curve = gdev.rg_curve
    x = rg_curve.x
    y = rg_curve.y
    x_, y_, rgv, qualities, valid_bools = get_connected_curve_info(rg_curve)
    mqualities = convert_to_milder_qualities(qualities)
    size = len(x[valid_bools])
    rrgv = get_reconstructed_curve(size, valid_bools, Cxr, rg_params)

    print("len(x_)=", len(x_))
    print("len(qialities[0])=", len(rg_curve.qualities[0]))
    diff = frobenius_xdiffmax(rgv, rrgv, debug=True)
    print("diff=", diff)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("compare_gdev_measures_impl")
        axt = ax.twinx()
        axt.grid(False)
        ax.plot(x, y, label='Data')
        axt.plot(x_, rgv, label='measured Rg curve', color='gray')
        axt.plot(x_, rrgv, ":", label='reconstructed Rg curve', color='gray')
        axt.plot(x_, qualities*100, label='Quality', color='green', alpha=0.3)
        axt.plot(x_, mqualities*100, label='Milder Quality', color='cyan', alpha=0.3)
        fig.tight_layout()
        plt.show()