"""
Baseline.UvBaseline.py

This module has been extracted from the Batch.FullBatch.py
to make its on-the-fly debugging easier.
"""

def get_uv_base_params_impl(editor, xyt=None, debug=False):
    """

    ------------------------------------------------------------------
    editor.base_curve_info  DataTreatment   OptimalTrimming UvPreRecog

    editor.baseline_objects ModeledPeaks    UvPreRecog
    editor.baselines        ModeledPeaks    UvPreRecog
    editor.baseline_params  ModeledPeaks    UvPreRecog
    """
    # uv_base_params has been estimated using UV.UvPreRecog.
    # see QuickAnalysis.ModeledPeaks.get_curve_xy_impl
    uv_base_params = editor.baseline_params[0]
    uv_base_params_ = uv_base_params.copy()

    if debug:
        print("len(uv_base_params_)=", len(uv_base_params_))

    if len(uv_base_params_) == 8 or debug:
        if xyt is None:
            uv_x, uv_y = editor.ecurve_info[0:2]
            uv_ty = editor.uv_ty
        else:
            uv_x, uv_y, uv_ty = xyt

        if len(uv_base_params_) == 8:
            integ_scale, _ = editor.uv_base_curve.guess_integral_scale(uv_x, uv_y, uv_base_params_[:-1], ty=uv_ty)
            uv_base_params_[-1] = integ_scale

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        uv_basecurve1, uv_initparams1 = editor.base_curve_info
        baselines = editor.baselines
        uv_basecurve2 = editor.baseline_objects[0]
        uv_initparams2 = uv_base_params_
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("get_uv_base_params_impl")

            ax1.set_title("from DataTreatment")
            ax1.plot(uv_x, uv_y, label="data")
            ax1.plot(uv_x, uv_basecurve1(uv_x, uv_initparams1, uv_ty), label="baseline")
            ax1.legend()

            ax2.set_title("from ModeledPeaks")
            ax2.plot(uv_x, uv_y, label="data")
            ax2.plot(uv_x, baselines[0], label="baseline 1")
            ax2.plot(uv_x, uv_basecurve2(uv_x, uv_initparams2, uv_ty), label="baseline 2")
            ax2.legend()

            fig.tight_layout()
            plt.show()

    return uv_base_params_