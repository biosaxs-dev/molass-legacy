"""
    Estimators.TestTools.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def EGH_estimator_test_impl(editor):
    from importlib import reload
    import Estimators.EghEstimator
    reload(Estimators.EghEstimator)
    from Estimators.EghEstimator import EghEstimator
    print("estimator_test_impl")
    estimator = EghEstimator(editor)
    estimator.estimate_params(debug=True)

def SDM_estimator_test_impl(editor):
    from importlib import reload
    import Estimators.SdmEstimator
    reload(Estimators.SdmEstimator)
    from Estimators.SdmEstimator import SdmEstimator
    print("estimator_test_impl")
    estimator = SdmEstimator(editor)
    optimizer = editor.fullopt
    opt_name = optimizer.get_name()
    print("optimizer name=", opt_name)
    edm_available = opt_name[0:2] == "G2"   # 

    init_params = estimator.estimate_params(edm_available=edm_available, debug=True)
    if init_params is None:
        return

    class_code = 'G1100'
    if edm_available:
        # create SDM optimizer
        from molass_legacy.Optimizer.FuncImporter import import_objective_function
        optimizer_class = import_objective_function(class_code)
        optimizer = optimizer_class(
            editor.dsets,
            editor.get_n_components(),
            uv_base_curve=editor.base_curve_info[0],
            xr_base_curve=editor.baseline_objects[1],
            qvector=editor.sd.qvector,    # trimmed sd
            wvector=editor.sd.lvector,
            )

        optimizer.params_type.set_estimator(estimator)      # reconsider the neccesity of this line
        optimizer.prepare_for_optimization(init_params)

    lrf_info = optimizer.objective_func(init_params, return_lrf_info=True)

    def plot_components(ax, x, y, C, ty):
        ax.plot(x, y, label="data")
        for k, cy in enumerate(C):
            ax.plot(x, cy, ":", label="component-%d" % k)
        ax.plot(x, ty, ":", color="red", label="model total")

    with plt.Dp():
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
        fig.suptitle("estimator_test_impl")
        ax1, ax2 = axes[0,0], axes[0,1]
        ax3, ax4 = axes[1,0], axes[1,1]
        plot_components(ax1, lrf_info.uv_x, lrf_info.uv_y, lrf_info.matrices[3], lrf_info.uv_ty)
        plot_components(ax2, lrf_info.x, lrf_info.y, lrf_info.matrices[1], lrf_info.xr_ty)
        plot_components(ax3, lrf_info.uv_x, lrf_info.uv_y, lrf_info.scaled_uv_cy_array, lrf_info.uv_ty)
        plot_components(ax4, lrf_info.x, lrf_info.y, lrf_info.scaled_xr_cy_array, lrf_info.xr_ty)
        fig.tight_layout()
        ret = plt.show()

    if ret and edm_available:
        # replace editor optimizer
        # task: move this code to PeakEditor
        from molass_legacy._MOLASS.SerialSettings import set_setting
        from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
        set_setting('elution_model', 1)
        editor.optimizer = optimizer
        editor.fullopt = optimizer
        text = "Elution Decomposition of %s with func=%s" % (get_in_folder(), class_code)
        editor.fig.suptitle(text, fontsize=20)
        editor.draw_scores(init_params=init_params, create_new_optimizer=False)
        editor.fullopt_class = optimizer_class
        editor.class_code = class_code

    return ret

def EDM_estimator_test_impl(editor):
    from importlib import reload
    import Estimators.EdmEstimator
    reload(Estimators.EdmEstimator)
    from Estimators.EdmEstimator import EdmEstimator
    n_components = editor.get_n_components()
    print("estimator_test_impl: n_components=", n_components)
    estimator = EdmEstimator(editor, n_components=n_components)
    estimator.estimate_params(debug=True)

def estimator_test_impl(editor):

    extra_button_specs = [
        ("EGH Edtimator", lambda: EGH_estimator_test_impl(editor)),
        ("SDM Edtimator", lambda: SDM_estimator_test_impl(editor)),
        ("EDM Edtimator", lambda: EDM_estimator_test_impl(editor)),
    ]

    with plt.Dp(extra_button_specs=extra_button_specs):
        fig, ax = plt.subplots()
        plt.show()