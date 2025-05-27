"""
    BaselineUtils.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Mapping.MapperConstructor import create_mapper

def get_default_baseline_type():
    xray_baseline_type = get_setting("xray_baseline_type")
    ret_type = 1
    if xray_baseline_type == 5:
        ret_type = 2
    return ret_type

def get_corrected_sd_impl(sd, sd_orig, pre_recog, mapped_info=None):
    # xrav correction
    if mapped_info is None:
        mapper = create_mapper(None, sd, sd_orig, pre_recog)
        mapped_info = mapper.get_mapped_info()
    sd.apply_baseline_correction(mapped_info, basic_lpm=True)   # see Mapping.ElutionMapper.compute_xray_baseline()

    # uv correction
    # not required for v2 optimization

    return sd

def create_xr_baseline_object(unified_baseline_type=None):
    if unified_baseline_type is None:
        unified_baseline_type = get_setting("unified_baseline_type")

    if unified_baseline_type == 1:
        from .LinearBaseline import LinearBaseline
        return LinearBaseline()
    elif unified_baseline_type == 2:
        from .IntegralBaseline import IntegralBaseline
        return IntegralBaseline()
    elif unified_baseline_type == 3:
        from .FoulingBaseline import FoulingBaseline
        return FoulingBaseline()
    else:
        # currently not supported
        assert False

def demo(in_folder, logger):
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.UV.PlainCurve import make_secondary_e_curve_at
    from MatrixData import simple_plot_3d
    from molass_legacy.Elution.CurveUtils import simple_plot
    from DataUtils import get_in_folder

    set_setting("in_folder", in_folder)
    sp = StandardProcedure()
    sd = sp.load_old_way(in_folder)
    pre_recog = PreliminaryRecognition(sd)
    sd_copy = sd.get_copy()
    corrected = get_corrected_sd_impl(sd_copy, sd, pre_recog)

    D0, E0, qv0, xr_curve0 = sd.get_xr_data_separate_ly()
    U0,  _, wv0, uv_curve0 = sd.get_uv_data_separate_ly()
    uv_curve0_2 = pre_recog.flowchange.a_curve2

    D1, E1, qv1, xr_curve1 = corrected.get_xr_data_separate_ly()
    U1,  _, wv1, uv_curve1 = corrected.get_uv_data_separate_ly()
    uv_curve1_2 = make_secondary_e_curve_at(corrected.absorbance.data, wv1, uv_curve1, logger)

    with plt.Dp():
        in_folder_ = get_in_folder()
        fig = plt.figure(figsize=(20,10))
        fig.suptitle("Correction of %s" % in_folder_, fontsize=20)
        ax1 = fig.add_subplot(241, projection="3d")
        ax2 = fig.add_subplot(242, projection="3d")
        ax3 = fig.add_subplot(243)
        ax4 = fig.add_subplot(244)
        ax5 = fig.add_subplot(245, projection="3d")
        ax6 = fig.add_subplot(246, projection="3d")
        ax7 = fig.add_subplot(247)
        ax8 = fig.add_subplot(248)
        ax1.set_title("Input", fontsize=16)
        ax2.set_title("Corrected", fontsize=16)
        ax3.set_title("Input", fontsize=16)
        ax4.set_title("Corrected", fontsize=16)

        simple_plot_3d(ax1, D0, x=qv0)
        simple_plot_3d(ax2, D1, x=qv1)
        simple_plot(ax3, xr_curve0, color="orange")
        simple_plot(ax4, xr_curve1, color="orange")
        simple_plot_3d(ax5, U0, x=wv0)
        simple_plot_3d(ax6, U1, x=wv1)
        simple_plot(ax7, uv_curve0, color="blue")
        simple_plot(ax8, uv_curve1, color="blue")
        ax7t = ax7.twinx()
        ax8t = ax8.twinx()
        ax7t.grid(False)
        ax8t.grid(False)
        def plot_curve_sub(ax, curve):
            x = curve.x
            y = curve.y
            ax.plot(x, y, alpha=0.5)
        plot_curve_sub(ax7t, uv_curve0_2)
        plot_curve_sub(ax8t, uv_curve1_2)

        fig.tight_layout()
        plt.show()
