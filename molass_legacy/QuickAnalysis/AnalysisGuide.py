"""

    AnalysisGuide.py

    Copyright (c) 2019-2023, SAXS Team, KEK-PF

"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Mapping.SingleComponent import ACCEPTABLE_MIN_SCI, PEAK_EVAL_RANGE_RATIO

class AnalysisGuide:
    def __init__(self, mapper, judge_holder):
        self.mapper = mapper
        self.judge_holder = judge_holder

    def already_decomposed(self):
        use_elution_models = get_setting('use_elution_models')
        decomp_editor_info = get_setting('decomp_editor_info')
        return use_elution_models and decomp_editor_info is not None

    def has_bad_sci(self):
        min_sci = self.mapper.get_min_sci()
        print('min_sci=', min_sci)
        return min_sci < ACCEPTABLE_MIN_SCI, 'a peak with SCI %d less than %d' % (min_sci, ACCEPTABLE_MIN_SCI), None

    def has_overlapping_peaks(self, debug=False):
        """
        TODO:
            better sigma points by considering asymmetry
        """
        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            x = self.mapper.x_curve.x
            y = self.mapper.x_curve.y
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(x, y)
            ax.set_ylim(ax.get_ylim())

        bool_ret = False

        peaks = self.mapper.x_curve.get_emg_peaks()
        prev_point = None
        max_length = 0
        for k, peak in enumerate(peaks):
            try:
                points = peak.get_model_x_from_ratio(PEAK_EVAL_RANGE_RATIO)
            except:
                import logging
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                logger = logging.getLogger(__name__)
                etb = ExceptionTracebacker()
                logger.warning("failed to get foot points:" + str(etb))
                continue
            if debug:
                ax.plot(x, peak.get_model_y(x))
                ymin, ymax = ax.get_ylim()
                for p in points:
                    ax.plot([p, p], [ymin, ymax], ':', color='gray')
            if prev_point is not None:
                length = prev_point - points[0]
                if length > max_length:
                    max_length = length
                    bool_ret = True
            prev_point = points[1]

        if debug:
            plt.show()

        return bool_ret, 'an overlapping interval of length %.3g' % max_length, None

    def has_uncomputable_rdrs(self):
        assert self.judge_holder is not None
        judge = self.judge_holder.has_uncomputable_rdrs()
        if judge:
            ret_message = "an uncomputable RDR"
            extra_info = "uncomputable"
        else:
            ret_message = "no uncomputable RDRs"
            extra_info = None
        return judge, ret_message, extra_info

    def need_decomp(self):
        reasons = []
        extra_infos = []
        ret_judge = False
        for judge, reason, extra_info in [self.has_bad_sci(), self.has_overlapping_peaks(), self.has_uncomputable_rdrs()]:
            ret_judge = ret_judge or judge
            if judge:
                reasons.append(reason)
            if extra_info is not None:
                extra_infos.append(extra_info)
        return ret_judge, ' and '.join(reasons), extra_infos

    def get_decomp_guide_message(self, reason):
        return  ( "Use of Decomposition Editor is reccommended\n"
                + " due to %s." % reason
                )

class AnalysisGuideInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def decomp_proc_needed(self):
        # for the Tester to be able to call this
        return self.need_decomp and not self.decomposed

    def has_an_uncomputable_rdr(self):
        return len(self.extra_infos) > 0 and self.extra_infos[0] == "uncomputable"

def get_analysis_guide_info(mapper, judge_holder, discomfort):
    std_diff = mapper.std_diff

    if std_diff < 0.3:
        # TODO: refactor to unify 0.2
        if std_diff < 0.2:
            mapping_quality = "The mapping seems fairly good"
        else:
            mapping_quality = "The mapping seems acceptable"
    else:
        mapping_quality = "Some adjustment is recommended because the mapping is not so good."

    if discomfort < 0.1:
        three_d_message = ""
        three_d_guide = False
    else:
        three_d_message = "View the Xray scattering in 3D and consider baseline correction."
        three_d_guide = True

    guide = AnalysisGuide(mapper, judge_holder)

    decomposed = guide.already_decomposed()
    need_decomp, reason, extra_infos = guide.need_decomp()

    if need_decomp:

        message = guide.get_decomp_guide_message(reason)
        fg = 'orange'
        edit_mode = True

    else:

        if std_diff < 0.3:
            if discomfort < 0.1:
                message = mapping_quality + ". We can proceed by pressing [▶ Execute]."
                fg = 'green'
                edit_mode = False
            else:
                message = ( "Although " + mapping_quality.lower() + ", Xray scattering base discomfort is observed.\n"
                            + three_d_message
                            )
                fg = 'orange'
                edit_mode = True
        else:
            if discomfort < 0.1:
                message = mapping_quality
            else:
                message = ( mapping_quality + "\n"
                            + "Also, " + three_d_message.lower()
                            )
            fg = 'orange'
            edit_mode = True

    return AnalysisGuideInfo(
            message=message,
            fg=fg,
            edit_mode=edit_mode,
            three_d_guide=three_d_guide,
            decomposed=decomposed,
            need_decomp=need_decomp,
            extra_infos=extra_infos)

def get_data_correction_state_text():
    from molass_legacy.Mapping.MappingParams import get_mapper_opt_params

    opt_params = get_mapper_opt_params()

    uv_text = "UV data correction: " + opt_params.get_uv_correction_str()
    xr_text = "Xray data correction: " + opt_params.get_xray_correction_str()

    return uv_text + "; " + xr_text
