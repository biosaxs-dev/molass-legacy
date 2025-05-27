"""
    PreliminaryRecognition.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import copy
import numpy as np
import logging
from molass_legacy.UV.Absorbance import Absorbance
from . import FlowChange, TrimmingInfo, AutoRestrictor
from molass_legacy.UV.PlainCurve import make_secondary_e_curve_at
from molass_legacy.Mapping.CurveSimilarity import CurveSimilarity
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy.Baseline.BaselineGuinier import change_baseline_correction_defaults
# from molass_legacy.Trimming.PreliminaryRg import get_default_angle_range_impl
from .PreliminaryRg import get_flange_limit, PreliminaryRg, get_small_anlge_limit
from .EndingAnomaly import check_ending_anomaly
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

SECONDARY_WAVE_LENGTH   = 400
ACCEPTABLE_SIMILARITY_SCORE = 0.8   # > 0.75 for 20170304

def get_slices_impl(listname):
    print("get_slices_impl: listname=", listname)
    trimlist = get_setting(listname)
    if trimlist is None:
        ret_slices = [slice(None, None), slice(None, None)]
    else:
        ret_slices = []
        for triminfo in trimlist:
            ret_slices.append(slice(triminfo.start, triminfo.stop))
        ret_slices.reverse()
    print("get_slices_impl: ret_slices=", ret_slices)
    return ret_slices

class PreliminaryRecognition:
    def __init__(self, sd, recog_only=False, debug=False):
        """
        "Preliminary Recognition" consists of the following parts.

        self.set_info() --------------- (1) UV.PlainCurve.make_secondary_e_curve_at()
                                                Trimming.Sigmoid.guess_bent_sigmoid()
                                                UV.PlainCurve.check_diehardness()

                                        (2) self.flowchange = FlowChange()
                                                .PeakRegion
                                                    Mapping.CurveSimilarity
                                                        Mapping.PeakMapper

                                        (3) 

        self.set_restrict_info() ------ (4) self.ar = AutoRestrictor()

        """
        self.logger = logging.getLogger( __name__ )
        try:
            # for backward compatibility
            self.pre_recog_copy = sd.get_copy(pre_recog=self)
            self.set_info(self.pre_recog_copy, debug=debug)
        except:
            self.pre_recog_copy = None
            self.set_info(sd, debug=debug)
        self.Rg = None
        if recog_only:
            # for testing AutoRestrictor. See test_1117_TrimmingResult.py
            from importlib import reload
            import Trimming
            reload(Trimming)
            from molass_legacy.Trimming import AutoRestrictor
            pass
        else:
            if self.pre_recog_copy is None:
                self.set_restrict_info(sd)
            else:
                self.set_restrict_info(self.pre_recog_copy)

    def set_info(self, sd, debug=False):
        self.q = sd.intensity_array[0,:,0]
        x_curve = sd.get_xr_curve()

        # construction of absorbance (if needed) must be delayed until exclusion is complete
        self.absorbance = absorbance = sd.absorbance

        if not hasattr(sd, 'mtd_elution') or sd.mtd_elution is None:
            data = absorbance.data
            wl_vector = absorbance.wl_vector
            a_curve = absorbance.a_curve
            a_curve2 = make_secondary_e_curve_at(data, wl_vector, a_curve, self.logger)
            try:
                self.flowchange = FlowChange(a_curve, a_curve2, x_curve, debug=debug)
            except:
                self.logger.warning("FlowChange failed. Using NullFlowChange instead.")
                import molass.FlowChange.NullFlowChange
                from importlib import reload
                reload(molass.FlowChange.NullFlowChange)
                from molass.FlowChange.NullFlowChange import NullFlowChange
                self.flowchange = NullFlowChange(a_curve, a_curve2, x_curve) 
            self.cs = self.verify_approx_mapping(self.flowchange.get_similarity())
            check_ending_anomaly(x_curve, a_curve, a_curve2)
        else:
            self.flowchange = sd.mtd_elution.create_flow_change_proxy()
            self.cs = self.flowchange.similarity

        self.mapped_info = self.cs.mapped_info
        self.get_default_angle_range(sd)

    def verify_approx_mapping(self, similarity):
        # remake if it is bad
        score = similarity.compute_whole_similarity()
        print('score=', score)
        fc_list = self.flowchange.get_real_flow_changes()
        if score < ACCEPTABLE_SIMILARITY_SCORE:
            self.logger.info("re-constructing CurveSimilarity due to poor score %.2g and flowchages %s" % (score, str(fc_list)) )
            a_vector = copy.deepcopy(similarity.a_curve.y)
            for k, fc in enumerate(fc_list):
                if fc is None:
                    continue
                if k == 0:
                    a_vector[:fc] = 0
                else:
                    a_vector[fc:] = 0
            a_curve = ElutionCurve(a_vector)
            similarity = CurveSimilarity(a_curve, similarity.x_curve, orig_a_curve=similarity.a_curve)
            score = similarity.compute_whole_similarity()
            print('score=', score)
            self.logger.info("got a revised score %.2g" % score)

        self.similarity_score = score
        return similarity

    def get_flow_changes(self):
        return self.flowchange.get_flow_changes()

    def get_real_flow_changes(self):
        return self.flowchange.get_real_flow_changes()

    def get_default_angle_range(self, sd):
        from molass_legacy.DataStructure.LPM import get_corrected_matrix        # moved due to ImportError: cannot import name 'get_corrected_matrix' from partially initialized module 'LPM' (most likely due to a circular import)

        D, E, qv, e_curve = sd.get_xr_data_separate_ly()

        # angle_start, flange_limit, pre_rg = get_default_angle_range_impl(D, E, e_curve, qv, sd.xray_index, self.logger)
        qlimit = sd.xr_index

        flange_limit = get_flange_limit(D, E, e_curve, qv)

        change_baseline_correction_defaults(sd, self.flowchange, self.logger)
        correctedD = get_corrected_matrix(D)

        pre_rg = PreliminaryRg(correctedD, E, e_curve, qv, flange_limit)

        angle_start = get_small_anlge_limit(pre_rg, correctedD, E, e_curve, qv, qlimit, self.logger)

        self.angle_start = angle_start
        self.flange_limit = flange_limit
        self.pre_rg = pre_rg
        self.Rg = pre_rg.Rg
        return angle_start, flange_limit

    def get_angle_range(self):
        return self.angle_start, self.flange_limit

    def get_gunier_interval(self):
        sg = self.pre_rg.sg
        return sg.guinier_start, sg.guinier_stop

    def get_rg(self):
        """
        simpler solution is needed,
        because of the trouble with 20180206_TG
        """
        ret_rg = self.Rg
        if ret_rg is None:
            ret_rg = self.pre_rg.compute_rg()
            self.Rg = ret_rg
        return ret_rg

    def set_restrict_info(self, sd, debug=False, figfile=None):
        xr_restrict_list = get_setting('xr_restrict_list')
        uv_restrict_list = get_setting('uv_restrict_list')

        if xr_restrict_list is not None or uv_restrict_list is not None:
            return

        xray_elution_size = sd.intensity_array.shape[0]
        angle_size = sd.intensity_array.shape[1]
        uv_elution_size = sd.conc_array.shape[1]

        angle_start, angle_end = self.get_default_angle_range(sd)

        if angle_start == 0 and angle_end is None:
            xr_angle_restrict = None
        else:
            xr_angle_restrict = TrimmingInfo(1, angle_start, angle_end, angle_size)

        if sd.mtd_elution is None:
            fc = self.flowchange.get_real_flow_changes()
        else:
            fc = sd.mtd_elution.get_linear_slope_ends()

        if fc[0] is None and fc[1] is None:
            uv_restrict_list = None
            xr_elution_restrict = None
        else:
            if sd.mtd_elution is None:
                narrow_trimming = get_setting('narrow_trimming')
                if narrow_trimming:
                    jj = self.flowchange.get_safe_flow_changes()
                    uv_restrict_list = [TrimmingInfo(1, jj[0], jj[1]+1, uv_elution_size), None]
                else:
                    jj = self.flowchange.get_flow_changes()
            else:
                jj = sd.mtd_elution.get_linear_slope_ends()

            # print('self.similarity_score=', self.similarity_score)
            # if self.similarity_score >= ACCEPTABLE_SIMILARITY_SCORE or True:
            if True:
                # TODO: better control for 20180304
                ii = self.get_restrict_list_from_uv_flow_changes(jj, len(sd.absorbance.a_vector), len(sd.xray_curve.x))
                xr_elution_restrict = TrimmingInfo(1, ii[0], ii[1]+1, xray_elution_size)
            else:
                # 20170304
                xr_elution_restrict = None

        xr_restrict_list = [xr_elution_restrict, xr_angle_restrict]

        use_moment_trimming = get_setting('use_moment_trimming')
        if use_moment_trimming:
            from molass_legacy.Trimming.MomentTrimming import get_moment_trimming_info
            from molass_legacy.Trimming.TrimmingUtils import get_default_uv_wl_trimming
            from molass_legacy.Trimming.AutoRestrictorProxy import AutoRestrictorProxy
            ret_info = get_moment_trimming_info(sd)
            uv_elution_restrict = ret_info[0]
            xr_elution_restrict = ret_info[1]
            self.logger.info("moment_trimming_info has been set as %s in PreliminaryRecognition", str(ret_info))
            if uv_restrict_list is None:
                uv_wl_restrict = get_default_uv_wl_trimming(sd)
                uv_restrict_list = [uv_elution_restrict, uv_wl_restrict]
            else:
                uv_restrict_list[0] = uv_elution_restrict
                if uv_restrict_list[1] is None:
                    uv_wl_restrict = get_default_uv_wl_trimming(sd)
                    uv_restrict_list[1] = uv_wl_restrict

            xr_restrict_list[0] = xr_elution_restrict           
            ar = AutoRestrictorProxy(sd, uv_restrict_list, xr_restrict_list)
        else:
            if sd.mtd_elution is None:
                ar = AutoRestrictor(sd, self.cs)
                uv_restrict_list, xr_restrict_list = ar.get_better_restriction(uv_restrict_list, xr_restrict_list, debug=debug, figfile=figfile)
                if debug:
                    print("********************* uv_restrict_list=", uv_restrict_list)
                    print("********************* xr_restrict_list=", xr_restrict_list)
            else:
                ar = None
        self.ar = ar

        set_setting('uv_restrict_list', uv_restrict_list)
        set_setting('xr_restrict_list', xr_restrict_list)
        self.logger.info("uv_restrict_list has been set to %s in PreliminaryRecognition", str(uv_restrict_list))
        self.logger.info("xr_restrict_list has been set to %s in PreliminaryRecognition", str(xr_restrict_list))
        set_setting('uv_restrict_copy', copy.deepcopy(uv_restrict_list))
        set_setting('xr_restrict_copy', copy.deepcopy(xr_restrict_list))

    def reset_restrict_info(self):
        set_setting('xr_restrict_list', None)
        set_setting('uv_restrict_list', None)
        self.set_restrict_info(self.pre_recog_copy)

    def get_restrict_list_from_uv_flow_changes(self, fc, a_size, x_size):
        ret_list = []
        j_end = a_size - 1
        for j in fc:
            if j == 0:
                i = 0
            elif j == j_end:
                i = x_size - 1
            else:
                i = self.cs.inverse_int_value(j)
            ret_list.append(i)

        if False:
            def mapped_plot_closure(ax):
                y = self.cs.x_curve.y
                for i in ret_list:
                    ax.plot(i, y[i], 'o', color='cyan')
            self.cs.plot_the_mapped_state(mapped_plot_closure)

        return ret_list

    def get_xr_slices(self):
        return get_slices_impl("xr_restrict_list")

    def get_uv_slices(self):
        return get_slices_impl("uv_restrict_list")

    def restrict_info_changed(self):
        uv_restrict_list = get_setting('uv_restrict_list')
        uv_restrict_copy = get_setting('uv_restrict_copy')
        xr_restrict_list = get_setting('xr_restrict_list')
        xr_restrict_copy = get_setting('xr_restrict_copy')
        if (    str(uv_restrict_list) == str(uv_restrict_copy)
            and str(xr_restrict_list) == str(xr_restrict_copy)
            ):
            return False
        else:
            return True

    def get_pre_recog_copy(self):
        print('pre_recog_copy.absorbance=', self.pre_recog_copy.absorbance)
        return  self.pre_recog_copy

    def get_xray_shift(self):
        restrict = get_setting('xr_restrict_list')[0]
        shift = 0 if restrict is None else restrict.start
        if shift is None:
            shift = 0
        return shift

    def get_restricted_mapped_info(self):
        best_params, max_score_pair, max_info = self.mapped_info
        A, B = best_params
        shift = self.get_xray_shift()
        res_params = np.array( [A, B-shift])

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            uv_curve = self.pre_recog_copy.absorbance.a_curve
            xray_curve = self.pre_recog_copy.xray_curve
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
            ax1, ax2 = axes
            ax1.plot(uv_curve.x, uv_curve.y, color='blue')
            ax2.plot(xray_curve.x, xray_curve.y, color='orange')
            if restrict is not None:
                ymin, ymax = ax2.get_ylim()
                ax2.set_ylim(ymin, ymax)
                for p in [restrict.start, restrict.stop]:
                    ax2.plot([p, p], [ymin, ymax], ':', color='gray')
            fig.tight_layout()
            plt.show()

        return [res_params, max_score_pair, max_info]
