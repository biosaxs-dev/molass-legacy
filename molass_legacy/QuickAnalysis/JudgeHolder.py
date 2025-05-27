"""

    QuickAnalysis.JudgeHolder.py

    the purpose of this class is to separate CD-related codes from the mapper and sd

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
import logging
from bisect import bisect_right
from molass_legacy._MOLASS.SerialSettings import set_setting
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Conc.ConcDepend import CD_COLORS, CD_BOUNDARIES
from molass_legacy.QuickAnalysis.PreDecomposer import PreDecomposer

def rdr_is_uncomputable(rdr_hints, k):
    return rdr_hints is not None and not rdr_hints[k][0]

def convert_to_colors(cd_list, rdr_hints):
    colors = []
    for k, scd in enumerate(cd_list):
        if scd is None:
            color = 'black'
        else:
            color = CD_COLORS[bisect_right(CD_BOUNDARIES, scd)]

        if rdr_is_uncomputable(rdr_hints, k):
            if color == "green":
                color = 'white'
            else:
                color = 'orange'
        colors.append(color)
    return colors

def convert_to_degrees(cd_list, rdr_hints):
    degrees = []
    for k, scd in enumerate(cd_list):
        if scd is None:
            degree = 1
        else:
            cd_level = bisect_right(CD_BOUNDARIES, scd)
            degree = 2 if cd_level> 0 else 1
        degrees.append(degree)
    return degrees

class JudgeHolder:
    def __init__(self, sd, pre_recog, analysis_copy):
        self.logger = logging.getLogger(__name__)
        self.update_count = 0
        self.sd_orig = sd
        self.pre_recog = pre_recog
        self.sd = analysis_copy
        self.predec = PreDecomposer(sd, pre_recog, analysis_copy, debug=False)
        self.rank_info = None
        self.rdr_hints = None
        self.cd = None
        self.cd_info = None
        self.cd_list = None
        self.cd_colors = None
        self.cd_degrees = None
        self.guide_info = None
        self.x_curve = None

    def set_mapper(self, mapper):
        self.mapper = mapper        # used for RDR calculation

    def update_sd(self, new_sd):
        # called by mapper
        self.sd = new_sd
        self.predec.update_sd_copy(new_sd)
        self.update_count += 1

    def clear_cd_info_dependents(self):
        # called by mapper
        self.cd_list = None
        self.cd_colors = None
        self.cd_degrees = None

    def update_x_curve(self, curve):
        # called by mapper
        self.x_curve = curve

    def get_callbacks(self):
        return self.update_sd, self.clear_cd_info_dependents, self.update_x_curve

    def get_cd_list(self):
        if self.cd_list is None:
            if self.update_count == 0:
                self.logger.warning("using not-yet-corrected sd to get cd_list is not desirable.")
            peak_info = self.x_curve.peak_info
            cd_list = []
            cd_infos = self.get_conc_depend_info()
            n = len(cd_infos)
            kstart = 0
            for p, pkrec in enumerate(peak_info):
                top_x_xr = pkrec[1]
                snd = None
                for k in range(kstart, n):
                    cdrec = cd_infos[k]
                    top_x_cd = cdrec[0]
                    if abs(top_x_cd - top_x_xr) < 5:
                        snd = cdrec[1]
                        kstart = k + 1
                        break
                    else:
                        continue
                cd_list.append(snd)
            self.logger.info("got cd_list as " + str(cd_list))
            self.cd_list = cd_list
        return self.cd_list

    def add_conc_depend_info(self):
        from molass_legacy.Conc.ConcDepend import ConcDepend
        rdr_hints = self.get_rdr_hints()

        q, data, error = self.sd.get_xray_data()
        ecurve = self.mapper.x_curve
        cd = ConcDepend(q, data, error, ecurve, xray_scale=self.sd.get_xray_scale())
        self.cd_info = cd.compute_judge_info(rdr_hints, debug_info=[self])
        self.logger.info("added cd_info as %s", str(self.cd_info))

    def get_conc_depend_info(self):
        if self.cd_info is None:
            if self.sd.baseline_corrected:
                self.logger.info("safely adding cd_info: %s", self.sd.get_id_info())
            else:
                self.logger.warning("adding cd_info before baseline correction is not desirable: %s", self.sd.get_id_info())
            self.add_conc_depend_info()
        return self.cd_info

    def has_uncomputable_rdrs(self):
        if self.rdr_hints is None:
            return False

        ret_judge = False
        for computable, rdr in self.rdr_hints:
            if not computable:
                ret_judge = True
                break
        return ret_judge

    def get_rdr_hints(self):
        if self.rdr_hints is None:
            self.compute_rdr_hints()
        return self.rdr_hints

    def compute_rdr_hints(self):
        from .RgDiffRatios import RgDiffRatios
        sd = self.sd
        self.rdrs = RgDiffRatios(sd, self.mapper.x_curve)
        self.rdr_hints = self.rdrs.get_rank_hints()
        self.logger.info("rdr_hints=%s", str(self.rdr_hints))

    def get_cd_colors(self, guide_info=None, debug=False):
        """
        see ElutionMapperCanvas.get_cd_colors to understand the following control flow.
            1. self.get_cd_list() # substantial computation which implies RDR computation
            2. caller.update_guide_info()   # which adds RDR-info to guide_info
            3. convert_to_colors(cd_list, guide_info)
        """
        if guide_info is None:
            # do this ahead of getting the guild_info
            self.get_cd_list()
            return  # to let the caller.update_guide_info()
        else:
            # do the following after the caller.update_guide_info()
            self.guide_info = guide_info

        if self.cd_colors is None:
            self.logger.info("guide_info.has_an_uncomputable_rdr()? %s", str(guide_info.has_an_uncomputable_rdr()))
            cd_list = self.get_cd_list()    # this is skipping the comutation
            rdr_hints = self.get_rdr_hints()
            self.cd_colors =  convert_to_colors(cd_list, rdr_hints)
            set_setting('mapper_cd_color_info', (self.x_curve.peak_top_x.copy(), self.cd_colors.copy()))
            self.logger.info("mapper_cd_color_info=(%s, %s, %s)", str(self.x_curve.peak_top_x), str(self.cd_colors), str(rdr_hints))
            if debug:
                x_curve = self.x_curve
                x = x_curve.x
                y = x_curve.y
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("get_cd_colors debug")
                    ax.plot(x, y)
                    for px in x_curve.peak_top_x:
                        ax.plot(px, x_curve.spline(px), "o")
                    fig.tight_layout()
                    plt.show()

        return self.cd_colors

    def get_cd_degrees(self):
        if self.cd_degrees is None:
            rdr_hints = self.get_rdr_hints()
            self.cd_degrees =  convert_to_degrees(self.get_cd_list(), rdr_hints)
        return self.cd_degrees

    def get_cd_degree_from_range(self, start, stop):
        cd_degrees = self.get_cd_degrees()
        pno = self.x_curve.peak_no_from_range(start, stop)
        if pno is None:
            degree = 1
            self.logger.warning("conc. dependency is set to %d for unidentified range %d:%d", degree, start, stop)
        else:
            degree = self.cd_degrees[pno]
        return degree
