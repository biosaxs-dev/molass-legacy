"""
    ElutionMatrix.py - successor to the LRF.ConcMatrix

    refactoring phase 1     using ConcMatrix
    refactoring phase 2     rewrite

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from .ConcMatrix import ConcMatrix

USE_DICTINCT_CD = True

class ElutionMatrix:
    def __init__(self, depr_solver):
        self.logger = logging.getLogger(__name__)
        self.conc = depr_solver.conc
        self.conc_factor = depr_solver.conc_factor
        self.ecurve = depr_solver.ecurve
        self.j0 = depr_solver.j0
        self.data = depr_solver.data
        self.cd_slice = depr_solver.cd_slice
        self.xray_scale = depr_solver.xray_scale
        self.mapper = depr_solver.mapper
        self.conc_depend = depr_solver.conc_depend
        self.paired_ranges = depr_solver.cnv_ranges
        self.rank_control = depr_solver.rank_control
        self.mc_vector = depr_solver.mc_vector

    def make_elution_matrix(self, start, stop, peakset_info, lrf_rank=None, debug=False):
        pno, nth, peakset, known_peak_info = peakset_info
        paired_ranges_ = [self.paired_ranges[i] for i in peakset]

        x = self.ecurve.x[start:stop]
        if self.mc_vector is None:
            mc_vector = None
        else:
            mc_vector = self.mc_vector[start:stop]

        if self.rank_control:
            if lrf_rank is None:
                if USE_DICTINCT_CD:
                    cmatrix = ConcMatrix(x, self.conc, conc_depend=1,
                                            paired_ranges=paired_ranges_, mc_vector=mc_vector,
                                            conc_factor=self.conc_factor,
                                            ecurve=self.ecurve, j0=self.j0,
                                            )
                    M = self.data[:,start:stop]
                    conc_depend = compute_distinct_cd(M, cmatrix.data, self.cd_slice, self.xray_scale, self.logger)
                else:
                    conc_depend = self.mapper.get_cd_degree_from_range(start, stop)
            else:
                if lrf_rank == 1:
                    conc_depend = 1
                else:
                    conc_depend = 2
        else:
            conc_depend = self.conc_depend

        cmatrix = ConcMatrix(x, self.conc, conc_depend=conc_depend,
                                paired_ranges=paired_ranges_, mc_vector=mc_vector,
                                conc_factor=self.conc_factor,
                                ecurve=self.ecurve, j0=self.j0,
                                )

        if debug:
            x_ = self.ecurve.x
            y_ = self.ecurve.y
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("make_elution_matrix: [%d:%d]" % (start, stop))
                if self.mc_vector is not None:
                    ax1.plot(x_, self.mc_vector)
                for k, cy in enumerate(cmatrix.data):
                    if k%2 == 0:
                        ax1.plot(x, cy, ":", lw=3)
                ax2.plot(x_, y_)
                fig.tight_layout()
                plt.show()

        return cmatrix.data
