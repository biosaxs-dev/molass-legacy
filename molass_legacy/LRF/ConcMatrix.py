"""
    ConcMatrix.py

    Copyright (c) 2019-2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting

class ConcMatrix:
    def __init__(self, x, conc, conc_depend=2, paired_ranges=None, mc_vector=None,
                    conc_factor=5,  add_conc_const=False, ecurve=None, j0=0):
        self.logger = logging.getLogger(__name__)
        self.x = x
        self.conc = conc
        self.conc_depend = conc_depend
        self.cdl_list = []
        self.paired_ranges = paired_ranges
        self.mc_vector = mc_vector
        self.conc_factor = conc_factor
        self.ecurve = ecurve
        self.j0 = j0
        self.concentration_datatype = get_setting('concentration_datatype')
        self.logger.info("creating C matrix in ranges %s with concentration_datatype=%d", str(paired_ranges), self.concentration_datatype)

        if conc is None:
            if self.concentration_datatype == 1:
                C_list = self.make_from_xr_ecurve()
            elif self.concentration_datatype == 3:
                assert mc_vector is not None
                C_list = self.make_from_mc_vector()
            else:
                """
                    note that difference info between concentration_datatype 0 and 2
                    has been already included in paired_ranges
                    when they were made in DecompEditorFrame.make_range_info()
                    by selecting opt_recs or opt_recs_uv
                """
                C_list = self.make_from_sec_model()
        else:
            C_list = self.make_from_conc()

        if add_conc_const:
            C_list.append( np.ones( len(x) ) )

        self.data = np.array( C_list )
        self.logger.info("C matrix data.shape is %s", str(self.data.shape))

    def make_from_xr_ecurve(self):
        C = self.ecurve.spline(self.x)
        C_list = [C]
        if self.conc_depend > 1:
            for k in range(2, self.conc_depend+1):
                C_list += [C**k]

        self.cdl_list.append(len(C_list))

        return C_list

    def make_from_mc_vector(self):
        C = self.mc_vector      # already scaled with conc_factor
        C_list = [C]
        if self.conc_depend > 1:
            for k in range(2, self.conc_depend+1):
                C_list += [C**k]

        self.cdl_list.append(len(C_list))

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            C_ = np.array(C_list)
            fig = plt.figure(figsize=(14,7))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.set_title("Rows of C matrix", fontsize=20)
            ax1.plot(C_[0,:], label='$C[0,:]$')
            ax1.plot(C_[1,:], label='$C[1,:]=C[0,:]^2=$')

            U, s, VT = np.linalg.svd(C_)
            ax2.set_title("Singular Values of C matrix", fontsize=20)
            ax2.plot(s[0:5], ':', marker='o')
            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(0, ymax)

            ax1.legend()
            fig.tight_layout()
            plt.show()

        return C_list

    def make_from_sec_model(self, debug=False):
        C_list = []
        x = self.x

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.KekLib.OurMatplotlib import get_color
            from molass_legacy.DataStructure.AnalysisRangeInfo import shift_paired_ranges
            print('conc_factor=', self.conc_factor)
            print('paired_ranges=', self.paired_ranges, 'conc_depend=', self.conc_depend)
            j0 = self.j0
            plt.push()
            fig = plt.figure()
            ax = fig.gca()
            axt = ax.twinx()
            ax.set_title('debug for ' + str(shift_paired_ranges(-j0, self.paired_ranges)))
            ecurve = self.ecurve
            ax.plot(ecurve.x+j0, ecurve.y, color='orange')

        for k, paired_range in enumerate(self.paired_ranges):
            cy = np.zeros(len(x))

            for elm_rec in paired_range.elm_recs:
                e   = elm_rec[0]
                fnc = elm_rec[1]
                y = fnc(x)
                cy += y
                if debug:
                    color = get_color(e)
                    x_ = self.ecurve.x
                    y_ = fnc(x_)
                    ax.plot(x_+j0, y_, ':', color=color)
                    ax.plot(x +j0, y, color=color, label=str(e))

            C = cy * self.conc_factor

            if debug:
                axt.plot(x+j0, C, color='red', label=str([k]) + ' conc. scaled')

            C_rows = [C]
            if self.conc_depend > 1:
                for k in range(2, self.conc_depend+1):
                    C_rows += [C**k]

            C_list += C_rows
            self.cdl_list.append(len(C_rows))

        if debug:
            from PlotUtils import align_yaxis_np
            ymin, ymax = ax.get_ylim()
            ymint, ymaxt = axt.get_ylim()
            axt.set_ylim(ymin, ymaxt)
            ax.legend(loc='upper left')
            axt.legend(loc='upper right')
            align_yaxis_np(ax, axt)
            fig.tight_layout()
            plt.show()
            plt.pop()

        return C_list

    def make_from_conc(self):
        if self.conc_depend == 2:
            ret_conc = self.conc
        else:
            if len(self.conc) == 4:
                ret_conc = [self.conc[0], self.conc[2]]
            else:
                # work-around for 3-state model
                # TODO: do this properly
                ret_conc = self.conc
        return ret_conc
