# coding: utf-8
"""
    Rdf.py

    Copyright (c) 2020-2022, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.NumpyUtils import np_loadtxt_robust
from molass_legacy.AutorgKekAdapter import AutorgKekAdapter
from molass_legacy.ATSAS.DatGnom import DatgnomExecutor

class Rdf:
    def __init__(self, q, I):
        self.interp = UnivariateSpline(q, I, s=0, ext=1)
        self.q = q
        self.vec_int = np.vectorize(self.expint)

    def integrand(self, t, r):
        return self.interp(t)*t*r*np.sin(t*r)

    def expint(self, r):
        return 4*np.pi*quad(self.integrand, 0, self.q[-1], args=(r))[0]

    def compute(self, r):
        return self.vec_int(r)

class AtsasDatGnomDdf:
    def __init__(self, path):
        self.datgnom = DatgnomExecutor()
        self.path = path
        self.array, _ = np_loadtxt_robust(path)
        self.q = self.array[:,0]
        self.y = self.array[:,1]
        autorg_kek = AutorgKekAdapter( self.array )
        self.guinier_result = autorg_kek.run()

    def guess_best_dmax(self):
        temp_folder = get_setting('analysis_folder') + '/.temp'
        if not os.path.exists(temp_folder):
            from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
            mkdirs_with_retry(temp_folder)
        self.out_file = temp_folder + '/datgnom_temp.out'
        rg = self.guinier_result.Rg
        self.result = self.datgnom.execute(self.path, rg, self.out_file)
        ret = self.get_pr_curve()
        r = ret[:,5]
        dmax = r[np.isfinite(r)][-1]
        return dmax

    def get_pr_curve(self, debug=False):
        from molass_legacy.ATSAS.DatGnom import datgnom_read_data

        def positive_value( v ):
            v_ = float(v)
            return v_ if v_ > 0 else np.nan

        exper_fit, real_space = datgnom_read_data(self.out_file, null_value=np.nan, null_func=positive_value)

        num_rows = max(len(exper_fit), len(real_space))
        rows = []
        for i in range(num_rows):
            if i < len(exper_fit):
                row = exper_fit[i]
            else:
                row = [np.nan] * 5
            if i < len(real_space):
                row += real_space[i]
            else:
                row += [np.nan] * 3
            rows.append( row )

        ret = np.array(rows)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            fig, axes = plt.subplots(nrows=1,  ncols=2, figsize=(14, 6))
            ax1, ax2 = axes

            s = ret[:,0]
            jexp = ret[:,1]
            iexp = ret[:,4]
            r = ret[:,5]
            pr = ret[:,6]

            ax1.plot(s, np.log10(jexp))
            ax1.plot(s, np.log10(iexp))
            ax2.plot(r, pr)

            fig.tight_layout()
            plt.show()

        return ret
