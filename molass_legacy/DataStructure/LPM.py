"""

    LPM.py

    Copyright (c) 2017-2025, SAXS Team, KEK-PF

"""
import numpy as np
import logging
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
from scipy.interpolate import LSQUnivariateSpline
from molass_legacy.SerialAnalyzer.BasePercentileOffset import base_percentile_offset
from molass_legacy.KekLib.SciPyCookbook import smooth
from molass_legacy.DataStructure.MatrixData import simple_plot_3d
from molass_legacy.SerialAnalyzer.ElutionBaseCurve import ElutionBaseCurve
from molass_legacy.Baseline.LambertBeer import BasePlane
from molass_legacy.Baseline.BaseScattering import get_baseplane

def get_corrected(y, x=None):
    if x is None:
        x = np.arange(len(y))
    sbl = ScatteringBaseline(y, x=x)
    A, B = sbl.solve()
    return y - (A*x+B)

def compute_base_surface(shape, line0, line1):
    n = shape[1]
    w = np.arange(n)/(n-1)
    return (1-w)*line0[:,np.newaxis] + w*line1[:,np.newaxis]

def debug_plot(data, base_surface=None):
    import molass_legacy.KekLib.DebugPlot as plt
    plt.push()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    simple_plot_3d(ax, data)
    if base_surface is not None:
        simple_plot_3d(ax, base_surface, alpha=0.5)
    fig.tight_layout()
    plt.show()
    plt.pop()    


class BestPercentile:
    def __init__(self, ecurve):
        self.x = ecurve.x
        knots   = np.linspace( 0, len(self.x), len(self.x)//10 )
        self.iknots  = knots[1:-1]
        self.size_sigma = ecurve.compute_size_sigma()

    def get(self, row):
        spline  = LSQUnivariateSpline(self.x, row, self.iknots)
        noisiness = self.compute_noisiness(spline, row)
        percentile =  base_percentile_offset(noisiness, size_sigma=self.size_sigma)
        return percentile

    def compute_noisiness(self, spline, Z):
        pp  = np.percentile( Z, [95, 5] )
        height  = pp[0] - pp[1]
        noisiness = np.std( Z - spline(self.x) ) / height
        return noisiness

class LPM_3d:
    def __init__(self, data, ecurve_y=None, smoothing=True, integral=False, for_all_q=True, e_index=None, end_slices=None, debug=False):
        # debug_plot(data)
        self.logger = logging.getLogger(__name__)

        if integral:
            from molass_legacy.Baseline.Baseline import integrative_curve
            smoothing = False
            self.logger.info("smoothing will not be applied in LPM, which is yet to be supported with integral.")

        if ecurve_y is None:
            self.logger.info("applying LPM correction with fixed percentile.")
        else:
            self.logger.info("applying LPM correction varying percentile.")
            ebase_curve = ElutionBaseCurve(ecurve_y)
            bp = BestPercentile(ebase_curve)

        if for_all_q:
            if smoothing:
                base_points_list = []
                xlim = data.shape[1] - 1
                end_points = [0, xlim]
                points = None

                for k, row in enumerate(data):
                    try:
                        sbl = ScatteringBaseline(row, suppress_warning=True)
                        if ecurve_y is None:
                            A, B = sbl.solve()
                        else:
                            percentile = bp.get(row)
                            A, B = sbl.solve(p_final=percentile)
                        points = [A*x+B for x in end_points]
                    except Exception as exc:
                        assert points is not None
                        self.logger.warning("previous points [%g, %g] are used as %d-th end_points due to '%s'", *points, k, str(exc))
                    base_points_list.append(points)

                base_points = np.array(base_points_list)
                line0 = smooth(base_points[:,0])
                line1 = smooth(base_points[:,1])
                base_surface = compute_base_surface(data.shape, line0, line1)
                corrected_data = data - base_surface
            else:
                x = np.arange(data.shape[1])
                corrected_data = data.copy()
                for k, row in enumerate(corrected_data):
                    try:
                        sbl = ScatteringBaseline(row, suppress_warning=True)
                        if ecurve_y is None:
                            A, B = sbl.solve()
                        else:
                            percentile = bp.get(row)
                            A, B = sbl.solve(p_final=percentile)
                    except Exception as exc:
                        assert points is not None
                        self.logger.warning("previous params (%g, %g) are used as %d-th end_points due to '%s'", A, B, k, str(exc))

                    baseline = A*x + B
                    if integral:
                        baseline = integrative_curve(row, baseline)
                    row -= baseline
        else:
            assert e_index is not None

            print('e_index=', e_index)
            x = np.arange(data.shape[1])
            y = ecurve_y
            # y = data[e_index,:]
            sbl = ScatteringBaseline(y, suppress_warning=True)
            if ecurve_y is None:
                A, B = sbl.solve()
            else:
                percentile = bp.get(y)
                A, B = sbl.solve(p_final=percentile)

            baseline = A*x + B
            if integral:
                baseline = integrative_curve(y, baseline)

            assert end_slices is not None
            self.logger.info("making integral BP with %s" % str(end_slices))
            dy = np.average(data[:,end_slices[1]], axis=1) - np.average(data[:,end_slices[0]], axis=1)
            BP = get_baseplane(baseline, end_slices, dy)

            if debug:
                import molass_legacy.KekLib.DebugPlot as plt
                plt.push()
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')

                xm = np.ones(len(x))*e_index

                if True:
                    simple_plot_3d(ax, BP)
                else:
                    simple_plot_3d(ax, data)
                    ax.plot(xm, x, y, color='orange')

                ax.plot(xm, x, baseline, color='red')
                x_ = np.arange(data.shape[0])
                y_ = np.ones(data.shape[0])*(len(y)-1)
                ax.plot(x_, y_, dy, color='green')

                fig.tight_layout()
                plt.show()
                plt.pop()
            corrected_data = data - BP

        self.data = corrected_data

    def adjust_with_mf(self, index, ecurve):
        bp = BasePlane(self.data, index, ecurve, denoise=False)
        bp.solve()
        print('bp.params=', bp.params)
        BP = bp.get_baseplane()
        self.data -= BP
        return BP

def get_corrected_matrix(D):
    xray_baseline_type = get_setting('xray_baseline_type')
    if xray_baseline_type == 0:
        retD = D.copy()
    else:
        integral = xray_baseline_type == 5
        lpm = LPM_3d(D, smoothing=True, integral=integral, for_all_q=True)
        retD = lpm.data
    return retD
