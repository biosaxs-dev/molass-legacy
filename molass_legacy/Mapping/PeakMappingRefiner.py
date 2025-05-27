# coding: utf-8
"""

    PeakMappingRefiner.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""
import copy
import numpy            as np
import logging
from scipy.stats        import pearsonr, linregress
from scipy.interpolate  import UnivariateSpline
from itertools          import combinations
import OurStatsModels   as sm
from molass_legacy.SerialAnalyzer.ElutionCurve       import ElutionCurve
from molass_legacy.Elution.CurveUtils  import simple_plot
from molass_legacy.KekLib.SciPyCookbook      import smooth
from molass_legacy.KekLib.BasicUtils         import Struct
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
import molass_legacy.KekLib.DebugPlot        as plt

NUM_ROOTS_TO_AVOID_NOISE    = 5     # dupicate

class PeakMappingRefiner:
    def __init__(self):
        self.debug = False

    def find_counter_part( self, start, stop, top_x1, peak_rec, A_, B_, curve2 ):
        x = curve2.x
        y = curve2.y
        mapped_top_x = A_ * top_x1 + B_
        slice_ = slice(start, stop)
        x_ = x[slice_]
        y_ = y[slice_]

        if len(x_) < 3:
            raise RuntimeError("Invalid slice: " + str((start, stop)))

        slope, intercept = linregress( x_, y_)[0:2]
        baseline = slope * x_ + intercept
        ry_ = y_ - baseline

        try:
            sry_ = smooth(ry_, window_len=20)
        except ValueError:
            raise RuntimeError("smooth failed. for len(ry_)=%d." % len(ry_))

        neighbor_spline = UnivariateSpline(x_, sry_, s=0, ext=3)
        d1 = neighbor_spline.derivative(1)
        d1y = d1(x_)
        d1_spline = UnivariateSpline(x_, d1y, s=0, ext=3)
        d1_roots = d1_spline.roots()

        if self.debug:
            print('len(d1_roots)=', len(d1_roots) )

        # assert len(d1_roots) < NUM_ROOTS_TO_AVOID_NOISE

        d2 = neighbor_spline.derivative(2)
        top_candidates = d1_roots[ d2(d1_roots) < 0 ]

        assert len(top_candidates) > 0, "len(top_candidates) == 0"

        dist = np.abs(top_candidates - mapped_top_x)
        n = np.argmin(dist)
        top_x = top_x_c = top_candidates[n]

        if curve2.spline(mapped_top_x) > curve2.spline(top_x):
            top_x = mapped_top_x

        # assert curve2.locally_tall_enough(top_x)      # not appropriate for SUB_TRN1

        prev_boundaries = d1_roots[ np.logical_and(d1_roots < top_x, d2(d1_roots) > 0) ]
        if len(prev_boundaries) > 0:
            prev_boundary = int(prev_boundaries[-1] + 0.5)
        else:
            # TODO:
            prev_boundary = start

        next_boundaries = d1_roots[ np.logical_and(d1_roots > top_x, d2(d1_roots) > 0) ]
        if len(next_boundaries) > 0:
            next_boundary = int(next_boundaries[0] + 0.5)
        else:
            # TODO
            next_boundary = stop

        r_noise = np.average(np.abs(ry_ - sry_))
        boundaries = [prev_boundary, next_boundary]
        r_height = np.max(neighbor_spline([top_x, top_x_c])) - np.max(neighbor_spline(boundaries))
        # print('r_noise/r_height=', r_noise/r_height)
        r_noise_ratio = r_noise/r_height
        # assert r_noise_ratio < 0.5, "r_noise_ratio(%.3g) >= 0.5" % (r_noise_ratio)

        top_x_i = int(top_x + 0.5)
        range_info = [prev_boundary, top_x_i, next_boundary]

        if False:
            plt.push()
            fig, axes = plt.subplots( figsize=(21, 7), nrows=1, ncols=3 )
            fig.suptitle("find_counter_part debug")
            ax1, ax2, ax3 = axes
            ax1.set_title("Whole View")
            ax2.set_title("Neighbor View")
            ax3.set_title("Transformed View")
            ax1.plot(x, y)
            for k, ax in enumerate(axes):
                if k < 2:
                    ax.plot(x_, y_, label='neighborhood')
                    ax.plot(x_, baseline, ':', color='red', label='baseline')
                    ax.plot(mapped_top_x, curve2.spline(mapped_top_x), 'o', color='yellow', label='mapped peak top')
                    ax.plot(boundaries, curve2.spline(boundaries), 'o', color='green', label='new boundaries')
                    ax.plot(top_candidates, curve2.spline(top_candidates), 'o', color='pink', markersize=8, label='top_candidates')
                    ax.plot(top_x, curve2.spline(top_x), 'o', color='red', label='new peak top')
                else:
                    ax.plot(x_, ry_, label='neighborhood')
                    ax.plot(x_, sry_, label='smoothed neighborhood')
                    ax.plot(d1_roots, neighbor_spline(d1_roots), 'o', color='cyan', markersize=10, label='d1_roots')
                    ax.plot(boundaries, neighbor_spline(boundaries), 'o', color='green', label='new boundaries')
                    ax.plot(top_candidates, neighbor_spline(top_candidates), 'o', color='pink', label='top_candidates')
                    ax.plot(top_x, neighbor_spline(top_x), 'o', color='red', label='new peak top')

                ax.legend()
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            plt.show()
            plt.pop()

        # assert len(top_candidates) == 1

        peak_rec = [int(A_*j + B_ + 0.5) for j in peak_rec]

        bottom_info = Struct(prev_bottom=prev_boundary, next_bottom=next_boundary )

        return Struct(top_x=top_x, peak_rec=peak_rec, bottom_info=bottom_info, boundary_score=0, range_info=range_info)

    def maps_well_linearly(self, curve1, curve2, A, B):
        if len(curve1.peak_info) == 1:
            return True

        y = curve1.peak_top_x + curve1.boundaries
        x = curve2.peak_top_x + curve2.boundaries

        correlation = pearsonr(x, y)[0]
        print('correlation=', correlation)  # correlation = 

        if correlation > ADEQUACY_CORRELATION:
            return True

        if len(curve1.boundaries) < 2:
            # 20181203
            # better check if it is easily done
            return correlation > RISKY_CORRELATION_LIMIT

        # in cases like 20180206
        for k in range(len(curve1.boundaries)):
            boundaries1 = copy.deepcopy(curve1.boundaries)
            boundaries2 = copy.deepcopy(curve2.boundaries)
            boundaries1.pop(k)
            boundaries2.pop(k)
            y = curve1.peak_top_x + boundaries1
            x = curve2.peak_top_x + boundaries2

            correlation = pearsonr(x, y)[0]
            print([k], 'correlation=', correlation)  # correlation = 
            if correlation > ADEQUACY_CORRELATION:
                b1 = curve1.boundaries[k]
                mb = int(A*curve2.boundaries[k] + B + 0.5)
                curve1.boundaries[k] = mb
                self.logger.info("moved %dth bottom from %d to %d." % (k, b1, mb))
                return True

        return False

    def refine_mapping_params(self, curve1, curve2, max_params, max_correl, max_simil, index, reversed_):
        # assert False
        B, A = max_params

        top_x1 = curve1.peak_top_x
        top_x2 = curve2.peak_top_x
        pk_info1 = curve1.peak_info
        pk_info2 = curve2.peak_info

        x = []
        y = []
        w = []

        for m in range(len(pk_info1)):
            y.append(top_x1[m])
            x.append(top_x2[m])
            w.append(8)
            rec1 = pk_info1[m]
            rec2 = pk_info2[m]
            for k in [0,2]:
                y.append(rec1[k])
                x.append(rec2[k])
                w.append(1)

        X   = sm.add_constant(x)
        mod = sm.WLS( y, X, weights=w )
        res = mod.fit()
        rB, rA = res.params

        correl, simil = self.compute_mapped_curve_simimarity(res.params, x, y, w, index, curve1, curve2, reversed_ )

        if False:
            from CallStack import CallStack
            cstack = CallStack()
            print('refine_mapping_params call stack=', cstack)
            plt.push()
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
            ax1, ax2 = axes

            ax1.plot(curve1.x, curve1.y)
            x_ = A * curve2.x + B
            y_ = curve2.y * curve1.max_y/curve2.max_y
            ax1.plot(x_, y_)

            ax2.plot(curve1.x, curve1.y)
            x_ = rA * curve2.x + rB
            y_ = curve2.y * curve1.max_y/curve2.max_y
            ax2.plot(x_, y_)

            fig.tight_layout()
            plt.show()
            plt.pop()

        assert correl + simil > max_correl + max_simil

        return rA, rB

    def peak_tops_correct(self, curve1, m, peak_top_x1, peak_info1, curve2, k, peak_top_x2, peak_info2, A, B):
        scale = curve2.height/curve1.height
        peak_height1 = curve1.spline(peak_top_x1) * scale
        peak_height2 = curve2.spline(peak_top_x2)
        if peak_height1 > peak_height2:
            A_  = 1/A
            B_  = -B/A
            peak_top_x2 = A_ * peak_top_x1 + B_
            peak_info2  = [ int(A_ * p + B_ + 0.5 ) for p in peak_info1 ]
            if m > 0 and m - 1 < len(curve1.boundaries):
                boundary1 = curve1.boundaries[m - 1]
                boundary2 = A_ * boundary1 + B_
            else:
                assert False
        else:
            peak_top_x1 = A * peak_top_x2 + B
            peak_info1  = [ int(A * p + B + 0.5 ) for p in peak_info2 ]
            if k > 0 and k - 1 < len(curve2.boundaries):
                boundary2 = curve2.boundaries[k - 1]
                boundary1 = A * boundary2 + B
            else:
                assert False
        return Struct(
                    peak_top_x1=peak_top_x1, 
                    peak_info1=peak_info1, 
                    prev_boundary1=boundary1, 
                    peak_top_x2=peak_top_x2, 
                    peak_info2=peak_info2, 
                    prev_boundary2=boundary2
                    )
