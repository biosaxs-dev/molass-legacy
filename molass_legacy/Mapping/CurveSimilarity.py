"""
    CurveSimilarity.py

        evaluation of simularity between curves

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize

USE_WLS_FOR_APPROX_MAPPING  = True
if USE_WLS_FOR_APPROX_MAPPING:
    import statsmodels.api  as sm
else:
    from scipy              import stats

BETTER_MAPPING_WANTED_SCORE = 0.5

def compute_chi_square(x, y, slope, intercept, weights):
    return np.sum( (x*slope + intercept - y)**2 * weights )/np.sum(weights)

class RegressInfo:
    def __init__(self, x, y, w, chisqr):
        self.x = x
        self.y = y
        self.w = w
        self.chisqr = chisqr

def compute_similarlity_factors(A, B, reg_info, a_curve, x_curve, reg_info_ref=None):
    size = len(x_curve.x)

    start = 0
    mapped_list = []
    max_peak_width = None
    for k, info in enumerate(x_curve.peak_info):
        left_i, top_i, right_i = info
        peak_width = right_i - left_i
        if max_peak_width is None or peak_width < max_peak_width:
            max_peak_width = peak_width
        if k < len(x_curve.boundaries):
            stop = x_curve.boundaries[k]
        else:
            stop = size
        i = np.arange(start, stop)
        j = A*i + B
        sync_y = a_curve.spline(j)
        top_j = A*top_i + B
        scale = x_curve.spline(top_i)/a_curve.spline(top_j)
        mapped_list.append(sync_y*scale)
        start = stop

    mapped_vector = np.hstack(mapped_list)
    spline_vector = x_curve.y_for_spline

    if False:
        import molass_legacy.KekLib.DebugPlot as plt
        fig = plt.figure()
        ax = fig.gca()
        ax.plot(spline_vector, label='spline_vector')
        b = x_curve.boundaries
        ax.plot(b, x_curve.spline(b), 'o', color='yellow', label='boundaries')
        ax.plot(mapped_vector, label='mapped_vector')
        ax.legend()
        fig.tight_layout()
        plt.show()

    if reg_info_ref is None:
        wpoints = reg_info.x
        reg_info_ref = reg_info
    else:
        wpoints = reg_info_ref.x

    weights = np.zeros(len(x_curve.x))
    hwidth = size*0.025
    for p in wpoints:
        f = max(0, int(p - hwidth))
        t = min(size, int(p + hwidth))
        weights[f:t] = 1

    y = np.array(reg_info_ref.y)
    x = np.array(reg_info_ref.x)
    y_ = A*x + B

    if False:
        fig = plt.figure(figsize=(21,6))
        fig.suptitle("compute_similarlity_factors")
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        ax3t = ax3.twinx()
        ax1.plot(a_curve.y, color='blue')
        ax1.plot(y, a_curve.spline(y), 'o', color='yellow')
        ax1.plot(y_, a_curve.spline(y_), 'o', color='orange')
        ax2.plot(x_curve.y, color='orange')
        ax2.plot(x, x_curve.spline(x), 'o', color='yellow')
        ax3.plot(mapped_vector, color='blue')
        ax3.plot(spline_vector, color='orange')
        ax3t.plot(weights)
        fig.tight_layout()
        plt.show()

    weights_ = weights/np.sum(weights)

    v_chisqr = np.sum(((mapped_vector - spline_vector)*weights_/x_curve.max_y)**2)
    # h_chisqr = reg_info.chisqr/max_peak_width**2
    w_ = reg_info_ref.w/np.sum(reg_info_ref.w)
    h_chisqr = np.sum(((y_ - x)*w_/max_peak_width)**2)
    # print('v_chisqr=', v_chisqr, 'h_chisqr=', h_chisqr)
    return v_chisqr, h_chisqr

class CurveSimilarity:
    def __init__(self, a_curve, x_curve, orig_a_curve=None, debug=False):
        if debug:
            from importlib import reload
            import Mapping.PeakMapper
            reload(Mapping.PeakMapper)
        from .PeakMapper import PeakMapper

        if orig_a_curve is None:
            orig_a_curve = a_curve
        self.orig_a_curve = orig_a_curve

        if False:
            # this is for plotting the untrimmed curves
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                fig.suptitle("CurveSimilarity debug (1)")
                ax1.plot(a_curve.x, a_curve.y)
                ax2.plot(x_curve.x, x_curve.y)
                fig.tight_layout()
                plt.show()

        pm = PeakMapper( a_curve, x_curve, debug=False )
        a_curve_, x_curve_ = pm.get_mapped_curves()
        self.a_curve    = a_curve_
        self.x_curve    = x_curve_
        slope, intercept = pm.best_info[1]

        if len(x_curve_.peak_info) == 1:
            score = self.compute_rough_mapping_score(slope, intercept)
            if score > BETTER_MAPPING_WANTED_SCORE:
                improved_result = self.try_improve_mapping(slope, intercept, score)
                if improved_result is not None:
                    if debug:
                        import molass_legacy.KekLib.DebugPlot as plt

                        def plot_mapped_curves(ax, a, b):
                            x = x_curve_.x
                            y = x_curve_.y
                            uv_x = x*a + b
                            uv_y = a_curve_.spline(uv_x)
                            ax.plot(x, y, color='orange')
                            ax.plot(x, uv_y*x_curve_.max_y/a_curve_.max_y, ":", color="blue")

                        with plt.Dp():
                            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                            fig.suptitle("CurveSimilarity debug (2)")
                            ax1.set_title("previous")
                            ax2.set_title("improved")
                            plot_mapped_curves(ax1, slope, intercept)
                            plot_mapped_curves(ax2, *improved_result)
                            fig.tight_layout()
                            plt.show()

                    slope, intercept = improved_result

        self.slope      = slope
        self.intercept  = intercept
        self.mapped_info = pm.mapped_info
        self.reliable = pm.reliable

    def get_mapped_info(self):
        return self.mapped_info

    def compute_rough_mapping_score(self, a, b):
        x = self.x_curve.x
        y = self.x_curve.y
        uv_x = x*a + b
        uv_y = self.a_curve.spline(uv_x)
        mapped_y = uv_y*self.x_curve.max_y/self.a_curve.max_y
        area1 = np.sum(y)
        area2 = np.sum(mapped_y)
        return abs(area1 - area2)*2/(area1 + area2)

    def try_improve_mapping(self, init_a, init_b, score):
        import logging
        logger = logging.getLogger(__name__)
        logger.info("tried to improve mapping due to the score %.3g", score)

        lower, _, upper = self.x_curve.peak_info[0]
        slice_ = slice(lower, upper)
        x_ = self.x_curve.x[slice_]
        y_ = self.x_curve.spline(x_)

        def objective(p):
            a, b = p
            uv_x_ = x_*a + b
            uv_y_ = self.a_curve.spline(uv_x_)
            mp_y_ = uv_y_*self.x_curve.max_y/self.a_curve.max_y
            return np.sum((mp_y_ - y_)**2)

        ret = minimize(objective, (init_a, init_b))
        new_score = self.compute_rough_mapping_score(*ret.x)

        if new_score < score:
            logger.info("got a better score %.3g", new_score)
            return ret.x
        else:
            logger.info("got a worse score %.3g, which will be discarded.", new_score)
            return None

    def mapped_value(self, i):
        j = self.slope * i + self.intercept
        return max(0, min(len(self.a_curve.y)-1, j))

    def mapped_int_value(self, i):
        j = int( self.slope * i + self.intercept + 0.5 )
        return max(0, min(len(self.a_curve.y)-1, j))

    def inverse_int_value(self, j, cut=True):
        i = int( (j - self.intercept)/self.slope + 0.5 )
        ret_i = max(0, min(len(self.x_curve.y)-1, i)) if cut else i
        return ret_i

    def plot_the_mapped_state(self, mapped_plot_closure=None):
        import molass_legacy.KekLib.DebugPlot as plt
        from molass_legacy.Elution.CurveUtils import simple_plot
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
        ax1, ax2, ax3 = axes
        simple_plot(ax1, self.a_curve, color='blue')
        simple_plot(ax2, self.x_curve, color='orange')
        simple_plot(ax3, self.x_curve, color='orange', legend=False)
        y = self.get_uniformly_mapped_a_curve()
        ax3.plot(y, ':', color='blue')
        if mapped_plot_closure is not None:
            mapped_plot_closure(ax3)
        fig.tight_layout()
        plt.show()

    def get_extended_x(self):
        start = self.inverse_int_value(0, cut=False)
        stop = self.inverse_int_value(len(self.a_curve.x), cut=False)
        x = np.arange(start, stop)
        return x

    def get_uniformly_mapped_a_curve(self, x=None):
        if x is None:
            x = self.x_curve.x
        j = self.slope * x + self.intercept
        scale = self.x_curve.max_y/self.orig_a_curve.max_y
        mapped_y = self.orig_a_curve.spline(j)*scale
        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            plt.push()
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
            fig.suptitle("get_uniformly_mapped_a_curve")
            ax1, ax2, ax3 = axes
            simple_plot(ax1, self.orig_a_curve, "self.orig_a_curve")
            simple_plot(ax2, self.x_curve, "self.x_curve")
            simple_plot(ax3, self.x_curve, "self.x_curve", legend=False)
            ax3.plot(self.x_curve.x, mapped_y, label='mapped_y')
            ax3.legend()
            fig.tight_layout()
            plt.show()
            plt.pop()
        return mapped_y

    def compute_similarity_at( self, j, width_ratio=0.1, debug=False ):
        try:
            return self.compute_similarity_impl(j, width_ratio, debug=debug)
        except:
            # occured with 20170426
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            import logging
            etb = ExceptionTracebacker()
            logger = logging.getLogger(__name__)
            logger.warning("compute_similarity_at(%d) failed." % j + etb.last_lines() )
            return 0

    def compute_similarity_impl(self, j, width_ratio, debug=False):
        # print('compute_similarity_impl debug=', debug)

        ix  = (j - self.intercept)/self.slope
        x   = self.x_curve.x
        y   = self.x_curve.y

        i   = int(ix + 0.5)
        hw  = int( len(self.x_curve.x) * width_ratio/2 )
        start   = max( 0, i - hw )
        stop    = min( len(x), i + hw + 1 )

        cover_ratio = (stop - start)/(2*hw)     # =0.44 for right fc of HIF, which should not be discarded
        # print('cover_ratio=', cover_ratio)

        ratio_x = self.x_curve.spline(ix)/self.x_curve.max_y
        # print( 'ix=', ix, 'ratio_x=', ratio_x )

        slice_ = slice(start, stop)
        x_  = x[slice_]
        y_  = y[slice_]
        sy_ = self.x_curve.spline(x_)
        sy_min  = np.min(sy_)
        sy_max  = np.max(sy_)

        j_  = self.slope * x_ + self.intercept
        ay_ = self.a_curve.spline(j_)
        ay_min  = np.min(ay_)
        ay_max  = np.max(ay_)

        x_width = x_[-1] - x_[0]
        y_hight = sy_max - sy_min

        scale = y_hight/(ay_max - ay_min)
        ays = ay_ * scale
        x_spline = self.x_curve.spline

        def obj_func(params):
            A, B = params
            return np.sum((ays - (x_spline(x_ + A) + B ))**2)

        bounds = ((-x_width*0.3, x_width*0.3), (-y_hight*0.3, y_hight*0.3))

        result = minimize(obj_func, (0, 0), bounds=bounds)
        redchi = result.fun/(len(x_) - 2)
        diff_ratio  = np.sqrt(redchi) / y_hight
        similarity  = max( 0, 1 - diff_ratio )

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from matplotlib.patches     import Polygon
            from molass_legacy._MOLASS.SerialSettings         import get_setting

            opt_A, opt_B = result.x

            fy  = self.x_curve.spline( x_ + opt_A ) + opt_B

            ax  = plt.gca()
            ax.cla()
            ax.set_title( 'Curve Simualrity near j=%.3g in ' % j + get_setting('in_folder'), fontsize=16 )
            # ax.plot( x_, y_, color='orange' )
            ax.plot( x_, sy_, color='orange', label='original Xray (spline)' )
            ax.plot( x_, ays, label='mapped UV (spline)' )
            ax.plot( x_, fy, color='red', label='shifted Xray (spline)' )

            poly_points = list(zip( x_, ays )) + list( reversed( list( zip( x_, fy ) ) ) )
            diff_poly   = Polygon( poly_points, alpha=0.2 )
            ax.add_patch(diff_poly)
            ax.legend( fontsize=16 )

            ti  = i - start
            fc_y = ays[ti]
            ann_upper = fc_y  > (sy_min + sy_max)/2
            fc_y_delta = ( -1 if ann_upper else +1 ) * y_hight*0.2

            ax.annotate( "j=%.3g" % j, fontsize=16, xy=(i, fc_y), xytext=(i, fc_y + fc_y_delta ),
                        ha='center', va='center',
                        arrowprops=dict( headwidth=5, width=0.5, color='black', shrink=0.05) )

            tx  = x_[0] * 0.8  + x_[-1] * 0.2
            w   = 0.65 if ann_upper else 0.35
            ty  = sy_min * w + sy_max * (1 - w)

            ax.text( tx, ty, "Similarity=%.2g" % similarity, fontsize=50, alpha=0.2, va='center' )

            plt.tight_layout()
            plt.show()

        return similarity * cover_ratio
        # return 0

    def compute_whole_similarity(self):
        mapped_y = self.get_uniformly_mapped_a_curve()
        diff_area = np.sum(np.abs(mapped_y - self.x_curve.y))
        whole_area = self.x_curve.max_y * len(self.x_curve.y)
        return abs(1 -  diff_area/whole_area)
