"""
    UV.PlainCurve.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from sklearn.cluster import KMeans
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve

DIEHARD_LIMIT = 0.05        # 
FLATTENABLE_RATIO = 0.3     # > 0.21 for FER_OA_302, < 1.54 for 20211021
EXTINGUISHED_RATIO = 0.01   # 
CERTAINTY_LIMIT = 0.5
BRIGHT_LINE_WAVELENGTHS = [486, 581, 656]    # see https://www.klv.co.jp/corner/what-is-d2-lump.html
PEAK_CONSIDER_LIMIT = 0.5
MIN_NUM_CLUSTER_MEMBERS = 5

class PlainCurve(ElutionCurve):
    def __init__(self, x, y, guess_sigmoid=False, save_fig=False):
        self.x = x
        self.y = y
        self.sigmoid_params = None
        self.sigmoid_certainty = 0
        if guess_sigmoid:
            from molass_legacy.Trimming.Sigmoid import guess_bent_sigmoid
            ret = guess_bent_sigmoid(x, y, return_certainty=True, save_fig=save_fig)
            self.sigmoid_params = ret[0]
            self.sigmoid_certainty = ret[1]

    def save(self, path):
        np.savetxt(path, np.array([self.x, self.y]).T)

def check_diehardness(a_curve, a_curve2, debug=False):
    x = a_curve.x
    certainty = a_curve2.sigmoid_certainty
    if certainty < CERTAINTY_LIMIT:
        x0 = x[0]
    else:
        # good possibility of flow change
        x0 = a_curve2.sigmoid_params[1]
    y1 = a_curve.y
    y2 = a_curve2.y
    extinct_ratios = []
    for k, rec in enumerate(a_curve.peak_info):
        j = rec[1]
        if x[j] < x0:
            # skip a flow change (mis-recognized peak)
            pass
        else:
            ratio = y2[j]/y1[j]
            extinct_ratios.append((x[j], y2[j], ratio))
    if len(extinct_ratios):
        extinct_ratios = np.array(extinct_ratios)
        max_k = np.argmax(extinct_ratios[:,1])
        x_, y_, max_ratio = extinct_ratios[max_k]
        judge = y_ > y2[-1] and max_ratio > DIEHARD_LIMIT   # checking y_ > y2[-1] to avoid misjudge as in 20160628
    else:
        # as in 20200624_2
        judge, max_k, max_ratio = False, None, None

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("check_diehardness debug")
            ax1.plot(x, y1)
            ax2.plot(x, y2)
            if len(extinct_ratios) > 0:
                axt = ax2.twinx()
                axt.grid(False)
                axt.plot(*extinct_ratios.T, "o", color="red")
            fig.tight_layout()
            plt.show()

    return judge, max_k, max_ratio

def find_an_alternative_curve(D, wv, a_curve, index, logger, debug=False):
    ridge_curves = []

    pick = bisect_right(wv, get_setting('absorbance_picking'))
    zabs = get_setting('zero_absorbance')

    max_y = a_curve.max_y
    found = False
    i = None
    brights = []
    for w in BRIGHT_LINE_WAVELENGTHS:
        if w < wv[-1]:
            brights.append(bisect_right(wv, w))
    required_cond = wv > zabs
    for b in brights:
        required_cond[b-10:b+11] = False    # not near bright lines

    WSCALE = 0.0001

    for k, rec in enumerate(a_curve.peak_info):
        j = rec[1]
        y_pick = D[pick,j]
        if y_pick/max_y < PEAK_CONSIDER_LIMIT:
            # ignore minor peaks
            continue

        y = D[:,j]
        y -= np.min(y)
        extinguished = np.where(np.logical_and(required_cond, y/max_y < EXTINGUISHED_RATIO))[0]

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title(get_in_folder())
                ax.plot(wv, y)
                ax.plot(wv[extinguished], y[extinguished], "o")
                fig.tight_layout()
                plt.show()

        if len(extinguished) > MIN_NUM_CLUSTER_MEMBERS*2:

            X = np.array([wv[extinguished]*WSCALE, y[extinguished]]).T
            n_clusters = min(10, len(extinguished)//MIN_NUM_CLUSTER_MEMBERS)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
            labels = kmeans.labels_
            scores = []
            wheres = []
            for k in range(n_clusters):
                where = labels == k
                x_ = extinguished[where]
                y_ = y[x_]
                m = np.mean(y_)
                s = np.std(y_)
                scores.append((m**2+s**2)/len(x_))
                wheres.append(where)
            k = np.argmin(scores)
            where = wheres[k]

            if debug:
                print("n_clusters=", n_clusters)
                scale = 10000
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("KMeans Grouping debug")
                    ax.set_ylabel("Absorbance")
                    ax.set_xlabel("Wavelength")
                    ax.scatter(X[:,0]/WSCALE, X[:,1], c=labels, s=40, cmap='viridis')
                    ax.plot(X[where,0]/WSCALE, X[where,1], "o", color="red")
                    fig.tight_layout()
                    plt.show()

            where_ = np.where(where)[0]
            m = int(np.median(where_))

            found = True
            tail = extinguished[m:]
            i = tail[0]
            if debug:
                from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
                from time import sleep
                limit_y = max_y*EXTINGUISHED_RATIO
                debug_path = get_setting("debug_path")
                with plt.Dp():
                    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                    fig.suptitle("Finding an alternative Flat Curve for %s" % get_in_folder(), fontsize=20)
                    ax1.set_title("Whole Absorbance Curve", fontsize=16)
                    ax2.set_title("Zoomed Absorbance Curve", fontsize=16)
                    for ax in ax1, ax2:
                        ax.set_xlabel("Wavelength (nm)")
                        ax.set_ylabel("Absorbance")
                        ax.plot(wv, y, label="eno=%d" % j)
                        ax.plot(wv[extinguished], y[extinguished], "o", color="yellow", label="y/max_y < %g" % EXTINGUISHED_RATIO)
                        ax.plot(wv[tail], y[tail], "o", color="green", label="all-below tail")
                        ax.plot(wv[i], y[i], "o", color="red", label="first point of tail")
                    xmin, xmax = ax1.get_xlim()
                    ymin, ymax = ax1.get_ylim()
                    ax1.set_ylim(ymin, ymax)
                    w = wv[index]
                    ax1.plot([w, w], [ymin, ymax], ":", color="red", label="baseline wavelength")
                    xmin2 = xmin*0.1 + xmax*0.9
                    xmax2 = xmax
                    ymin2 = ymin
                    ymax2 = ymin*0.9 + ymax*0.1
                    ax2.set_xlim(xmin2, xmax2)
                    ax2.set_ylim(ymin2, ymax2)
                    ax2.plot([xmin2, xmax2], [limit_y, limit_y], ":", color="cyan", label="limit_y")
                    w = wv[i]
                    ax2.plot([w, w], [ymin, ymax], ":", color="red", label="baseline wavelength")
                    for ax in [ax1, ax2]:
                        ax.legend(fontsize=12)
                    fig.tight_layout()
                    if True:
                        plt.show()
                    else:
                        plt.show(block=False)
                        fig.savefig(debug_path)
                        sleep(1)

        elif len(extinguished) > 0:
            i = extinguished[0]

        if i is not None:
            ridge_curves.append((j, y, i))

    if debug:
        import molass_legacy.KekLib.DebugPlot as plt
        with plt.Dp():
            fig, ax = plt.subplots()
            for j, y, i in ridge_curves:
                ax.plot(wv, y, label="j=%d" % j)
                ax.plot(wv[i], y[i], "o", color="red")
            ax.legend()
            fig.tight_layout()
            plt.show()

    ridge_curves = sorted(ridge_curves, key=lambda x: -x[2])

    if found:
        if i < index:
            logger.info("alternative curve found at wv[%d]=%g will be replaced by one with the specified (longer) wv[%d]=%g.", 
                        i, wv[i], index, wv[index])
            i = index
        x = a_curve.x
        y = D[i,:]
        ret_curve = PlainCurve(x, y)
        w = wv[i]
    else:
        ret_curve = None
        w = None

    return ret_curve, w

def flattenable(ratio):
    if ratio is None:
        # as in 20200624_2
        return True
    else:
        return ratio < FLATTENABLE_RATIO

def make_secondary_e_curve_at(D, wv, e_curve, logger, wavelen=None, debug=False, save_fig=False):
    if wavelen is None:
        wavelen = get_setting("zero_absorbance")

    index = min(D.shape[0]-1, bisect_right(wv, wavelen))
    y = D[index,:]
    try:
        curve = PlainCurve(e_curve.x, y, guess_sigmoid=True, save_fig=save_fig)
        diehard, max_k, ratio = check_diehardness(e_curve, curve, debug=debug)
        if diehard:
            curve_, w = find_an_alternative_curve(D, wv, e_curve, index, logger, debug=debug)
            if curve_ is None:
                x = e_curve.x

                if flattenable(ratio):
                    flat_y = y.copy()
                    info = e_curve.peak_info[max_k]
                    flat_y[info[0]:info[2]] = 0
                else:
                    # as in 20211021 where ratio == 1.54
                    flat_y = np.zeros(len(y))
                    logger.warning("flat_y for uv baseline guess has been set to all zeros due to ratio=%.3g", ratio)

                if debug:
                    import molass_legacy.KekLib.DebugPlot as plt
                    from molass_legacy.Elution.CurveUtils import simple_plot
                    with plt.Dp():
                        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
                        fig.suptitle("Secondary Elutin Curve at %.3g nm" % wavelen, fontsize=20)
                        ax1.set_title("a_curve")
                        ax2.set_title("a_curve2")
                        simple_plot(ax1, e_curve)
                        ax2.plot(x, y)
                        ax2.plot(x, flat_y, ":")
                        fig.tight_layout()
                        plt.show()

                curve = PlainCurve(x, flat_y)
                logger.info("UV data seem to include a diehard peak like ferritin from the extinction ratio=%.3g", ratio)
            else:
                set_setting('flat_wavelength', w)
                logger.info("UV baseline will be estimated using a auto-corrected wavelength %.4g", w)
                curve = curve_
    except:
        etb = ExceptionTracebacker()
        logger.warning("failed to create curve at wavelength %.3g with exception %s" % (wavelen,  etb.last_lines()))
        curve = None    # but, will cause an exceotopn later

    # curve.save(r"D:\TODO\20230623\temp\secondary_e_curve-1.txt")

    return curve
