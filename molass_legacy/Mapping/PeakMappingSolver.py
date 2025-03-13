"""
    PeakMappingSolver.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from sklearn.cluster import KMeans
from molass_legacy.Elution.CurveUtils import simple_plot

PENALTY_DIFF_RATIO = 0.1    # > 0.05 which is too small for 20161104/BL-6A/HIF

class PeakMappingSolver:
    def __init__(self, a_curve, x_curve, mapping, debug=False):
        self.logger = logging.getLogger(__name__)

        uv_x = a_curve.x
        uv_y = a_curve.y

        self.penalty_diff = len(uv_x)* PENALTY_DIFF_RATIO

        A, B = mapping
        xr_x = x_curve.x
        xr_y = x_curve.y
        x = A*xr_x + B

        uv_pt_x = a_curve.peak_top_x
        xr_pt_x = x_curve.peak_top_x

        mp_pt_x = A*xr_pt_x + B
        points = np.concatenate([uv_pt_x, mp_pt_x])
        X = points.reshape((len(points), 1))
        if debug:
            print("points=", points)
            scale = a_curve.max_y/x_curve.max_y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("PeakMappingSolver (1)")
                simple_plot(ax, a_curve)
                ax.plot(x, scale*xr_y, ":")
                fig.tight_layout()
                plt.show()

        uv_np = len(uv_pt_x)
        xr_np = len(xr_pt_x)
        max_n = max(uv_np, xr_np)

        min_score = None
        opt_mappers = None
        for n in range(max_n, max_n+2):
            kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
            labels = kmeans.labels_
            score, mappers = self.evaluate_grouping(n, uv_pt_x, mp_pt_x, labels[:uv_np], labels[uv_np:], debug=debug)
            if debug:
                print([n], "labels=", labels, "score=", score, "mappers=", mappers)
            if min_score is None or score < min_score:
                min_score = score
                opt_mappers = mappers

        if debug:
            print("min_score=", min_score)
            print("opt_mappers=", opt_mappers)
            uv_mapper, xr_mapper = opt_mappers
            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20,5))
                fig.suptitle("PeakMappingSolver (2)")
                simple_plot(ax1, a_curve, color="blue", legend=False)
                ax1.plot(x, xr_y, ":", label="mapped")
                for px in x_curve.peak_top_x:
                    mx = A*px + B
                    ax1.plot(mx, x_curve.spline(px), "o", color="yellow")
                ax1.legend()
                for i, j in enumerate(uv_mapper):
                    if j < 0:
                        continue
                    px = uv_pt_x[i]
                    ax1.plot(px, a_curve.spline(px), "o", color="cyan", markersize=20, alpha=0.2)

                simple_plot(ax2, x_curve, color="orange")

                for i, j in enumerate(xr_mapper):
                    if j < 0:
                        continue
                    px = xr_pt_x[i]
                    ax2.plot(px, x_curve.spline(px), "o", color="cyan", markersize=20, alpha=0.2)

                fig.tight_layout()
                plt.show()

        self.opt_mappers = opt_mappers

    def get_opt_mappers(self):
        return self.opt_mappers

    def evaluate_grouping(self, n, uv_pt_x, mp_pt_x, uv_labels, xr_labels, debug=False):

        uv_dict = dict([ (j, i) for i, j in enumerate(uv_labels) ])
        xr_dict = dict([ (j, i) for i, j in enumerate(xr_labels) ])

        if debug:
            print([n], "uv_labels=", uv_labels)
            print([n], "xr_labels=", xr_labels)
            print("uv_dict=", uv_dict)
            print("xr_dict=", xr_dict)

        uv_mapper = np.ones(len(uv_pt_x))*(-1)
        xr_mapper= np.ones(len(mp_pt_x))*(-1)

        total_diff = 0
        match_count = 0
        for label in range(n):
            uv_i = uv_dict.get(label)
            xr_i = xr_dict.get(label)
            if uv_i is None and xr_i is None:
                # assert False                  # true for 20191031
                self.logger.info("unexpected label %d in uv_labels=%s", label, uv_labels)
                self.logger.info("unexpected label %d in xr_labels=%s", label, xr_labels)
                diff = self.penalty_diff
            elif uv_i is not None and xr_i is not None:
                uv_mapper[uv_i] = xr_i
                xr_mapper[xr_i] = uv_i
                diff = (uv_pt_x[uv_i] - mp_pt_x[xr_i])**2
                match_count += 1
            else:
                diff = self.penalty_diff
            if debug:
                print([label], "diff=", diff)
            total_diff += diff

        score = (n - match_count)/n * total_diff

        if debug:
            print("---- uv_mapper=", uv_mapper)
            print("---- xr_mapper=", xr_mapper)
            print("---- match_count=", match_count)
            print("---- total_diff=", total_diff)
            print("---- score=", score)

        return score, (uv_mapper, xr_mapper)
