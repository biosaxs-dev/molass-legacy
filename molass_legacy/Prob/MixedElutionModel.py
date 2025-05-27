# coding: utf-8
"""
    MixedElutionModel.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import molass_legacy.KekLib.DebugPlot as plt

USE_DPHEM_REDUCTIONE = False

def guess_n_components(x, y, p=0.3):
    if p is None:
        x_ = x
        y_ = y
    else:
        kth = int(len(y)*p)
        pp = np.argpartition(y, kth)
        pp_ = pp[kth:]
        x_ = x[pp_]
        y_ = y[pp_]

    X_ = np.expand_dims(y_,1)
    scores = []
    kmax = 8

    kx = np.arange(2, kmax+1)
    for k in kx:
      gmm = GaussianMixture(n_components=k, max_iter=40).fit(X=X_)
      scores.append((gmm.aic(X_), gmm.bic(X_)))

    scores = np.array(scores)

    if True:

        plt.push()

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,5))
        ax1, ax2, ax3 = axes

        ax1.set_title("%.2g percentile discarded" % (p*100))
        ax2.set_title("Information Criterions")
        ax1.plot(x, y, 'o')
        ax1.plot(x_, y_, 'o')
        ax2.plot(kx, scores[:,0], '-o', label='AIC')
        ax2.plot(kx, scores[:,1], '-o', label='BIC')
        ax2.legend()
        fig.tight_layout()
        plt.show()

        plt.pop()

    k = np.argmin(scores[:,1])
    return kx[k]

class MixedElutionModel(GaussianMixture):
    def __init__(self, **kwargs):
        GaussianMixture.__init__(self, **kwargs)

def spike_demo(in_folder, lpm_correct=False, eslice=None):
    from Prob.GaussianMixture import gm_curve, hist_to_source, debug_plot_gmm
    from Prob.gmm_dphem_learn import gmm_dphem_learn

    if in_folder is None:
        x, y = gm_curve()
    else:
        from RawData import RawXray
        from molass_legacy.Saxs.RankAnalysis import RankAnalysis
        rx = RawXray(in_folder)
        i = rx.get_row_index(0.02)
        x = np.arange(rx.data.shape[1])
        y = rx.data[i,:]
        if lpm_correct:
            from LPM import get_corrected
            y = get_corrected(y)
        if eslice:
            y = y[eslice]
            x = np.arange(len(y))

    sy = hist_to_source(x, y)

    k = guess_n_components(x, y)
    print('k=', k)

    temp_gmm = MixedElutionModel(n_components=k)
    temp_gmm.fit(X=np.expand_dims(sy,1))

    if USE_DPHEM_REDUCTIONE:
        tagk = 2
        opt = {'pimode': 'preserve', 
                'initmode': 'random'}
        result_gmm, post, info, diffel = gmm_dphem_learn(temp_gmm,tagk,opt)
    else:
        result_gmm = temp_gmm
        temp_gmm = None

    debug_plot_gmm(x, y, sy, result_gmm, gmm2=temp_gmm)
