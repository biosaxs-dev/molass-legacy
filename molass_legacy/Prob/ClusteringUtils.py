# coding: utf-8
"""
    ClusteringUtils.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import time
import numpy as np
import scipy.stats as stats
from scipy.optimize import fsolve
from sklearn.cluster import AgglomerativeClustering, KMeans
import molass_legacy.KekLib.DebugPlot as plt
from .ProbDataUtils import generate_samle_datasets

def spike_demo(in_folder, **kwargs):
    n_components = kwargs.pop('n_components', 2)

    y1, y2, data1, data2 = generate_samle_datasets(in_folder, kwargs)

    st = time.time()
    # clustering = AgglomerativeClustering(n_clusters=n_components, linkage='ward').fit(X = np.expand_dims(data1,1))
    clustering = KMeans(n_clusters=n_components).fit(X = np.expand_dims(data1,1))
    elapsed_time = time.time() - st
    labels = clustering.labels_
    print("Elapsed time: %.2fs" % elapsed_time)
    print("Number of points: %i" % labels.size)
    print('labels[:5]=', labels[:5])
    label_values = np.unique(labels)
    print('label_values', label_values)

    num_bins = len(y1)
    print('num_bins=', num_bins)
    plt.push()
    fig  = plt.figure()
    ax = fig.gca()
    ax.hist(data1, bins=num_bins)
    ax.set_ylim(ax.get_ylim())

    for k in label_values:
        data = data1.copy()
        data[labels!=k] = 0
        hist, bin_edges = np.histogram(data, bins=num_bins, range=(0, num_bins-1))
        print([k], 'len(hist)=', len(hist))
        hist[0] = 0
        ax.plot(hist)

    fig.tight_layout()

    plt.show()
    plt.pop()
