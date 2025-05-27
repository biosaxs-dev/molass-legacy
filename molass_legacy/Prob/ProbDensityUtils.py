# coding: utf-8
"""
    ProbDensityUtils.py.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
try:
    # for numba 1.49 or later
    from numba.core.decorators import jit
except:
    from numba.decorators import jit
import molass_legacy.KekLib.DebugPlot as plt

@jit(nopython=True)
def generate_sample_data_impl(y, p, ymax):
    n = 10**p
    data = []
    for i, y_ in enumerate(y):
        for j in range(int(y_/ymax*n)):
            data.append(i)
    return np.array(data)

def generate_sample_data(y, p):
    ymax = np.max(y)
    return generate_sample_data_impl(y, p, ymax)

def set_consistent_base(ax1, axt):
    ymin1, ymax1 = ax1.get_ylim()
    ymint, ymaxt = axt.get_ylim()
    y_ = ymin1/ymax1*ymaxt
    axt.set_ylim(y_, ymaxt)

def plot_hist_data(y1, y2, data1, data2, axes=None, plot_class=plt):
    from DataUtils import get_in_folder

    if axes is None:
        fig, axes = plot_class.subplots(nrows=1, ncols=2, figsize=(14,7))
        fig.suptitle("Histogram of Generated Data overlaid with the Elution Curve from %s" % get_in_folder(), fontsize=20)

    ax1, ax2 = axes
    ax1t = ax1.twinx()
    ax2t = ax2.twinx()
    ax1t.grid(False)
    ax2t.grid(False)

    ax1.set_title("UV Data", fontsize=16)
    ax2.set_title("X-ray Data", fontsize=16)

    ax1.plot(y1)
    ax2.plot(y2)
    ax1t.hist(data1, bins=len(y1), alpha=0.5, color='orange')
    set_consistent_base(ax1, ax1t)
    ax2t.hist(data2, bins=len(y2), alpha=0.5, color='orange')
    set_consistent_base(ax2, ax2t)

    if axes is None:
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

def plot_hist_data_list(y_list, data_list, axes=None, titles=None, plot_class=plt):
    from DataUtils import get_in_folder

    if axes is None:
        fig, axes = plot_class.subplots(nrows=1, ncols=2, figsize=(14,7))
        fig.suptitle("Histogram of Generated Data overlaid with the Elution Curve from %s" % get_in_folder(), fontsize=20)

    for n, (ax, y, data) in enumerate(zip(axes, y_list, data_list)):
        axt = ax.twinx()
        axt.grid(False)
        if titles is not None:
            ax.set_title(titles[n], fontsize=16)
        ax.plot(y)
        axt.hist(data, bins=len(y), alpha=0.5, color='orange')
        set_consistent_base(ax, axt)

    if axes is None:
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)

def generate_samle_datasets(in_folder, kwargs, quad=False):
    lpm_correct = kwargs.pop('lpm_correct', False)
    lpm_2d = kwargs.pop('lpm_2d', False)
    smoothing = kwargs.get('smoothing', False)

    if lpm_correct:
        from CorrectedData import  CorrectedXray,  CorrectedUv
        ru = CorrectedUv(in_folder)
        rx = CorrectedXray(in_folder)
    else:
        from RawData import RawXray, RawUv
        ru = RawUv(in_folder)
        rx = RawXray(in_folder)

    i = ru.get_row_index(280)
    y1 = ru.data[i,:]
    if lpm_2d:
        from LPM import get_corrected
        y1 = get_corrected(y1)

    i = rx.get_row_index(0.02)
    y2 = rx.data[i,:]
    if lpm_2d:
        from LPM import get_corrected
        y2 = get_corrected(y2)

    if smoothing:
        from molass_legacy.KekLib.SciPyCookbook import smooth
        y1 = smooth(y1)
        y2 = smooth(y2)

    data1 = generate_sample_data(y1, 2)
    data2 = generate_sample_data(y2, 2)

    if not quad:
        return y1, y2, data1, data2

    i = ru.get_row_index(260)
    y3 = ru.data[i,:]
    if lpm_2d:
        from LPM import get_corrected
        y3 = get_corrected(y3)

    i = rx.get_row_index(0.04)
    y4 = rx.data[i,:]
    if lpm_2d:
        from LPM import get_corrected
        y4 = get_corrected(y4)

    if smoothing:
        from molass_legacy.KekLib.SciPyCookbook import smooth
        y3 = smooth(y3)
        y4 = smooth(y4)

    data3 = generate_sample_data(y3, 2)
    data4 = generate_sample_data(y4, 2)

    return [y1, y3, y2, y4], [data1, data3, data2, data4]
