"""

    Rank.SrrStats.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

def demo(srr_csv):
    srr_list = []
    with open(srr_csv) as fh:
        for line in fh:
            str_values = line[:-1].split(',')
            values = [np.nan if s == "None" else float(s) for s in str_values[-6:]]
            srr_list.append(values)

    srr_array = np.array(srr_list)

    if srr_array.shape[1] > 2:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))
    else:
        fig, ax2 = plt.subplots()

    ax1.set_title("SCD Distribution", fontsize=16)
    scd = srr_array[:,0]
    ax1.hist(scd, bins=40, label="SCD")
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax)
    for name, scd_ in [("20180526/GI", 5.37495), ("20180526/OA", 10.5122), ("20190527/EndoAB", 0.469393)]:
        ax1.plot([scd_, scd_], [ymin, ymax], ":", label="%s: SCD=%.3g" % (name, scd_))
    ax1.legend()
    ax1.set_xlabel("SCD")
    ax1.set_ylabel("Counts")

    ax2.set_title("Two Types of SRR Distributions", fontsize=16)

    ax2.hist(srr_array[:,2], bins=40, label="whole q-range", alpha=0.5)
    ax2.hist(srr_array[:,3], bins=40, label="small angle only", alpha=0.5)

    ymin, ymax = ax2.get_ylim()
    ax2.set_ylim(ymin, ymax)

    for name, srr in [("20180526/GI", 0.511472), ("20180526/OA", 0.638757), ("20190527/EndoAB", 0.957003)]:
        ax2.plot([srr, srr], [ymin, ymax], ":", label="%s: SRR=%.3g" % (name, srr))

    ax2.legend()
    ax2.set_xlabel("SRR")
    ax2.set_ylabel("Counts")

    if srr_array.shape[1] > 4:
        rg1 = srr_array[:,4]
        rg2 = srr_array[:,5]
        rdr = (rg2 - rg1)*2/(rg1 + rg2)
        ax3.set_title("RDR Distribution", fontsize=16)
        ax3.hist(rdr, bins=40, label="RDR")

        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(ymin, ymax)

        for name, rdr in [("20180526/GI", -0.034760479), ("20180526/OA", 0.144374836), ("20190527/EndoAB", 0.033599555)]:
            ax3.plot([rdr, rdr], [ymin, ymax], ":", label="%s: RDR=%.3g" % (name, rdr))

        ax3.legend()
        ax3.set_xlabel("RDR")
        ax3.set_ylabel("Counts")

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    import os
    import sys
    this_dir = os.path.dirname( os.path.abspath( __file__ ))
    home_dir = os.path.abspath( this_dir + '/..' )
    sys.path.append(home_dir)
    from molass_legacy.KekLib.OurMatplotlib import mpl_1_5_backward_compatible_init
    mpl_1_5_backward_compatible_init()
    # srr_csv = r"D:\TODO\20230904\SRR-save\srr\srr.csv"
    srr_csv = r"D:\TODO\20230904\SRR-save3\srr\srr.csv"
    demo(srr_csv)
