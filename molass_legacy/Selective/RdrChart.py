"""
    Selective.RdrChart.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def draw_rdr_chart(title, pv, rdr, devel=True):
    print("len(pv)=", len(pv))
    indeces = [str(i) for i in range(len(rdr))]
    props = ["%.3g" % p for p in pv]
    m = np.argmin(rdr)
    x = m
    # y = np.max(rdr)*0.5
    y = 0.05
    dy = (np.min(rdr) - y)*0.8
    head_length = y*0.2
    with plt.Dp(window_title='RDR_AD Chart', ok_only=True, ok_text="Close"):
        fig, ax = plt.subplots()
        ax.set_title(title, fontsize=16)
        ax.bar(indeces, rdr, alpha=0.5)
        ax.set_xticklabels(props, rotation=45, ha='right')
        ax.arrow(x=x,y=y,dx=0,dy=dy, width=0.5,
                 head_width=1, head_length=head_length, length_includes_head=True,
                 color='red', alpha=0.5)
        ax.text(x, y+abs(dy)*0.1, "(%.3g, %.2g)" % (pv[m], rdr[m]), ha="center", fontsize=16)
        ax.set_ylim(0, 0.1)
        fig.tight_layout()
        plt.show()
