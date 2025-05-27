"""
    V2PropOptimizer.RdrChart.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def draw_rdr_chart(title, rdr, devel=True):
    indeces = [str(i) for i in range(len(rdr))]
    m = np.argmin(rdr)
    x = m
    y = np.max(rdr)*0.5
    dy = (np.min(rdr) - y)*0.8
    head_length = y*0.2
    with plt.Dp(window_title='RDR_AD Chart', ok_only=True, ok_text="Close"):
        fig, ax = plt.subplots()
        ax.set_title(title, fontsize=16)
        ax.bar(indeces, rdr)
        ax.arrow(x=x,y=y,dx=0,dy=dy, width=0.5, head_width=1, head_length=head_length,length_includes_head=True, color='red', alpha=0.5)
        fig.tight_layout()
        plt.show()
