"""
    HdcTheory.LaminarFlow.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib as mpl
# print(mpl.style.available)
# sns.set_theme()
mpl.style.use('dark_background')

def demo():

    tv = np.linspace(0, 0.1, 10)

    L = 30
    R = 1
    margin = 0.1
    y = np.linspace(margin, R*2 - margin, 100)
    mv = np.ones(len(y))*margin
    # rv = np.max([mv, np.abs(y - R)], axis=0)
    rv = np.abs(y - R)
    lv = rv/R
    u = 1 - lv**2

    fig, ax = plt.subplots(figsize=(10,2))
    ax.set_title("Animation of Laminar Flow in a Pipe")
    ax.set_axis_off()

    curves = []
    for t in tv:
        x = u*L*t**2/2
        curve, = ax.plot(x, y, "o", markersize=1)
        curves.append(curve)

    ax.set_xlim(0, 10)

    dt = 0.1
    def update(i):
        elapse_time = i*0.02 
        for k, t in enumerate(elapse_time + tv):
            x = u*L*t**2/2
            # q, r = divmod(x, 100)
            curves[k].set_xdata(x)
        return curves

    def init():
        return update(0)

    num_frames = 100
    anim = FuncAnimation(fig, update, init_func=init,
                               frames=num_frames, interval=100, blit=True)
    fig.tight_layout()
    # anim.save("laminar-flow.gif")
    plt.show()


if __name__ == "__main__":
    import sys
    sys.path.append("../lib")    
    # demo()
    demo()
