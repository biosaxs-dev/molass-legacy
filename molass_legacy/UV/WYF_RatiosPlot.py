"""
    UV.WYF_RatiosPlot.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn
seaborn.set()

SIGN_DICT = {
    "D" : -1,
    "C" :  0,
    "U" : +1,
    "L" : -1,
    "R" : +1,
    }

annotate_points = [
    ("$OA$",  0.981, 0.551, "RD"),
    ("$ALD$", 0.960, 0.433, "RD"),
    ("$ALD_{bump}$", 0.965,0.511, "RD"),
    ("$GI$",  0.965, 0.566, "LD"),
    ("$BSA$", 1.01,  0.611, "LU"),
    ("$HasA$",1.00,  0.715, "RU"),
    ("$HasA$",1.00,  0.715, "RU"),
    ("$PKS$", 0.977, 0.592, "LC"),
    ("$HypC$", 1.16, 0.605, "RC"),
    ("$C1015s$", 1.03, 1.09, "RU")
    ]

def plot_wyf_ratios(wyf_ratios_file):
    value_list = []
    with open(wyf_ratios_file) as fh:
        for line in fh:
            row = line[:-1].split(",")
            value_list.append([float(s) for s in row[1:]])
    R = np.array(value_list)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title("WYF Absorbance Ratios: $A_{258}/A_{280}$ vs. $A_{275}/A_{280}$", fontsize=16)

    ax.set_xlabel("$A_{275}/A_{280}$")
    ax.set_ylabel("$A_{258}/A_{280}$")

    ax.plot(R[:,0], R[:,1], "o", markersize=3)

    ax.set_xlim(0.1, 1.7)
    ax.set_ylim(0.1, 1.7)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = (xmax - xmin)*0.15
    dy = (ymax - ymin)*0.15

    def get_text_xy(x, y, loc):
        return x + SIGN_DICT[loc[0]]*dx, y + SIGN_DICT[loc[1]]*dy

    for name, r1, r2, loc in annotate_points:
        ax.plot(r1, r2, "o", color="red", markersize=3)
        tx, ty = get_text_xy(r1, r2, loc)
        color = "orange" if name.find("bump") >= 0 else None
        arrow_color = color if color == "orange" else "k"
        ax.annotate(name, xy=(r1, r2), xytext=(tx, ty), color=color, ha='center', arrowprops=dict(arrowstyle="->", color=arrow_color))

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    wyf_ratios_file = r"D:\TODO\20231218\wyf-ratios\wyf-ratios-save.csv"
    plot_wyf_ratios(wyf_ratios_file)
