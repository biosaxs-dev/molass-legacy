"""
    Optimizer.FvSynthesizerHistory.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

score_names = [
    "XR_2D_fitting",
    "XR_LRF_residual",
    "UV_2D_fitting",
    "UV_LRF_residual",
    "Guinier_deviation",
    "Kratky_smoothness",
    "SEC_conformance",
    ]

weights_history = [
    [0.11111111, 0.22222222, 0.11111111, 0.22222222, 0.11111111, 0.11111111, 0.11111111],
    [0.2, 0.1, 0.2, 0.1, 0.133, 0.133, 0.133],
    [0.3, 0.1, 0.3, 0.1, 0.067, 0.067, 0.066],
    [0.35, 0.1, 0.35, 0.1, 0.033, 0.033, 0.034],
    ]

def plot_history():
    ncols = len(weights_history)
    xpos = np.arange(len(score_names))
    colors = ["C1", "C0", "C1", "C0", "C0", "C0", "C0"]
    fig, axes = plt.subplots(ncols=ncols, figsize=(12,4))
    fig.suptitle("Tried Weights Variations")
    trial = 0
    for ax, weights in zip(axes, weights_history):
        trial += 1
        ax.set_title("Trial %d" % trial)
        ax.bar(score_names, weights, color=colors)
        ax.set_xticks(xpos, score_names, rotation=90)
        ax.set_ylim(0, 0.5)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    import seaborn
    seaborn.set()
    plot_history()
