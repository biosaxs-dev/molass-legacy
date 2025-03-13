"""
    Optimizer.FvScoreConverter.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np

def sigmoid(x, L ,x0, k, b):
    if not np.isscalar(x):
        # see https://stackoverflow.com/questions/59297543/why-do-i-get-the-loop-of-ufunc-does-not-support-argument-0-of-type-int-error-f
        x = x.astype(float)
    return L/(1 + np.exp(-k*(x-x0))) + b

def convert_score(value):
    return sigmoid(value, -200, 0, 1.5, 100)

convert_scores = np.vectorize(convert_score)

def invert_score(sv):
    from scipy.optimize import root

    assert np.isscalar(sv)

    def fun(x):
        return [convert_score(x)[0] - sv]

    sol = root(fun, (0, ) )
    return sol.x[0]

def compute_fv_adjustment(sv, target_sv):
    return invert_score(target_sv) - invert_score(sv)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn
    seaborn.set()
    x = np.linspace(-5, 5)
    print(x.shape)
    y = convert_score(x)

    y_ = y[[20, 30]]
    x_ = [invert_score(v) for v in y_]

    print("compute_fv_adjustment: ", compute_fv_adjustment(-75, 50))

    fig, ax = plt.subplots()
    ax.set_title("FV Conversion to Scale of Hundred (SV)", fontsize=20)
    ax.set_xlabel("FV", fontsize=16)
    ax.set_ylabel("SV", fontsize=16)
    ax.plot(x, y, label="conversion")
    ax.plot(x_, y_, "o", label="inversion verification")
    ax.legend()
    fig.tight_layout()
    plt.show()
