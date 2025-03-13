"""
    DiniSigmoid.py

    from:

        Normalized tunable sigmoid functions
        https://dinodini.wordpress.com/2010/04/05/normalized-tunable-sigmoid-functions/

        Normalized Tunable Sigmoid Function
        https://dhemery.github.io/DHE-Modules/technical/sigmoid/#function

"""
import numpy as np
from pynverse import inversefunc

def dini_sigmoid(x, k):
    return x*(1 - k)/(k - 2*k*np.abs(x) + 1)

def dini_sigmoid_inv(y, k):
    def func(y):
        return dini_sigmoid(y, k)
    return inversefunc(func, y_values=y)

def demo():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    x = np.linspace(-1, 1, 100)
    for k in np.linspace(-0.95, 0.95, 9):
        y = dini_sigmoid(x, k)
        ax.plot(x, y, label="k=%.3g" % k)

    k = -0.95
    y_ = 0.9
    x_ = dini_sigmoid_inv(y_, k)
    ax.plot(x_, y_, "o", color="red")

    ax.legend()
    plt.show()

if __name__ == '__main__':
    demo()
