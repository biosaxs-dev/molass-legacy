# coding: utf-8
"""
    EghBySampling.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from numpy.random import uniform
from scipy.interpolate import interp1d
from scipy.optimize import root
import matplotlib.pyplot as mplt
import molass_legacy.KekLib.DebugPlot as plt
from .EghMixture import egh_pdf, e1, e2, e3

class EghSampler:
    def __init__(self, tR=100, sigma=10, tau=10, start=0, stop=200):
        x = np.arange(start, stop, 0.5)
        y = egh_pdf(x, tR, sigma, tau)
        if False:
            plt.plot(x, y)
            plt.show()

        cdf_y = np.cumsum(y)
        cdf_y = cdf_y/cdf_y.max()
        if False:
            plt.plot(cdf_y, x)
            plt.show()
        if tau >= 0:
            self.inv_cdf = interp1d(cdf_y,x)
        else:
            pos_y = cdf_y > 0
            self.inv_cdf = interp1d(cdf_y[pos_y],x[pos_y])

    def __call__(self, num_samples):
        return  self.inv_cdf(np.random.rand(num_samples))

def moments_proof(tau_list):
    N = 1000000

    m1_list = []
    m2_list = []
    m3_list = []
    k3_list = []
    sigma = 10
    th_list = []
    tR = 100

    for tau in tau_list:
        print('tau=', tau)
        sampler = EghSampler(tR=tR, sigma=sigma, tau=tau)
        Z = sampler(N)
        th = np.arctan2(abs(tau), sigma)
        print('th=', th)
        th_list.append(th)
        M1 = np.mean(Z)
        m1 = tR + tau*e1(th)
        print('M1=', M1, m1)
        m1_list.append((M1, m1))
        M2 = np.sum((Z - M1)**2)/N
        m2 = (sigma**2 + sigma*abs(tau) + tau**2)*e2(th)
        print('M2=', M2, m2)
        m2_list.append((M2, m2))

        M3 = np.sum((Z - M1)**3)/N
        k3 = tau*(3*sigma**2 + 4*sigma*abs(tau) + 4*tau**2)
        m3 = k3*e3(th)
        print('M3=', M3, m3)
        k3_list.append(k3)
        m3_list.append((M3, m3))

    m1_array = np.array(m1_list)
    m2_array = np.array(m2_list)
    m3_array = np.array(m3_list)
    k3_array = np.array(k3_list)
    th_array = np.array(th_list)

    T = np.array([ th_array**k for k in range(7) ])
    Y = (m3_array[:,0]/k3_array).reshape((1, len(m3_array)))
    A = np.dot(Y, np.linalg.pinv(T))
    print('A=', A)

    x = tau_list

    plt.push()
    fig = plt.figure()
    ax = fig.gca()
    axt = ax.twinx()
    ax.set_title("Proof of EGH Moment Formulas with %d Sampling" % N, fontsize=16)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Moments')
    ax.plot(x, m1_array[:,0], color='red', label=r'$M_1$ (raw)')
    ax.plot(x, m1_array[:,1], ':', color='red', label=r'$t_R + \tau\epsilon_1$')
    ax.plot(x, m2_array[:,0], color='green', label=r'$M_2$ (central)')
    ax.plot(x, m2_array[:,1], ':', color='green', label=r'$(\sigma^2 + \sigma|\tau| + \tau^2)\epsilon_2$')
    ax.plot(x, m3_array[:,0], color='blue', label=r'$M_3$ (central)')
    ax.plot(x, m3_array[:,1], ':', color='blue', label=r'$\tau(3\sigma^2 + 4\sigma|\tau| + 4\tau^2)\epsilon_3$')
    ax.legend()

    axt.plot(x, th_array, 'o', color='orange', label=r'$\Theta$')
    axt.legend(loc='center right')

    fig.tight_layout()
    plt.show()
    plt.pop()
