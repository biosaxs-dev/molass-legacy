# coding: utf-8
"""

    ModeledData.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""
import numpy as np
from molass_legacy.Models.ElutionCurveModels import emg
from DataModels import GuinierPorod
import molass_legacy.KekLib.DebugPlot as plt
from MatrixData import simple_plot_3d

class ModeledData:
    def __init__(self, ivector, n_elutions, rg_list=[25], d_list=[1], max_height=1, noise=0, h_list=None, mu_list=None, sigma_list=None, tau_list=None):
        assert len(rg_list) == len(d_list)
        self.num_components = len(rg_list)
        self.i = ivector
        self.j = np.arange(n_elutions)
        self.rg_list = rg_list
        self.d_list = d_list
        self.max_height = max_height
        self.num_peaks = len(rg_list)
        self.make_C_matrix(h_list, mu_list, sigma_list, tau_list)
        self.make_P_matrix()

    def make_C_matrix(self, h_list, mu_list, sigma_list, tau_list):
        if mu_list is None:
            self.top_i = np.linspace(self.j[0], self.j[-1], self.num_peaks+2)[1:-1]
        else:
            self.top_i = mu_list
        C_list = []
        self.e_params = []
        for k in range(self.num_peaks):
            h = 1 if h_list is None else h_list[k]
            mu = self.top_i[k]
            sigma = len(self.j)/10 if sigma_list is None else sigma_list[k]
            tau = 0 if tau_list is None else tau_list[k]
            C_list.append(emg(self.j, h=h, mu=mu, sigma=sigma, tau=tau))
            self.e_params.append((h, mu, sigma, tau))
        self.C = np.array(C_list)

    def make_P_matrix(self):
        h = self.max_height
        d = 1
        P_list = []
        for rg, d in zip(self.rg_list, self.d_list):
            gp_model = GuinierPorod(h, rg, d)
            P_list.append(gp_model(self.i))
        self.P = np.array(P_list).T

    def get_data(self, noise=0.01, debug=False):
        M = np.dot(self.P, self.C)
        if noise > 0:
            M_ = M/np.max(M)
            M *= (np.random.normal(0, 1, M.shape)*noise/M_ + 1)

        if debug:
            # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7), subplot_kw={'projection': '3d'})
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            simple_plot_3d(ax, M, x=self.i)
            fig.tight_layout()
            plt.show()

        return M

    def plot_components(self):
        n = self.num_components
        fig = plt.figure(figsize=(21, 6))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133, projection='3d')
        for i in range(n):
            ax1.plot(self.i, self.P[:,i])
            ax2.plot(self.C[i,:])
        M = self.get_data()
        simple_plot_3d(ax3, M, x=self.i)
        fig.tight_layout()
        plt.show()
