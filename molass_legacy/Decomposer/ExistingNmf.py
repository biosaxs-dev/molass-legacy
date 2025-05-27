# coding: utf-8
"""

    ExistingNmf.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""
import copy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from ThreeDimUtils import compute_plane
from LmfitThreadSafe import minimize, Parameters
from ModeledData import ModeledData, simple_plot_3d
import molass_legacy.KekLib.DebugPlot as plt

class ExistingNmf:
    def __init__(self, M, module='nimfa'):
        if True:
            import nimfa
            tool = "Nimfa"
            if False:
                lsnmf = nimfa.Lsnmf(M, seed='random_vcol', rank=2, max_iter=100)
                lsnmf_fit = lsnmf()
                W = np.asarray(lsnmf_fit.basis())
                H = np.asarray(lsnmf_fit.coef())
            else:
                pmf = nimfa.Pmf(M, seed='random_vcol', rank=2, max_iter=100)
                pmf_fit = pmf()
                W = np.asarray(pmf_fit.basis())
                H = np.asarray(pmf_fit.coef())
        else:
            from sklearn.decomposition import NMF
            tool = "Scikit-Learn"
            model = NMF(n_components=2, init='random', random_state=0)
            W = model.fit_transform(M)
            H = model.components_

        self.tool = tool
        self.H = H
        self.W = W

class ExistingNmfDemo:
    def __init__(self, data, module='nimfa'):
        M = copy.deepcopy(data)
        M[M<0] = 0

        n_angles = M.shape[0]
        n_elutions = M.shape[1]

        if True:
            nmf = ExistingNmf(M, module=module)
            W = nmf.W
            H = nmf.H
            tool = nmf.tool
        else:
            import pymf
            nmf = pymf.NMF(M, num_bases=2)
            nmf.factorize()
            W = nmf.W
            H = nmf.H
            tool = 'PyMF'

        if True:
            from DataUtils import get_in_folder

            print('W=', W.shape)
            print('H=', H.shape)

            fig = plt.figure(figsize=(14,7))
            in_folder = get_in_folder()
            fig.suptitle("Non-negative Matrix Factorization using " + tool + " for " + in_folder, fontsize=20)

            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)

            simple_plot_3d(ax1, M, color='cyan', alpha=0.2)

            x = np.ones(n_elutions)*0.02
            y = np.arange(n_elutions)
            for k in range(H.shape[0]):
                z = H[k,:]
                ax1.plot(x, y, z)
                ax2.plot(y, z)

            fig.tight_layout()
            fig.subplots_adjust(top=0.9)
            plt.show()

def sklearn_nmf_spike():
    qsize = 600
    qvector = np.linspace(0.01, 0.6, qsize)
    n_elutions = 300
    pd = ModeledData(qvector, n_elutions)
    M = pd.get_data(debug=False)

    model = NMF(n_components=1, init='random', random_state=0)
    W = model.fit_transform(M)
    H = model.components_
    print('W=', W.shape)
    print('H=', H.shape)
    print('W@H=', (W@H).shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    simple_plot_3d(ax, M)
    x = np.ones(n_elutions)*0.02
    y = np.arange(n_elutions)
    z = H[0,:]
    ax.plot(x, y, z, color='orange')
    x = np.arange(qsize)
    y = np.ones(qsize)*150
    z = W[:,0]
    ax.plot(x, y, z, color='green')
    fig.tight_layout()
    plt.show()


def nimfa_spike():
    import nimfa

    qsize = 600
    qvector = np.linspace(0.01, 0.6, qsize)
    n_elutions = 300
    pd = ModeledData(qvector, n_elutions)
    M = pd.get_data(debug=False)

    lsnmf = nimfa.Lsnmf(M, seed='random_vcol', rank=1, max_iter=100)
    lsnmf_fit = lsnmf()

    W = np.asarray(lsnmf_fit.basis())
    H = np.asarray(lsnmf_fit.coef())

    print('W=', W.shape)
    print('H=', H.shape, type(H))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    simple_plot_3d(ax, M)
    x = np.ones(n_elutions)*0.02
    y = np.arange(n_elutions)
    z = H[0,:]
    scale = np.max(z)
    print('z=', z.shape)
    ax.plot(x, y, z/scale, color='orange')

    x = np.arange(qsize)
    y = np.ones(qsize)*150
    z = W[:,0]
    print('z=', z.shape)
    ax.plot(x, y, z*scale, color='green')
    fig.tight_layout()
    plt.show()

class MatrixFactorization:
    def __init__(self, sd):
        pass
