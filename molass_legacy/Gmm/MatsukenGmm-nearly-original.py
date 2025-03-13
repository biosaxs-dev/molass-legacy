# coding: utf-8
"""
    MatsukenGmm.py.

    adapted from:
        EMアルゴリズム徹底解説
        https://qiita.com/kenmatsu4/items/59ea3e5dfa3d4c161efb

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import numpy as np
import numpy.random as rd
import scipy as sp
from scipy import stats as st
from collections import Counter

import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import matplotlib.animation as ani
plt.rcParams["patch.force_edgecolor"] = True
#rc('text', usetex=True)
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]

c = ['r', 'g', 'b']
n = [200, 150, 150]
N = np.sum(n)
D = 2
K = 3

def calc_likelihood(data, mu, sigma, pi, K):
    likelihood = np.zeros((np.sum(n), 3))
    for k in range(K):
        likelihood[:, k] = [pi[k]*st.multivariate_normal.pdf(d, mu[k], sigma[k]) for d in data]
    return likelihood

def calc_prob_gmm(data, mu, sigma, pi, K):
    return [[pi[k]*st.multivariate_normal.pdf(d, mu[k], sigma[k]) for k in range(K)] for d in data]

def print_gmm_contour(mu, sigma, pi, K):
    # display predicted scores by the model as a contour plot
    X, Y = np.meshgrid(np.linspace(min_x, max_x), np.linspace(min_y, max_y))
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = np.sum(np.asanyarray(calc_prob_gmm(XX, mu, sigma, pi, K)), axis=1)
    Z = Z.reshape(X.shape)
    CS = plt.contour(X, Y, Z, alpha=0.2, zorder=-100)
    
    plt.title('pdf contour of a GMM')

class GMM:
    def __init__(self, k, max_iter):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        pass

def generate_demo_data():
    global data

    seed = 77

    mu_true = np.asanyarray(
         [[0.2, 0.5],
          [1.2, 0.5],
          [2.0, 0.5]])

    D = mu_true.shape[1]

    sigma_true = np.asanyarray(
            [ [[0.1,  0.085],[ 0.085, 0.1]],
              [[0.1, -0.085],[-0.085, 0.1]],
              [[0.1,  0.085],[ 0.085, 0.1]]
            ])

    rd.seed(seed)
    org_data = None #np.empty((np.sum(n), 3))
    for i in range(3):
        print("check: ", i, mu_true[i], sigma_true[i], np.linalg.det(sigma_true[i]))
        #org_data.append(st.multivariate_normal.rvs(mean=mu[i], cov=sigma[i], size=n[i]))
        if org_data is None:
            org_data = np.c_[st.multivariate_normal.rvs(mean=mu_true[i], cov=sigma_true[i], size=n[i]), np.ones(n[i])*i]
        else:
            org_data = np.r_[org_data, np.c_[st.multivariate_normal.rvs(mean=mu_true[i], 
                                                                        cov=sigma_true[i], 
                                                                        size=n[i]), np.ones(n[i])*i]]
            
    # plot generated data        

    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.scatter(org_data[org_data[:,2]==i][:,0], org_data[org_data[:,2]==i][:,1], s=30, c=c[i], alpha=0.5)
        
    # drop true cluster label
    data = org_data[:,0:2].copy()
    return data

def initilize():
    global mu, sigma, pi
    global max_x, min_x, max_y, min_y

    # initialize pi
    pi = np.zeros(K)
    for k in range(K):
        if k == K-1:
            pi[k] = 1 - np.sum(pi)
        else:
            pi[k] = 1/K
    print('init pi:', pi)

    # initialize mu
    max_x, min_x = np.max(data[:,0]), np.min(data[:,0])
    max_y, min_y = np.max(data[:,1]), np.min(data[:,1])
    mu = np.c_[rd.uniform(low=min_x, high=max_x, size=K), rd.uniform(low=min_y, high=max_y, size=K) ]
    print('init mu:\n', mu)

    # visualize for check
    #plt.figure(figsize=(12,8))
    plt.figure(figsize=(12, 5))
    plt.scatter(data[:,0], data[:,1], s=30, c='gray', alpha=0.5, marker="+")

    for i in range(3):
        plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o')
        
    plt.show()

    # initialize sigma
    sigma = np.asanyarray(
            [ [[0.1,  0],[ 0, 0.1]],
              [[0.1,  0],[ 0, 0.1]],
              [[0.1,  0],[ 0, 0.1]] ])

    # calculate likelihood
    likelihood = calc_likelihood(data, mu, sigma, pi, K)
    print('initial sum of log likelihood:', np.sum(np.log(likelihood)))

    print('pi:\n', pi)
    print('mu:\n', mu)
    print('sigma:\n', sigma)

    plt.title('initial state')

def animate(nframe):
    global mu, sigma, pi

    print('nframe:', nframe)
    plt.clf()
    
    if nframe <= 3:
        print('initial state')
        plt.scatter(data[:,0], data[:,1], s=30, c='gray', alpha=0.5, marker="+")
        for i in range(3):
            plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
        print_gmm_contour(mu, sigma, pi, K)
        plt.title('initial state')
        return

    # E step ========================================================================
    # calculate responsibility(負担率)
    likelihood = calc_likelihood(data, mu, sigma, pi, K)
    #gamma = np.apply_along_axis(lambda x: [xx/np.sum(x) for xx in x] , 1, likelihood)
    gamma = (likelihood.T/np.sum(likelihood, axis=1)).T
    N_k = [np.sum(gamma[:,k]) for k in range(K)]

    # M step ========================================================================

    # caluculate pi
    pi =  N_k/N

    # calculate mu
    tmp_mu = np.zeros((K, D))

    for k in range(K):
        for i in range(len(data)):
            tmp_mu[k] += gamma[i, k]*data[i]
        tmp_mu[k] = tmp_mu[k]/N_k[k]
        #print('updated mu[{}]:\n'.format(k) , tmp_mu[k])
    mu_prev = mu.copy()
    mu = tmp_mu.copy()
    #print('updated mu:\n', mu)

    # calculate sigma
    tmp_sigma = np.zeros((K, D, D))

    for k in range(K):
        tmp_sigma[k] = np.zeros((D, D))
        for i in range(N):
            tmp = np.asanyarray(data[i]-mu[k])[:,np.newaxis]
            tmp_sigma[k] += gamma[i, k]*np.dot(tmp, tmp.T)
        tmp_sigma[k] = tmp_sigma[k]/N_k[k]

        #print('updated sigma[{}]:\n'.format(k) , tmp_sigma[k])
    sigma = tmp_sigma.copy()

    # calculate likelihood
    prev_likelihood = likelihood
    likelihood = calc_likelihood(data, mu, sigma, pi, K)
    
    prev_sum_log_likelihood = np.sum(np.log(prev_likelihood))
    sum_log_likelihood = np.sum(np.log(likelihood))
    diff = prev_sum_log_likelihood - sum_log_likelihood
    
    print('sum of log likelihood:', sum_log_likelihood)
    print('diff:', diff)

    print('pi:', pi)
    print('mu:', mu)
    print('sigma:', sigma)

    # visualize
    #plt.figure(figsize=(12,8))
    for i in range(N):
        plt.scatter(data[i,0], data[i,1], s=30, c=gamma[i], alpha=0.5, marker="+")

    for i in range(K):
        ax = plt.axes()
        ax.arrow(mu_prev[i, 0], mu_prev[i, 1], mu[i, 0]-mu_prev[i, 0], mu[i, 1]-mu_prev[i, 1],
                  lw=0.8, head_width=0.02, head_length=0.02, fc='k', ec='k')
        plt.scatter([mu_prev[i, 0]], [mu_prev[i, 1]], c=c[i], marker='o', alpha=0.8)
        plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
        #plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o')
    plt.title("step:{}".format(nframe))
    
    print_gmm_contour(mu, sigma, pi, K)
    # plt.show()
    
    if np.abs(diff) < 0.0001:
        plt.title('likelihood is converged.')
    else:
        plt.title("iter:{}".format(nframe-3))

def blog_demo():
    generate_demo_data()
    plt.show()

    initilize()
    plt.show()

    fig = plt.figure(figsize=(12,5))
    anim = ani.FuncAnimation(fig, animate, frames=48)
    # anim.save('gmm_anim.gif', writer='imagemagick', fps=3, dpi=128)
    plt.show()
