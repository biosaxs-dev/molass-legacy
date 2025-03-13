# DPHEM library which is compatible with sklearn.mixture.GaussianMixture 
# ---
# If you use this code, please cite our paper:
#   Lei Yu, Tianyu Yang, and Antoni B. Chan. "Density Preserving Hierarchical EM Algorithm: Simplifying 
#   Gaussian Mixture Models for Approximate Inference." TPAMI 2019
# ---
# Copyright (c) 2019-04-30
# Lei Yu, Antoni B. Chan
# Department of Computer Science
# City University of Hong Kong 
#
# v0.1


import numpy as np
from sklearn.mixture import GaussianMixture
from .pylib_dphem import *
import warnings


def gmm_dphem_learn(basegmm, tagk, opt):
    # extract the GMM parameters to implement our algorithm

    if 'showplot' not in opt:
        opt['showplot'] = 0

    if 'LLterm' not in opt:
        opt['LLterm'] = 1e-5

    if 'cvminreg' not in opt:
        opt['cvminreg'] = 1e-6

    if 'verbose' not in opt:
        opt['verbose'] = 1

    if 'pimode' not in opt:
        opt['pimode'] = 'preserve'

    if 'maxiter' not in opt:
        opt['maxiter'] = 1000

    if 'Nvs' not in opt:
        opt['Nvs'] = 10*basegmm.n_components

    maxiter = opt['maxiter']
    showplot = opt['showplot']
    ncomp_base,dim = np.shape(basegmm.means_)  
                                    

    # Initialization of reducedgmm
    # 0. fixed initgmm with some given GMM
    # 1. randomly select tagk components from the basegmm
    # 2. weighted kmeans of base mixture components
    # 3. splitting from one component
    if opt['initmode'] == 'fix':
        if 'initgmm' not in opt or not isinstance(opt['initgmm'],GaussianMixture):
            raise ValueError('Init GMM with GaussianMixture type should be given.')
        else:
            gmm = opt['initgmm']

        # with an init reducedgmm, we'll iterate with EM to optimize the reducedgmm
        num_iter = 0
        dataLLall = []
        dataLL = 0.0

        while True:
            dataLLold = dataLL
            LL,LLcomp,Y,LB,entropyLB = E_step(basegmm,gmm,opt)

            dataLL = LB

            dataLLall.append(dataLL)

            do_break = False

            if num_iter > 0:
                dLL = dataLL - dataLLold
                pLL = np.fabs(dLL/dataLL)
                if opt['verbose']:
                    print(('iter = {0}; LL = {1}; dLL = {2}; pLL = {3} \n'.
                        format(num_iter, dataLL, dLL, pLL)))

                if dLL < 0:
                    print(('\nnum_iter = {0}\n'.format(num_iter)))
                    warnings.warn('LL change is negative.')

                if pLL < opt['LLterm']:
                    do_break = True

            if num_iter >= maxiter:
                warnings.warn('Max iterations reached.')
                do_break = True

            if do_break:
                break

            gmm = M_step(gmm, basegmm, Y, opt)
            num_iter += 1

        gpost = Y
        info = {'dataLL': dataLL,
                'dataLLall': np.asarray(dataLLall),
                'opt': opt}
        diffel = entropyLB - LB

        return gmm, gpost, info, diffel


    if opt['initmode'] == 'random':
        # randomly select tagk components from the basegmm as init reducedgmm
        # only means are the same but weights and covariances are not
        initgmm = GaussianMixture(n_components=tagk,covariance_type=basegmm.covariance_type)
        weights = np.divide(np.ones(tagk),tagk)
        means = []
        covariances = []

        cvall = 0
        for j in range(ncomp_base):
            if (basegmm.covariance_type == 'spherical' or 
               basegmm.covariance_type == 'diag'):
               cvall += np.mean(basegmm.covariances_[j])
            elif basegmm.covariance_type == 'full':
                cvall += np.mean(np.diagonal(basegmm.covariances_[j]))
            else:
                raise ValueError('Undefined covariance type.')

        cvall = float(cvall)/ncomp_base

        foo = np.random.permutation(list(range(ncomp_base)))

        for j in range(tagk):
            means.append(basegmm.means_[foo[j]])
            if basegmm.covariance_type == 'spherical':
                covariances.append(cvall)
            elif basegmm.covariance_type == 'diag':
                covariances.append(cvall*np.ones(dim))
            elif basegmm.covariance_type == 'full':
                covariances.append(cvall*np.eye(dim))
            else:
                raise ValueError('Undefined covariance type.')

        initgmm.weights_ = weights
        initgmm.means_   = np.asarray(means)
        initgmm.covariances_ = np.asarray(covariances)

        opt['initmode'] = 'fix'
        opt['initgmm']  = initgmm
        reducedgmm, gpost, info, diffel = gmm_dphem_learn(basegmm, tagk, opt)

        return reducedgmm, gpost, info, diffel


    if opt['initmode'] == 'wtkmeans':
        # Initialize the reducedgmm with weighted kmeans of the basegmm components
        initgmm = GaussianMixture(n_components=tagk,covariance_type=basegmm.covariance_type)
        weights = np.divide(np.ones(tagk),tagk)
        covariances = []  # use the average covariances of basegmm components

        # randomly select tagk components as the cluster centers
        foo = np.random.permutation(list(range(ncomp_base)))
        init_cluster_center = []
        for i in range(tagk):
            init_cluster_center.append(basegmm.means_[foo[i]])

        init_cluster_center = np.asarray(init_cluster_center).T
        cluster_label,cluster_center = my_weighted_kmeans(basegmm.means_.T,
             basegmm.weights_, init_cluster_center, tagk, 100)
        means = cluster_center.T

        cvall = 0
        for j in range(ncomp_base):
            if (basegmm.covariance_type == 'spherical' or 
               basegmm.covariance_type == 'diag'):
               cvall += np.mean(basegmm.covariances_[j])
            elif basegmm.covariance_type == 'full':
                cvall += np.mean(np.diagonal(basegmm.covariances_[j]))
            else:
                raise ValueError('Undefined covariance type.')

        cvall = float(cvall)/ncomp_base

        for j in range(tagk):
            if basegmm.covariance_type == 'spherical':
                covariances.append(cvall)
            elif basegmm.covariance_type == 'diag':
                covariances.append(cvall*np.ones(dim))
            elif basegmm.covariance_type == 'full':
                covariances.append(cvall*np.eye(dim))
            else:
                raise ValueError('Undefined covariance type.')

        initgmm.weights_ = weights
        initgmm.means_   = means
        initgmm.covariances_ = covariances

        opt['initmode'] = 'fix'
        opt['initgmm']  = initgmm
        reducedgmm,gpost,info,diffel = gmm_dphem_learn(basegmm, tagk, opt)

        return reducedgmm,gpost,info,diffel


    


