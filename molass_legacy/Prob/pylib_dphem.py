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

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture



def do_split(ingmm,tagk,mytarget=None):
    '''gmm = do_split(ingmm,tagk)
       split ingmm to tagk components'''

    gmm = ingmm

    origk = ingmm.n_components
    if tagk > 2*origk:
        raise ValueError('Split target K too large.')

    myeigs = np.zeros(origk)
    for jj in range(origk):
        if not np.size(ingmm.covariances_[jj]):
            myeigs[jj] = -1
        else:
            if (ingmm.covariance_type == 'spherical' or
               ingmm.covariance_type == 'diag'):
               myeigs[jj] = max(ingmm.covariances_[jj])
            elif ingmm.covariance_type == 'full':
                D,V = np.linalg.eig(ingmm.covariances_[jj])
                myeigs[jj] = max(D)
            else:
                raise ValueError('Not Supported.')

    if mytarget is not None:
        dotarget = True
    else:
        dotarget = False

    while ingmm.n_components < tagk or dotarget:
        foo = max(myeigs)
        ind1 = myeigs.tolist().index(foo)

        # decide the target component
        if dotarget:
            ind2 = mytarget
            dotarget = False
        else:
            ind2 = ingmm.n_components
            gmm.n_components = ingmm.n_components + 1

        # make a new components
        tmp = ingmm.weights_[ind1]
        gmm.weights_[ind1] = tmp/2.0

        tmpcv = ingmm.covariances_[ind1]
        covariances = ingmm.covariances_.tolist().append(tmpcv)
        if ind2 in range(origk):
            gmm.weights_[ind2] = tmp/2.0
            gmm.covariances_[ind2] = tmpcv
        else:
            gmm.weights_ = np.asarray(np.append(ingmm.weights_,tmp/2.0))
            gmm.covariances_ = np.asarray(covariances)

        tmpmu = ingmm.means_[ind1]

        # perturb mu in maximum variance direction by std/2
        if ingmm.covariance_type == 'spherical':
            d = len(ingmm.means_[0])
            pert = np.sqrt(tmpcv)/2.0
            tmpd = pert * np.ones(d)
        elif ingmm.covariance_type == 'diag':
            foo2 = max(tmpcv)
            pert = np.sqrt(foo2)/2.0
            tmpd = np.zeros(len(tmpmu))
            tmpd[tmpcv==foo2] = pert
        elif ingmm.covariance_type == 'full':
            tD,tV = np.linalg.eig(tmpcv)
            tfoo = max(tD)
            indt = tD.tolist().index(tfoo)
            tmpd = tV[:,indt]*(np.sqrt(tfoo)/2.0)
        else:
            raise ValueError('Not Supported.')

        gmm.means_[ind1] = tmpmu - tmpd
        if ind2 in range(origk):
            gmm.means_[ind2] = tmpmu + tmpd
        else:
            means = gmm.means_.tolist().append(tmpmu+tmpd)
            gmm.means_ = np.asarray(means_)

        myeigs[ind1] = -1

    return gmm


def E_step(ingmm, gmm, opt):
    Nvs = opt['Nvs']
    ncomp_base,dim = np.shape(ingmm.means_)
    
    muLL, muLLcomp, mupost = gmm_ll(ingmm.means_, gmm)
    muLLcomp = muLLcomp - np.log(gmm.weights_)

    muLL1, muLLcomp1, mupost1 = gmm_ll(ingmm.means_, ingmm)
    muLLcomp1 = muLLcomp1 - np.log(ingmm.weights_)

    trLLcomp1 = 4.0*np.ones(np.shape(muLLcomp1))

    trLLcomp = np.zeros(np.shape(muLLcomp))
    for j in range(gmm.n_components):
        if gmm.covariance_type == 'spherical':
            for i in range(ingmm.n_components):
                trLLcomp[i,j] = dim*ingmm.covariances_[i]/gmm.covariances_[j]
        elif gmm.covariance_type == 'diag':
            for i in range(ingmm.n_components):
                tmptr = ingmm.covariances_[i]/gmm.covariances_[j]
                trLLcomp[i,j] = np.sum(tmptr)
        elif gmm.covariance_type == 'full':
            tmpcv = np.linalg.cholesky(gmm.covariances_[j])
            for i in range(ingmm.n_components):
                SigmaInvSigma = solve_chol(tmpcv, ingmm.covariances_[i])
                trLLcomp[i,j] = np.trace(SigmaInvSigma)

    if opt['pimode'] == 'original':
        LLcomp = (Nvs*ingmm.weights_*(muLLcomp - 0.5*trLLcomp).T).T + np.log(gmm.weights_) 
        LLcomp1 =(Nvs*ingmm.weights_*(muLLcomp1- 0.5*trLLcomp1).T).T + np.log(ingmm.weights_)
        LL  = logtrick2(LLcomp)
        LL1 = logtrick2(LLcomp1)

        logY  = LLcomp - LL
        logY1 = LLcomp1 - LL1

        Y  = np.exp(logY)
        Y1 = np.exp(logY1)

        LB = np.sum(Y*(LLcomp - logY))
        entropyLB = np.sum(Y1*(LLcomp1-Y1))

    elif opt['pimode'] == 'preserve':
        LLcomp = Nvs*(muLLcomp - 0.5*trLLcomp) + np.log(gmm.weights_)
        LLcomp1 = Nvs*(muLLcomp1- 0.5*trLLcomp1) + np.log(ingmm.weights_)
        LL  = logtrick2(LLcomp)
        LL1 = logtrick2(LLcomp1)

        logY  = LLcomp - LL
        logY1 = LLcomp1 - LL1

        Y  = np.exp(logY)
        Y1 = np.exp(logY1)

        tmplb = (ingmm.weights_*Y.T).T*(LLcomp - logY)
        LB = np.sum(tmplb)
        entropyLB = np.sum(ingmm.weights_*LLcomp1.T)
    else:
        raise ValueError('Not Supported.')

    return LL,LLcomp,Y,LB,entropyLB


def M_step(gmm, ingmm, Y, opt):
    ''' Update parameters of gmm
    gmm = M_step(gmm, ingmm, Y, opt)'''

    ncomp_base, dim = np.shape(ingmm.means_)
    pi_new = ingmm.weights_.reshape((ncomp_base,1))

    if opt['pimode'] == 'original':
        gmm.weights_ = np.sum(Y,axis=0)/ncomp_base
    elif opt['pimode'] == 'preserve':
        tmpz = pi_new*Y
        gmm.weights_ = np.sum(tmpz,axis=0)/np.sum(tmpz)
    else:
        raise ValueError('Not Supported.')

    W = Y*pi_new
    W = W/np.sum(W,axis=0)

    mumtx = ingmm.means_
    means = []
    covariances = []
    for j in range(gmm.n_components):
        if np.isinf(W[:,j]).any() or np.isnan(W[:,j]).any():
            mu = np.asarray([])
            cv = np.asarray([])
        else:
            Wj = W[:,j]
            mu = np.zeros(dim)
            for c in range(ncomp_base):
                mu = mu + Wj[c]*ingmm.means_[c]

            Xmu = mumtx - mu

            if gmm.covariance_type == 'spherical':
                cvm = np.zeros(dim)
                for c in range(ncomp_base):
                    cvm = cvm + Wj[c]*(ingmm.covariances_[c] + Xmu[c,:]*Xmu[c,:])
                cv = np.sum(cvm)/fload(dim)
                cv = max(cv, opt['cvminreg'])

            elif gmm.covariance_type == 'diag':
                cv = np.zeros(dim)
                for c in range(ncomp_base):
                    cv = cv + Wj[c]*(ingmm.covariances_[c] + Xmu[c,:]*Xmu[c,:])
                cv[cv<opt['cvminreg']] = opt['cvminreg']

            elif gmm.covariance_type == 'full':
                cvtmp = np.zeros_like(ingmm.covariances_[0])
                for jj in range(len(Wj)):
                    cvtmp += Wj[jj] * ingmm.covariances_[jj]

                cv = np.dot((Wj*Xmu.T),Xmu) + cvtmp

                if opt['cvminreg'] > 0:
                    D, V = np.linalg.eig(cv)
                    D[D<opt['cvminreg']] = opt['cvminreg']
                    cv = np.dot(np.dot(V, np.diag(D)), V.T)
            else:
                raise ValueError('Not Supported.')

        means.append(mu)
        covariances.append(cv)

    gmm.means_ = np.asarray(means)
    gmm.covariances_ = np.asarray(covariances)


    return gmm






def logtrick2(LLcomp):
    '''LLcomp: N x K, each column is the log-likelihoods of ingmm.mu on a component 
       of gmm = log(a), log(b) ... log(k)
       output a row vector N x 1 = log(sum(a+b+...k)) 
    '''
    N = np.shape(LLcomp)[0]
    mv = np.max(LLcomp, axis=1)
    mv = mv.reshape((N,1))
    cterm = np.sum(np.exp(LLcomp-mv), axis=1)
    s = mv + np.log(cterm.reshape((N,1)))
    return s





def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
        
def plot_gmm(gmm, X, label=False, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='rainbow', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')
    
    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)



def gmm_plot1d(gmm, color, opt='cpg', dim=0):
    """plot a gmm in 1d
    INPUTS:
      opt = 'c' -- plot component
          = 'p' -- plot priors
          = 'g' -- plot gmm
      dim = dimension to use [default = 0]"""

    # # select dimension (if necessary)
    # if len(gmm.means_[0]) != 1:
    #     for t in range(len(gmm.means_)):
    #         gmm.means_[t] = gmm.means_[t][dim]
    #         if gmm.covariance_type == 'diag':
    #             gmm.covariances_[t] = gmm.covariances_[t][dim]
    #         elif gmm.covariance_type == 'full':
    #             gmm.covariances_[t] = gmm.covariances_[t][dim, dim]

    stdext = 3
    xlo = []
    xhi = []
    for j in range(gmm.n_components):
        xlo.append(gmm.means_[j][0] - np.sqrt(gmm.covariances_[j][0][0]) * stdext)
        xhi.append(gmm.means_[j][0] + np.sqrt(gmm.covariances_[j][0][0]) * stdext)

    xlo = min(xlo)
    xhi = max(xhi)

    x = np.linspace(xlo, xhi, 200)
    x = x.reshape(-1,1)

    LL, LLcomp, post = gmm_ll(x, gmm)
    # print LL
    for j in range(gmm.n_components):
        if 'c' in opt:
            plt.plot(x, np.exp(LLcomp[:, j]), color+':')
        # if 'p' in opt:
            # text display the priors
        if 'g' in opt:
            plt.plot(x, np.exp(LL), color+'-')



def gmm_ll(X, gmm):
    """ gmm_ll -- log-likelihood of GMM

        USAGE: 
        LL, LLcomp, post = gmm_ll(X, gmm)

        INPUTS:
          X -- 2d array(matrix) with each row the mean of a GMM component
          gmm -- GMM model
        
        OUTPUTS:
          LL     -- log-likelihood of X        [n x 1]
                    = log(gmm(X))
          LLcomp -- component log-likelihoods  [n x K]
          post   -- posterior probabilities    [n x K]

        if bkgndclass is used, LLcomp and post are [n x (K+1)], where the 
        last column is the log-likelihood and posterior in the background class"""

    K = gmm.n_components
    N,d = np.shape(X)

    LLcomp = np.zeros((N, K))
    # print 'gmm = {0}\n'.format(gmm)
    for c in range(K):
        tmpLL = logComponent(gmm, X, c, d, N)
        LLcomp[:, c] = tmpLL

    LL = logtrick2(LLcomp)
    # LL should be a column vector
    # return LL, LLcomp

    post = np.exp(LLcomp - LL)
    return LL, LLcomp, post


def logComponent(gmm, X, c, d, N):
    # return the log-likelihood of component c
    # myLLcomp = logComponent(gmm, X, c, d, N)
    # myLLcomp should be a row vector

    mu = gmm.means_[c]
    # print 'mu = {0}\n'.format(mu)
    cv = gmm.covariances_[c]

    tempx = X - mu

    if gmm.covariance_type == 'spherical':
        g = np.sum(tempx * tempx)/cv
        ld = d * np.log(cv)

    elif gmm.covariance_type == 'diag':
        # if cv is diagonal matrix, it's stored as a 1d array in a row
        tmp = tempx/np.sqrt(cv)
        g = np.sum(tmp * tmp, axis=1)
        ld = np.sum(np.log(cv))

    elif gmm.covariance_type == 'full':
        L = np.linalg.cholesky(cv)
        InvCVX = solve_chol(L, tempx.T)
        g = np.sum(InvCVX.T * tempx, axis=1)
        # ld = np.linalg.slogdet(cv)[1]  # this function returns a tuple with two elements: sign and value
        ld = logdet_chol(L)
    else:
        raise ValueError('Not Supported.')

    # print 'gmm.K = {0},  c = {1}, gmm.pi = {2}'.format(gmm['K'], c, gmm['pi'])
    # if gmm['pi'][c]:

    myLLcomp = -0.5*g - (d/2.0)*np.log(2.0*np.pi) - 0.5*ld + np.log(gmm.weights_[c])

    # else:
    #     myLLcomp = -0.5*g - (d/2.0)*np.log(2.0*np.pi) - 0.5*ld
    return myLLcomp


def logdet_chol(L):
    """log(det(A)) where A = L*L^T with L lower triangular"""
    return 2.0*np.sum(np.log(np.diag(L)))

  

from scipy.linalg.lapack import get_lapack_funcs
# potrs, = get_lapack_funcs(('potrs',),(np.array([0.0]),))


def solve_chol(L, b):
    """inv(A)*b where A = L*L^T with L lower triangular"""
    potrs, = get_lapack_funcs(('potrs',),(L,))
    b, info = potrs(L, b, lower=True, overwrite_b=False)
    if info < 0:
        msg = "Argument %d to lapack's ?potrs() has an illegal value." % info
        raise TypeError(msg)
    if info > 0:
        msg = "Unknown error occured int ?potrs(): error code = %d" % info
        raise TypeError(msg)
    return b



def inv_chol(L):
    """inv(A) where A = L*L^T with L lower triangular"""
    invL = np.linalg.inv(L)
    return np.dot(invL.T, invL)


def logsumexp(x, axis=0):
    np.seterr(all='ignore')
    xShape = x.shape
    if len(xShape) == 0 or xShape[axis] == 0:
        return -np.inf
    alpha = np.amax(x, axis) + (2.0*np.log(x.shape[axis]) - __halflogmax)
    badx = np.logical_not(np.isfinite(x))
    xx = np.exp(x - alpha.reshape(xShape[0:axis] + (1,) + xShape[(axis+1):]))
    xx[badx] = 0
    sumexpx = np.sum(xx, axis)
    if np.isscalar(sumexpx):
        if sumexpx > 0:
            return alpha + np.log(sumexpx)
        else:
            return -np.inf
    else:
        ret = np.empty_like(sumexpx)
        ret[sumexpx > 0] = np.log(sumexpx[sumexpx>0])
        ret[sumexpx <= 0] = -np.inf 
        return alpha + ret



def my_weighted_kmeans(point, weights, init_cluster_center, ncluster, it_max):
    """ my_weighted_kmeans - perform weighted kmeans clustering on "data" with 
        respect to "weights" assigned on each data point.

        INPUT:
            data = d x n matrix
            weights = d x 1 vector
            init_cluster_center = d x ncluster matrix
            ncluster = number of clusters required
            it_max = maximum iteration

        OUTPUT:
            cluster_label = n x 1 vector, indicating the cluster of each data point
            cluster_center = d x ncluster matrix
    """

    it_num = 0
    cluster_num = ncluster
    cluster_center = init_cluster_center
    dim_num, point_num = np.shape(point)

    dist2cluster = np.zeros((cluster_num,point_num))

    # assign each data point to the nearest cluster
    for j in np.arange(cluster_num):
        curr_cluster_center = cluster_center[:,j].reshape(dim_num,1)
        dist2cluster[j,:] = np.sum((point-curr_cluster_center)**2,0)

    cluster = np.argmin(dist2cluster,0)


    # determine the cluster population and weights
    # compute the weighted cluster center
    cluster_center,cluster_population,cluster_weight = gcentroids(point,cluster,weights,cluster_num)


    # set the point energies
    #   - 1. compute the distance of each point to its assigned cluster center
    #   - 2. adjust the point energies by a weight factor
    f,cluster_energy = genergy(point,weights,cluster_center,cluster_weight,cluster)


    old_energy = np.sum(cluster_energy)
    energy_in_iteration = []
    energy_in_iteration.append(old_energy)


    while it_num < it_max:
    # for each cluster
    # find non members ~m
    # compute f(~m) to current cluster -> f'(~m)
    # save these energy f in a k x n matrix, with each element storing the energy of each point to corresponding cluster
    # find minimum energy for each point
    # assign all point to each minimum energy cluster
    # update cluster center, weight, population and energy
    #     print '\niteration = {0}\n'.format(it_num)

        fmat = np.zeros((cluster_num,point_num))
        for j in np.arange(cluster_num):
            members =  [i for i,x in enumerate(cluster) if x==j]
            nonmembers = [i for i,x in enumerate(cluster) if x!=j]
            fmat[j,members] = f[members].reshape(-1,)
            adjust_weight = cluster_weight[j]/(cluster_weight[j]+weights[nonmembers])
            f_nonmembers = np.sum((point[:,nonmembers]-cluster_center[:,j].reshape(-1,1))**2,0)
            fmat[j,nonmembers] = f_nonmembers

        cluster = np.argmin(fmat,0)

        cluster_center,cluster_population,cluster_weight = gcentroids(point, cluster, weights, cluster_num)
        f,cluster_energy = genergy(point,weights,cluster_center,cluster_weight,cluster)
        new_energy = np.sum(cluster_energy)
        if np.abs(new_energy-old_energy) < 1e-6:
            break
        else:
            old_energy = new_energy
            it_num = it_num + 1
            energy_in_iteration.append(new_energy)
        
    return cluster, cluster_center



def gcentroids(point, cluster, weight, cluster_num):
    ndim,npoint = np.shape(point)
    cluster_population = np.zeros((cluster_num,1))
    cluster_weight = np.zeros((cluster_num,1))
    centroids = np.zeros((ndim,cluster_num))

    for j in np.arange(cluster_num):
        members = [i for i,x in enumerate(cluster) if x==j]
        cluster_population[j] = len(members)
        cluster_weight[j]     = np.sum(weight[members])

        centroids[:,j]        = np.sum(point[:,members]*weight[members],1)

        if cluster_weight[j] > 0:
            centroids[:,j] = centroids[:,j]/cluster_weight[j] # cluster center
      

    return centroids, cluster_population, cluster_weight



def genergy(point,weight,cluster_center,cluster_weight,cluster):

    ndim,ndata = np.shape(point)
    cluster_num  = np.shape(cluster_center)[1]

    f= np.zeros((ndata,1))
    cluster_energy = np.zeros((cluster_num,1))

    for j in np.arange(cluster_num):
        members = [i for i,x in enumerate(cluster) if x==j]
        f[members] = np.sum((point[:,members]-cluster_center[:,j].reshape(ndim,1))**2,0).reshape(-1,1)
        cluster_energy[j] = np.sum(weight[members]*f[members])
        f[members] = f[members]*cluster_weight[j]/(cluster_weight[j]-weight[members]).reshape(-1,1)
    
    return f,cluster_energy


