"""
    Models.Stochastic.LognormalUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import root

def compute_mode(mu, sigma):
    return np.exp(mu - sigma**2)

def compute_stdev(mu, sigma):
    return np.sqrt((np.exp(sigma**2) - 1)*np.exp(2*mu + sigma**2))

def compute_mu_sigma(mode, stdev):
    def fun(x):
        mu, var = x
        mode_ = np.exp(mu - var)
        stdev_ = np.sqrt((np.exp(var) - 1)*np.exp(2*mu + var))
        return np.array([mode_ - mode, stdev_ - stdev])

    """
    See https://en.wikipedia.org/wiki/Log-normal_distribution
    We can solve mu, sigma from Mean and Variance,
    while solving from Mode and Variance seems difficult or impossible.
    (sympy cannot give a simple solution)
    """
    M = mode**2     # using mode while M = mean**2 is accurate
    mu_approx = np.log(M/np.sqrt(M + stdev**2))
    V = stdev**2
    var_approx = np.log(1 + V/M)
    res = root(fun, np.array([mu_approx, var_approx]))  # solving variance instead of sigma to avoid negative results.
    mu, var = res.x
    return mu, np.sqrt(var)

def compute_mu_sigma_from_mean(mean, stdev):
    # see https://en.wikipedia.org/wiki/Log-normal_distribution
    # the followins was genarated by Cody AI
    return np.log(mean**2/np.sqrt(mean**2 + stdev**2)), np.sqrt(np.log(1 + stdev**2/mean**2))

def compute_mean_from_mode(mode, stdev):
    mu, sigma = compute_mu_sigma(mode, stdev)
    return np.exp(mu + sigma**2/2)

if __name__ == "__main__":
    mode = compute_mode(6, 0.05)
    stdev = compute_stdev(6, 0.05)
    print("log(mode)=", np.log(mode))
    print("log(stdev)=", np.log(stdev))
    print(mode, stdev, compute_mu_sigma(mode, stdev))
    print(mode, stdev, compute_mean_from_mode(mode, stdev), np.exp(6 + 0.05**2/2))
    mean = compute_mean_from_mode(mode, stdev)
    mu, sigma = compute_mu_sigma_from_mean(mean, stdev)
    print("mean, mu, sigm", mean, mu, sigma)
    for mode in [400, 2.20742076e+02]:
        print("----------------")
        stdev = mode*0.1
        mean = compute_mean_from_mode(mode, stdev)
        print("mode, stdev, mean", mode, stdev, mean)


