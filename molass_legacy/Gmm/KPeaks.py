# coding: utf-8
"""
    KPeaks.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np

def kmeans(k, X, max_iter=300):
    X_size,n_features = X.shape
    centroids  = X[np.random.choice(X_size,k)]
    new_centroids = np.zeros((k, n_features))
    cluster = np.zeros(X_size)

    for epoch in range(max_iter):
        for i in range(X_size):
            distances = np.sum((centroids - X[i]) ** 2, axis=1)
            cluster[i] = np.argsort(distances)[0]
        for j in range(k):
            new_centroids[j] = X[cluster==j].mean(axis=0)
        if np.sum(new_centroids == centroids) == k:
            print("break")
            break
        centroids =  new_centroids
    return cluster

class KPeaks:
    def __init__(self, k, X, max_iter=300):
        nrows, ncols = X.shape
        centroids  = X[np.random.choice(X_size,k)]
