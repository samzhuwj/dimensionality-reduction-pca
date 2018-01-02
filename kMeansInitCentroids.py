import numpy as np


def kmeans_init_centroids(X, K):
    """
    initialization for K-means centroids
    """    
    # You should return this value correctly
    centroids = np.zeros((K, X.shape[1]))

    # ===================== Your Code Here =====================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X
    indices = np.random.randint(X.shape[0], size=K)
    centroids = X[indices]

    # ==========================================================

    return centroids
