import numpy as np
import scipy


def pca(X):
    """
    perform principal component analysis
    """    
    # Useful values
    (m, n) = X.shape

    # You need to return the following variables correctly.
    U = np.zeros(n)
    S = np.zeros(n)

    # ===================== Your Code Here =====================
    # Instructions: You should first compute the covariance matrix. Then, you
    #               should use the 'scipy.linalg.svd' function to compute the eigenvectors
    #               and eigenvalues of the covariance matrix.
    #
    # Note: When computing the covariance matrix, remember to divide by m (the
    #       number of examples).
    #

    sigma = np.dot(X.T, X) / m
    U, S, _ = scipy.linalg.svd(sigma, full_matrices=True, compute_uv=True)

    # ==========================================================

    return U, S
