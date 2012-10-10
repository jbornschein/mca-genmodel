#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: Academic Free License (AFL) v3.0
#

"""

Various preprocessing techniques
================================
"""
import numpy as np


def whiten(data, epsilon=1e-9):
    """
    Whiten *data* which is expected to be a (n_data, dim) shaped
    array according to the whitening procedure described in Natural Image
    Statistics, Hy.. Hurry and Hoyer.

    *epsilon* specifies a lower bound for the eigenvalues of the covariance
    matrix and is only used to warn the user if output dimensionality is lower 
    than the input dimensionality.
    """

    N, dim = data.shape
    data_mean = data.mean(axis=0)
    
    # Calculate covariance matrix
    cov_mat = np.zeros((dim, dim))
    for i in xrange(N):
        cov_mat += np.outer( data[i]-data_mean, data[i]-data_mean )
    cov_mat /= N

    # Eigenvalue decomposition
    eig_val, eig_vec = np.linalg.eig(cov_mat)
    
    # Check if all eigenval represent useful dimensions
    if (np.abs(eig_val) < epsilon).any():
        raise ArithmeticError("Dimensionality of input data too low")
    
    # Construct transformation matrix
    D = np.zeros([dim, dim], 'float64')
    for i in xrange(dim):
        D[i, i] = 1./np.sqrt(eig_val[i])
    trans_mat = np.dot(np.dot(eig_vec, D), np.transpose(eig_vec))

    # And finally transform data
    return np.dot(data, trans_mat)



def pca(data, n=0, epsilon=1e-3):
    """
    Performs PCA on *data* which is expected to be a (n_data, dim) shaped
    array according to the PCA procedure described in 'Pattern Recognition 
    and Machine Learning ' Bishop.

    *n* is a positive number that specifies the number of dimensions to keep.
    if n is not given, then *epsilon* specifies a lower bound for the eigenvalues 
    of the covariance matrix and is used to reduce the dimensionality of the 
    input data.
    """

    N, dim = data.shape
    data_mean = data.mean(axis=0)
    
    # Calculate covariance matrix
    cov_mat = np.zeros((dim, dim))
    for i in xrange(N):
        cov_mat += np.outer(data[i]-data_mean, data[i]-data_mean)
    cov_mat /= N


    # Eigenvalue decomposition
    eig_val, eig_vec = np.linalg.eig(cov_mat)

   
    # Construct transformation matrices
    if n==0:
        i0 = np.argmin((eig_val > epsilon).astype(int))
        if eig_val[0] > epsilon and i0 == 0:
            i0 = dim
    else:
        i0 = n
    A = np.zeros([dim, dim], 'float64')
    B = np.zeros([dim, dim], 'float64')
    for i in xrange(i0):
        A += np.outer(eig_vec[:, i], eig_vec[:, i])
    for i in list(np.arange(i0, dim)):
        B += np.outer(eig_vec[:, i], eig_vec[:, i])

    # And finally transform data
    return np.dot(data, A) + np.dot(B, np.mean(data, axis=0))
     



