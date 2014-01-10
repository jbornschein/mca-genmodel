#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" DoG processing

Statistical tools to test a given DoG data set.
Classifies and fits DoGs in data sets.

"""

from __future__ import division

import numpy as np
from numpy.random import uniform, randn, multivariate_normal

from math import pi
from scipy import inf
from scipy.optimize import fmin_powell

from mpi4py import MPI

from pulp.utils.parallel import pprint
from pulp.utils.datalog import dlog

from gabor_fitting import gabor_kern

################################################################################
def gauss(x, mu, sigma):
    pre = np.sqrt(2*pi) * sigma
    pre = 1./ pre
    expo = -0.5 * ((x-mu)/sigma)**2
    this_gauss = pre * np.exp(expo)
    return this_gauss


def dog_kern(params, shape):
    """Produces a DoG kernel.  

    *shape*: Tuple: (*height*, *width*) is the shape of the returned DoG
    *params*: array containing: (*mu_x*, *mu_y*, *sigma_x*, *sigma_y*,
    *sigmay_R*, *gamma*, *A1*, *A2*).
    gamma is the stretching factor of sigma matrix 2, A1 and A2 are the amplitudes
    \vec{x} = (x,y), \vec{mu} = (mu_x, mu_y).
    """

    # Extract parameters
    mu_x, mu_y = params[0:2]
    sigma_x, sigma_y, sigma_R = params[2:5]
    gamma = params[5]
    A1, A2 = params[6:8]

    # Prefactors
    pre1 = np.float(A1)/ (2 * pi * np.sqrt(sigma_x * sigma_y - sigma_R))
    pre2 = np.float(A2)/ (2 * pi * gamma * np.sqrt(sigma_x * sigma_y - sigma_R))

    # Denominator
    denom = sigma_x * sigma_y - sigma_R**2

    # Generate Grid
    h, w = shape
    gx, gy = np.ogrid[0:h, 0:w]

    # Generate the exponents
    sum1 = ((gx - mu_x) * (gy - mu_y) * sigma_R)/ denom
    sum2 = ((gx - mu_x)**2 * sigma_y)/ (2*denom)
    sum3 = ((gy - mu_y)**2 * sigma_x)/ (2*denom)
    exp1 = np.exp(sum1 - sum2 - sum3)
    sum1 /= gamma
    sum2 /= gamma
    sum3 /= gamma
    exp2 = np.exp(sum1 - sum2 - sum3)

    # Generate patch
    dog_kern = pre1 * exp1 - pre2 * exp2
    return dog_kern


def objective_function(to_estimate, *args):
    """
    """
    dog_shape, data = args
    return np.sum(np.abs(dog_kern(to_estimate, dog_shape) - data))


def dog_fit(data):
    """
    Actual fitting of the parameters
    """
    dog_shape = data.shape

    #Assure that both dimensions are equal:
    h, w = dog_shape
    if h != w:
        raise ValueError("Square input expected (width == height)")

    ######## Estimate initial cond.##########

    max_pos = np.argmax(np.abs(data))
    mu_x = max_pos//h #9
    mu_y = max_pos%h #18

    ######## Estimator2 ############
    # Sigma matrix
    sigma_R = 1.
    sigma_x = sigma_y = h/5
    gamma = 4.

    ######## Estimator2 ############
    # Amplitudes
    no_amplitude = gauss(0., 0., sigma_x) # Gauss value as if there were no amplitude defined
    A1 = np.max(data)/ no_amplitude
    no_amplitude = gauss(0, mu_x, gamma*sigma_x)
    A2 = np.abs(np.min(data)) / no_amplitude

    #Put the estimators together in params:
    params = np.empty(8)
    params[0] = mu_x
    params[1] = mu_y
    params[2] = sigma_x
    params[3] = sigma_y
    params[4] = sigma_R
    params[5] = gamma
    params[6] = A1
    params[7] = A2

    ########## Minimizing squared error. Gradient Method.##########
    # Go through different sigmas and gammas:
    cur_params = np.empty(8)
    cur_min_val = np.inf
    sigmas_x = [sigma_x/2, sigma_x, 2*sigma_x, 4*sigma_x]
    sigmas_y = [sigma_y/2, sigma_y, 2*sigma_y, 4*sigma_x]
    gammas = [0.05, 0.2, 0.5, 0.75]
    gammas.extend(range(1,10,2))
    for ind_sig_x in sigmas_x:
        for ind_sig_y in sigmas_y:
            for ind_gamma in gammas:
                params[2] = ind_sig_x
                params[3] = ind_sig_y
                params[5] = ind_gamma

                try:
                    min_params = fmin_powell(objective_function, params, (dog_shape, data), disp=0)
                    min_val = objective_function(min_params, dog_shape, data)

                    if min_val <= cur_min_val:
                        cur_min_val = min_val
                        cur_params = min_params
                except Exception as e:
                    print e

    # Calculate actual sigma_x_a and sigma_y_a:
    sigma_x = cur_params[2]
    sigma_y = cur_params[3]
    sigma_R = cur_params[4]
    sigma_mat = np.array([[sigma_x, sigma_R],[sigma_R, sigma_y]])
    sigmas = np.linalg.eigvals(sigma_mat)   
    
    return [cur_params, cur_min_val, sigmas]


def pdog_fit(RFs, comm=MPI.COMM_WORLD):
    """
    Fit DoGs to the supplied receptive fields.
    *RFs* is expected to be of shape (num_rfs, rf_size, rf_size)
    Returns the minimizing parameters, the min square error, and the actual (rotated) sigma values
    """
    num_RF, gabor_size, _ = RFs.shape
    gabor_shape = (gabor_size, gabor_size)

    my_errors = np.zeros(num_RF)        # Error per datapoint 
    my_params = np.zeros([num_RF, 8])   # 8 gabor params per datapoint
    my_sigmas = np.zeros([num_RF, 2])

    # Iterate over all RFs with comm.size stride. 
    #for i in xrange(comm.rank, comm.size, comm.size):
    for i in xrange(comm.rank, num_RF, comm.size):
        dlog.progress("DoG fitting %d of %d" % (i, num_RF), i/num_RF)
        this_RF = RFs[i]

        these_params, this_error, this_sigma = dog_fit(this_RF)
        this_error = np.sum(np.abs(this_RF-dog_kern(these_params, this_RF.shape)))

        my_params[i, :] = these_params
        my_errors[i] = this_error
        my_sigmas[i, :] = this_sigma

    # Aggregate results
    params = np.empty_like(my_params)
    errors = np.empty_like(my_errors)
    sigmas = np.empty_like(my_sigmas)

    comm.Allreduce([my_params, MPI.DOUBLE], [params, MPI.DOUBLE])
    comm.Allreduce([my_errors, MPI.DOUBLE], [errors, MPI.DOUBLE])
    comm.Allreduce([my_sigmas, MPI.DOUBLE], [sigmas, MPI.DOUBLE])

    return [params, errors, sigmas]


#############################################################################
# Globular finder...

def find_globulars(RF, gabor_params, gabor_errors, dog_params, dog_errors, max_aspect_ratio=3.0):
        
    H, _, _  = RF.shape
    rf_shape = RF.shape[1:]

    # extract Gabor parameters
    gabor_x0      = gabor_params[:,0]
    gabor_x1      = gabor_params[:,1]
    gabor_sigma_x = gabor_params[:,2]
    gabor_sigma_y = gabor_params[:,3]
    gabor_psi     = gabor_params[:,4]
    gabor_tau     = gabor_params[:,5]
    gabor_k0      = gabor_params[:,6]
    gabor_A       = gabor_params[:,7]

    # Extract DoG parameters
    dog_mu_x    = dog_params[:,0]
    dog_mu_y    = dog_params[:,1]
    dog_sigma_x = dog_params[:,2]
    dog_sigma_y = dog_params[:,3]
    dog_sigma_R = dog_params[:,4]
    dog_gamma   = dog_params[:,5]
    dog_A1      = dog_params[:,6]
    dog_A2      = dog_params[:,7]
    
    #########################################################################
    # Sqared Error
    #gabor_errors2 = np.zeros(H)
    #dog_errors2   = np.zeros(H)

    #for h in xrange(H):
    #    gabor_recons = gabor_kern(gabor_params[h], rf_shape)
    #    gabor_errors2[h] = np.sum( (RF[h]-gabor_recons)**2 )
    #
    #    dog_recons = dog_kern(dog_params[h], rf_shape)
    #    dog_errors2[h] = np.sum( (RF[h]-dog_recons)**2 )
    #dogs2 = dog_errors2 < gabor_errors2

    dog_sigma_w = np.zeros(H)
    dog_sigma_h = np.zeros(H)
    for h in xrange(H):
        sigma_mat = np.array(
                        [[dog_sigma_x[h], dog_sigma_R[h]],
                         [dog_sigma_R[h], dog_sigma_y[h]]]
        )
        width, height = np.linalg.eigvals(sigma_mat)   
        if width > height:
            dog_sigma_w[h], dog_sigma_h[h] = width, height
        else:
            dog_sigma_w[h], dog_sigma_h[h] = height, width
    dog_aspect_ratio = dog_sigma_w / dog_sigma_h

    dogs  = (dog_errors  < gabor_errors) * (dog_aspect_ratio < max_aspect_ratio)



    #print "Average Error: %3d DoG's => %4.1f %%" % (dogs.sum(),  100*dogs.sum()/H)
    #print "Squared Error: %3d DoG's => %4.1f %%" % (dogs2.sum(), 100*dogs2.sum()/H)
    #print "Where: ", np.where(dogs)
    #print "Excentricity: ", dog_aspect_ratio[dogs]

    return dogs
