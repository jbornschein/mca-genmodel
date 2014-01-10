#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Gabor processing, testing and benchmarking class. 

Statistical tools to test a given gabor data set.
Classifies and fits gabor in data sets.
Produces Gabor test sets.

Why gabors and not Mickey Mouses? Visit the Group-Wiki. 
After all the project is called *gabor*, not Mickey Mouse...

"""

from __future__ import division

import numpy as np
from numpy.random import uniform, randn, multivariate_normal
from numpy.fft import fft2, ifft2, fftshift

from math import pi
from scipy.optimize import fmin_powell

from mpi4py import MPI

from pulp.utils.parallel import pprint
from pulp.utils.datalog import dlog

################################################################################
def gabor_kern(params, shape):
    """Produces a gabor kernel.  

    *shape*: Tuple: (*height*, *width*)
    *params*: array containing: (*x0*, *y0*, *sigma_x*, *sigma_y*,
    *psi*, *tau*, *k*, *A*). Where *psi* is the wave vector angle
    and *tau* the phase. Consult the group wiki for a detailed 
    description.

    Returns an array of shape *shape* containing the generated gabor.

    """

    # Extract parameters
    x0 = params[0:2]
    #Ppal axes sigma
    sigma_unr = np.array( [[1/params[2]**2, 0], [0, 1/params[3]**2]] )
    psi, tau = params[4:6]
    k0 = params[6]
    A = params[7]
    h, w = shape

    #Rotation matrix used
    R0 = np.array( [[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]] )

    #Rotated sigma
    sigma_r = np.dot( np.transpose(R0), np.dot(sigma_unr, R0) )
    k_r = np.dot( np.transpose(R0), np.array([k0, 0]) )

    # Generate Grid, shifted by x0
    gx, gy = np.ogrid[0:h, 0:w]
    gx -= x0[0]
    gy -= x0[1]
    # Generate the exponents and the companion cosine 
    e = sigma_r[1,1]*gx*gx + (sigma_r[0,1]+sigma_r[1,0])*gx*gy + sigma_r[0,0]*gy*gy

    angular = (k_r[1]*gx + k_r[0]*gy) + tau
    gabor_kern = (A*np.exp(-0.5*e))*(np.cos(angular))
    return gabor_kern


def gabor_benchmark(dist_params, gabor_shape, num_basis):
    """ Produce a set of randomly generated gabor functions, according 
    to the given distribution.

    *dist_params* is a list/vector/tuple of (pk0, center_vec, cov_params, 
    bor, A): wave vector's *module*, (2D) *center* of the gaussian 
    distribution to draw from (x and y coord resp), cov_params = (*sigma_x*, 
    *sigma_y*, *angle*) to construct sample covariance matrix from, *borders*
    form the sides, and *amplitud* to use. *gabor_shape* is a 2 element
    list/vector/tuple that specifies the shape of the output. *num_basis* is
    the number of random gabors to be generated. 
    Returns tuple with 1_: 2D of shape (*num_gabor*, *D*) with the random
    gabors flattened in the second dim. 2_: the generative parameters used.
    """

    #Unpack parameters to use:
    pk0, center, cov_params, bor, A = dist_params
    sigma_x, sigma_y, psi = cov_params
    h, w = gabor_shape


    #Preallocate return structures:
    gen_params = np.empty([num_basis, 8])
    gabor_set = np.empty([num_basis, h*w])

    #Assign the fixed parameters *pk0* and *A*.
    gen_params[:, 6] = pk0
    gen_params[:, 7] = A

    #A_We select the centers x0, x1 from a uniform random distribution
    gen_params[:, 0]  = uniform(bor, h - bor, num_basis)
    gen_params[:, 1]  = uniform(bor, w -  bor, num_basis)

    #B_Construct Covariance:
    sigma_unr = np.array( [[sigma_x**2, 0.], [0., sigma_y**2]] )
    R0 = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])#Rotation matrix
    cov = np.dot(R0, np.dot(sigma_unr, np.transpose(R0)))

    #C_We select sigma_x, sigma_y from a random distribution
    gen_params[:, 2:4] = multivariate_normal(center, cov, num_basis)[:, :]
  
    #D_We select wave vector angle (4) and phase inside cosine
    gen_params[:, 4] = uniform(-pi/2., pi/2., num_basis)
    gen_params[:, 5] = uniform(0., pi/2., num_basis)
    
    #Create the gabor set
    for i in xrange(num_basis):
        gabor_set[i, :] = gabor_kern(gen_params[i], gabor_shape).reshape(h*w)     

    return gabor_set, gen_params

###############################################################################


def objective_function(to_estimate, *args):
    """
    """
    gabor_shape, data = args
    return np.sum(np.abs((gabor_kern(to_estimate, gabor_shape) - data)))


def check_bounds(a, b, side):
    """
    """
    if np.sum(np.array([a, b]) <= np.float(side-1)) < 2:
        return False
    elif np.sum(np.array([a, b]) >= 0.) < 2:
        return False
    else:
        return True


def best_approx(a, b, side):
    """
    """
    if a < 0.:
        ret_a = 0
    elif a > (side-1):
        ret_a = side-1
    else:
        ret_a = a
    if b < 0.:
        ret_b = 0
    elif b > (side-1):
        ret_b = side-1
    else:
        ret_b = b
    return [ret_a, ret_b]



def gabor_fit(data):
    """
    """
    gabor_shape = data.shape

    #Assure that both dimensions are equal:
    h, w = gabor_shape
    if h != w:
        raise ValueError("Sqare input expected (width == height)")

    #The real borders of the picture are separated 
    #from the canvas by (this is where the real 
    #patch is, the rest is and artifact of the 
    #convolution:

    bor = 2
    margin = h - bor
    ######## Estimate initial cond.##########

    ########Estimator 1############
    #Position of the absolut maximum
    max_pos = np.argmax(np.abs(data))

    x0 = max_pos//h
    x1 = max_pos%h

    ###########Estimator2 ############
    #Wave-vector (Via FFT)
    mod_trans = np.abs(fft2(data))

    #We centered it according to the fftshift function
    mod_trans_s = fftshift(mod_trans)
    max_vec = np.argsort(mod_trans_s.flatten())[-2:]

    delta_ver = -0.5*(max_vec[0]//w -max_vec[1]//w)
    delta_hor = 0.5*(max_vec[0]%w -max_vec[1]%w)

    #Because of the naming conventions we have to do this sort of weird association
    k_x = (2*pi*delta_ver)/w
    k_y = (2*pi*delta_hor)/w


    #Also, because of the way in which we operate on the poles of the Fourie Transform,
    # we need to change the sign of k_x and k_y 
    if k_y < 0:
        k_x = -k_x
        k_y = -k_y

    k0 = np.sqrt(k_x**2 + k_y**2)

    if k_y != 0.:
        psi = np.arctan(k_x/k_y)
    else:
        psi = pi/2.

    ########### Estimator3 ############
    #Amplitud, phase

    V_1 = data[x0, x1]
    add_x0 = np.int(np.round((pi/2.)*(k_x/(k0**2)))) #(vertical shift)
    add_x1 = np.int(np.round(-(pi/2.)*(k_y/(k0**2)))) #(horizontal shift)

#########EP
    if not check_bounds(x0 + add_x0, x1 + add_x1, h):
        add_x0 = -add_x0
        add_x1 = -add_x1
        if not check_bounds(x0 + add_x0, x1 + add_x1, h):
        #If after the heuristics it's still broken, clamp it.
            a,b = best_approx(x0 + add_x0, x1 + add_x1, h)
            V_2 = data[a,b]
        else:
            V_2 = -data[x0 + add_x0, x1 + add_x1]
    else:
        V_2 = data[x0 + add_x0, x1 + add_x1]


    add2_x0 = -np.int(np.round(2*pi*(k_x/(k0**2))))
    add2_x1 = np.int(np.round(2*pi*(k_y/(k0**2))))

#########EP

    if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
        add2_x0 = -add2_x0
        add2_x1 = -add2_x1
        if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
            #If after the heuristics it's still broken, clamp it.
            a,b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
            V_3 = data[a, b]
        else:
            V_3 = data[x0 + add2_x0, x1 + add2_x1]
    else:
        V_3 = data[x0 + add2_x0, x1 + add2_x1]



    #We need to correct for small deviations in V_3, sign should be that of V1.
    if np.sign(V_3) != np.sign(V_1):
        V_3 = -V_3
     
    factor = pow((V_3/V_1), 1./16.)
    V_2 = V_2/factor

    #The amplitud is given by:
    A = np.sqrt(V_1**2 + V_2**2)

    #An the angle by: (although it's not giving an accurate estimate)
    #There might be a bug in this estimate.
    if V_1 != 0:
        tau = np.mod(np.arctan(V_2/V_1) - (k_x*x0 + k_y*x1) , 2*pi)
        if tau > pi:
            tau -= pi
    else: 
        tau = np.mod( pi/2. - (k_x*x0 + k_y*x1), 2*pi)
        if tau > pi:
            tau -= pi

    ########### Estimator4 ############
    #Sigma matrix
    V_1 = data[x0, x1]

    add2_x0 = -np.int(np.round(4*pi*(k_x/(k0**2))))
    add2_x1 = np.int(np.round(4*pi*(k_y/(k0**2))))
#########EP

    if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
        add2_x0 = np.round(0.5*add2_x0)
        add2_x1 = np.round(0.5*add2_x1)
        if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
            add2_x0 = -add2_x0
            add2_x1 = -add2_x1
            if not check_bounds(x0 + add2_x0, x1 + add2_x1, h):
                #If after the heuristics it's still broken, clamp it.
                a,b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
                V_2 = data[a, b]
            else:
                V_2 = data[x0 + add2_x0, x1 + add2_x1]
        else:
            V_2 = data[x0 + add2_x0, x1 + add2_x1]
    else:
        V_2 = data[x0 + add2_x0, x1 + add2_x1]

    if np.sign(V_2) != np.sign(V_1):
        V_2 = -V_2

    add3_x0 = np.round(6*(k_y))
    add3_x1 = np.round(6*k_x)
#########EP
    if not check_bounds(x0 + add3_x0, x1 + add3_x1, h):
        #If it's out of range, then it's not fittable
        a,b = best_approx(x0 + add2_x0, x1 + add2_x1, h)
        V_3 = data[a, b]
    else:
        V_3 = data[x0 + add3_x0, x1 + add3_x1]


    if np.sign(V_3) != np.sign(V_1):
        V_3 = -V_3

    sigma_x = (4*pi/k0)*np.sqrt(1./(-2.*np.log(V_2/V_1)))
    sigma_y = np.sqrt(-18./np.log(V_3/V_1))

    #Put the estimators together in params:
    params = np.empty(8)
    params[0] = x0             # x-position
    params[1] = x1             # y-position
    params[2] = sigma_x        # 
    params[3] = sigma_y        # 
    params[4] = psi            # Wave-vector direction
    params[5] = tau            # Phase shift
    params[6] = k0             # Wave vector length
    params[7] = A              # Amplitude

    ######### Minimizing squared error. Gradient Method.##########
    num_iter_2 = 5

    #Just to be sure that we're on the right valey, we iterate over the two loops below
    min_params = np.empty(8)
    min_val = np.inf
    for j in xrange(-3, 4):
        for k in xrange(-3, 4):
            for i in xrange(num_iter_2):
                params_ = params.copy()
                params_[0] = params[0] + sigma_x/2*j
                params_[1] = params[1] + sigma_y/2*k
                params_[5] = 2*pi/num_iter_2*i

                #here we store the parameters found
                try:
                    cur_params = fmin_powell(objective_function, params_, (gabor_shape, data), disp=0)
                    cur_val = objective_function(cur_params, gabor_shape, data)

                    if cur_val < min_val:
                        min_params = cur_params
                        min_val = cur_val
                except Exception as e:
                    print e

    min_params[5] = np.mod(min_params[5], 2*pi) 
    return min_params


def fit_conv(params, error_vector, image_size):
    """
    Filter results so that only "sensible" gabor-prarams survive.
    """
    para = params.copy()
    error_vec = error_vector.copy()
    #Discard the miningless sigmas and wave vectors (hardcoded for now!)
    bound = (image_size)**2
    for i in (2, 3, 6):
        para[:, i] = np.abs(para[:, i])
        slicer = para[:, i] < bound
        para = para[slicer, :]
        error_vec = error_vector[slicer]

    #Shift angles and get frecuency from wave vector
    para[:, 4] = np.mod((para[:, 4] + pi/2.), pi)
    para[:, 5] = np.mod(para[:, 5] , 2*pi)
    para[:, 6] = (para[:, 6])/(2*pi)
    return (para, error_vec)    

def pgabor_fit(RFs, comm=MPI.COMM_WORLD):
    """
    Parralel gabor-fit the supplied receptive fields.

       *RFs* is expected to be of shape (num_rfs, rf_size, rf_size) and be the same 
       on all involved MPI ranks.
    """
    num_RF, gabor_size, _ = RFs.shape
    gabor_shape = (gabor_size, gabor_size)

    my_errors = np.zeros(num_RF)        # Error per datapoint 
    my_params = np.zeros([num_RF, 8])   # 8 gabor params per datapoint
    
    # Iterate over all RFs with comm.size stride. 
    for i in xrange(comm.rank, num_RF, comm.size):
        dlog.progress("Gabor fitting %d of %d" % (i, num_RF), i/num_RF)
        this_RF = RFs[i]

        this_params = gabor_fit(this_RF)
        this_error = np.sum(np.abs(this_RF-gabor_kern(this_params, gabor_shape)))

        my_params[i, :] = this_params
        my_errors[i] = this_error

    # Aggregate results
    params = np.empty_like(my_params)
    errors = np.empty_like(my_errors)

    comm.Allreduce([my_params, MPI.DOUBLE], [params, MPI.DOUBLE])
    comm.Allreduce([my_errors, MPI.DOUBLE], [errors, MPI.DOUBLE])
    # Filter results that do not seem useful
    #params, errors = fit_conv(params, errors, gabor_shape[0])

    return params, errors

