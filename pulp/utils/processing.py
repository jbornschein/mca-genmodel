#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" First stage images processing class. 

Emulates different methods that model the stages in visual processing WHITHOUT 
strong feedback known so far.

"""

from __future__ import division

from scipy.signal import convolve2d
from sys import exit, argv, stdout
import random
from pulp.preproc.image import extract_patches_from_single_image, extract_patches_from_images
from scipy.signal import convolve2d
from mpi4py import MPI




try:
    from abc import ABCMeta, abstractmethod
except ImportError, e:
    from pulp.utils.py25_compatibility import _py25_ABCMeta as ABCMeta
    from pulp.utils.py25_compatibility import _py25_abstractmethod as abstractmethod


################################################################################
#First prototype driver, print statement still uncomplete.

import tables as tb
from mpi4py import MPI
import numpy as np
from pulp.utils.autotable import AutoTable



class Proc_Driver():

    def __init__(self, in_file, out_file, data_key="images", out_key="patches", comm=MPI.COMM_WORLD):

        self.comm = comm
        self.schedule = []
        self.out_key = out_key

        try: 
            input_f = tb.openFile(in_file)
        except: raise IOError("Could not open %s" % (in_file,))

        if not data_key in input_f.root.__members__ :
            raise KeyError("provided key: %s not in input file" % (data_key, ))

        self.data = input_f.getNode(input_f.root, name=data_key)  
        self.in_file = input_f

        if comm.rank == 0:
            try:
                self.out_file = AutoTable(out_file)
            except: raise IOError("Could not create output %s" % (out_file,))
                              


    def save(self):
        data = self.data[:]
        comm = self.comm

        recv_shape = (comm.size,) +  data.shape
        to_append = np.empty(recv_shape, data.dtype)
            
        comm.Gather(data, to_append)

        to_append = to_append.reshape( (-1,) + data.shape[1:] )
        
        if comm.rank == 0:
            out_h5 = self.out_file.h5
            out_h5.createArray(out_h5.root, self.out_key, to_append)


    def run(self):
        for proc_stage in self.schedule:
            self.data = proc_stage.work(self.data, self.comm)
            self.comm.barrier()

        self.save()


    def close(self):

        self.in_file.close()            

        if self.comm.rank == 0:
            self.out_file.close()
            

    def __str__(self):

        init_str = "\n\n The scheduled processes are: \n\n "

        if len(self.schedule) == 0: return init_str + 'None'
        else:
            for proc_stage in self.schedule:
                print proc_stage, '\n\n'            



#End Proc_Driver prot.
###############################################################################















class FSP(object):
 
    __metaclass__ = ABCMeta

    """ Base class for First Stage Processing. """

    def __init__(self, input_ref, sampling_dist, comm=MPI.COMM_WORLD):
        """Creates First Processing Object containing the data and sampling info.

        
        """

        if isinstance(input_ref, list): 
            if len(input_ref) != 2: raise ValueError("input_ref must contain 2 entries")
            try: h5 = tables.openFile(input_ref[0], 'r')
            except: "Could not open '%s'" % input_ref[0]
            try: input_data = h5.root.input_ref[1]
            except: "no child named '%s'" % input_ref[1]

        elif isinstance(input_ref, np.ndarray): 
            input_data = input_ref
            del input_ref
            h5 = None

        else: raise TypeError("Expects list or ndarray as argument") 

        my_N = input_data.shape[0]//comm.size
        my_ub = (comm.rank + 1) * my_N 
        my_lb = comm.rank * my_N

        #Everything flattened in the second dimension
        self.my_input = input_data[my_ub:my_lb, -1] 
        self.samp_dist = sampling_dist
        self.comm = comm
        self.h5 = h5

    @abstractmethod
    def sampling_method(self, ):
        """ """
        pass

    @abstractmethod
    def processing(self, ):
        """ """
        pass

    def close(self):
        """ """
        if self.h5 != None: 
            del self.my_input
            self.h5.close()
        else: del self.my_input        



class DoG(FSP):
    def sampling_method(self, ):
        pass
    def processing(self, ):
        pass

class PcaWhitening(FSP):
    def sampling_method(self, ):
        pass
    def processing(self, ):
        pass

class PseudoWhitening(FSP):
    def sampling_method(self, ):
        pass
    def processing(self, ):
        pass

class Raw(FSP):
    def sampling_method(self, ):
        pass
    def processing(self, ):
        pass


class ZeroPhaseWhitening(PcaWhitening):
    def sampling_method(self, ):
        pass
    def processing(self, ):
        pass

class DoGMeanFree(DoG):
    pass


def DoG(n_p, n_m, s_p, s_m):
    """ Produces DoG kernel.

    Returns a DoG kernel of size 6*s_p x 6*s_p, 
    centered at the DoG center where *s_p* is the broadest variance in a DoG 
    and *s_m* the thinnest. *n_p* is the amplitud of the broadest gaussian, 
    and *n_m* the amplitud of the thinnest.

    """
    gx, gy = np.ogrid[0:np.int(6*s_p), 0:np.int(6*s_p)]
    gx -= np.int(6.*(s_p/2.))
    gy -= np.int(6.*(s_p/2.))
    #Generate the exponents
    e1 = (1./(s_p**2))*gx*gx + (1./(s_p**2))*gy*gy
    e2 = (1./(s_m**2))*gx*gx + (1./(s_m**2))*gy*gy
    G = -n_p*np.exp(-0.5*e1) + n_m*np.exp(-0.5*e2)
    return G




#TODO check whether there is already something like this!
def estimate_needed_time(i):
    """Estimates remining time for job to complete"""

    ll = len(str(i))
    stdout.write('\b'*ll + str(i))
    stdout.flush()


def deconv(s_p, s_m, GF, fsize, mode='full'):
    """
    

    expects the DoG var plus *s_p*, and minus *s_m*, together with 
    *GF* to process of shape (*num_hidden*, size, size), and filter
    size *fsize*. Returns array of shape (num_hidden, size+fsize, size+fsize)           

    """
    num_hidden, size, _ = GF.shape 

    #Arrays we need
    if mode == 'full':
        ext_patches = np.empty([num_hidden, np.int(size+fsize) - 1, np.int(size+fsize) - 1])
    elif mode == 'same':
        ext_patches = np.empty([num_hidden, size, size])
    #tmp = np.empty([size, size])

    for ll in xrange(num_hidden):
        n_p = 0.3*np.max(GF[ll])
        n_m = 9*n_p
        ext_patches[ll, :, :] = convolve2d(DoG(n_p, n_m, s_p, s_m), GF[ll], mode)
    return ext_patches


##TODO, all this I/O functionality should be included in an object
def deconvolve(full_input_path, output_path='RF.h5'):

    try:
        input_file = openFile(full_input_path, 'r')
    except:
        print "Could not open '%s'" % full_input_path
        exit()

    #########################################################
    #opening the set with the result GF
    #Using the output format of the data logger!
    imset = input_file.root.W
    shape_tuple = imset.shape

    if len(shape_tuple) > 3:
        num_iter, num_hidden, D = shape_tuple
        size = np.int(np.sqrt(D//2))
        fullimset = imset[num_iter-1, ...].reshape(num_hidden, 2*size, size)
    #########################################################

    output_file = AutoTable(output_path)

    ## TODO: This cannot be set by hand, adjust it from the input file!!!!
    #We use this set of variances for the kernel:
    s_p = 3.
    s_m = 1.

    #Arrays we need
    ext_patches = np.empty([num_hidden, size, size])
    tmp = np.empty([size, size])

    print '\n Extracting and convolving:\n'
    for ll in xrange(num_hidden):

        temp = fullimset[ll][0:size, :] - fullimset[ll][size:2*size,:]
        n_p = 0.3*np.max(temp)
        n_m = 9*n_p
        output_file.append('results', convolve2d(DoG(n_p, n_m, s_p, s_m), temp))
        printG(str(ll)+ '/' + str(num_hidden))

    print '\n\n Ready!\n'

    input_file.close()
    output_file.close()



##############################################################################


#The three pairs of sigmas used (plus, minus) : 1.5:0.5, 2.1:0.7, 3.:1. Following 
# De Angelis et al 1995 (consult preproc on the wiki for more info)
## TODO include this references in the processing function!

def sat_func(data, per):
    """


    *data* is a 2d array, *per* is number between 0 and 1 indicating the 
    percentage of pixels to keep intact. Returns a two element tuple first, the 
    used cut off. And second 2d array that has a fraction *per* of the initial 
    lowest pixel values in *data* without modification. 

    """
    image_ = data[:, :]
    a, b = image_.shape
    #compute threshold
    thre = np.sort(image_.flatten())[np.int(per*a*b)-1]
    #we transform the image
    image_[image_>thre] = thre
    return (thre, image_)


#TODO : Construct the preprocessing as convolve, DOG object!


#Missing sigma matrix inside the exponent (np.sum(...))!!
def selected_hidden(y, W, log_pi_s, H, H_prime, sigma):

    '''y is a data point (shape = (D,)), W is 

	    the gen_field matrix (shape = (D, H))'''

    D = y.shape[0]
    data = y.reshape(D, 1)
    max_matrix = np.maximum(W, data)
    diff_matrix = max_matrix - data
    log_selection_vector = log_pi_s-0.5*np.sum(diff_matrix*diff_matrix, axis=0)
    return np.argsort(log_selection_vector, kind='mergesort')[-H_prime:]
######################################################################


###############################################################################


def whiten(data, epsilon=1e-9):
    """Whiten given *data*. 

    *data* is expected to be a (number_data, dim) shaped array according to the 
    whitening procedure described in Natural Image Statistics, Hy.. Hurry and
    Hoyer. *epsilon* specifies a lower bound for the eigenvalues of the 
    covariance matrix and is only used to warn the user if output dimensionality 
    is lower than the input dimensionality.

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
    """Performs PCA on given *data*.

    *data* is expected to be a (n_data, dim) shaped
    array according to the PCA procedure described in 'Pattern Recognition 
    and Machine Learning' Bishop.
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
     


class PseudoWhiteningFilter:
    """This class implements pseudo whitening (Olshausen & Field in 1996)."""
    def __init__(self):
        self.kernel = np.zeros(0)

    def _gen_kernel(self, shape):
        """Generate and return the convolution filter in freq. space"""
        h, w = shape
        h2 = h // 2
        w2 = w // 2
        f0 = int(0.8 * min(h2, w2))
        y,x = np.ogrid[-h2:h-h2 , -w2:w-w2 ]
        F   = np.sqrt(x*x+y*y) * np.exp( -(np.sqrt(x*x+y*y)/f0)**4 )
        return fft.fftshift(F)

    def filter(self, images):
        """Pseudo whiten patches.

        Given an array of image-patches *images*, pseudo whiten them and
        return a identically shaped array.

        """
        n, h, w = images.shape

        # Do we have a correctly shaped cached filter?
        if self.kernel.shape != (h, w):
            self.kernel = self._gen_kernel((h, w))

        # Allocate return array
        filtered = np.empty((n, h, w))

        # Apply filter in freq. space
        for i in xrange(n):
            I = images[i].astype(np.float)
            I_ = fft.fft2(I)
            J_ = self.kernel * I_
            J  = np.real(fft.ifft2(J_))
            print J.shape
            filtered[i] = J
        
        return filtered

