#
#  Author:   Jorg Bornschein <bornschein@fias.uni-frankfurt.de)
#  Lincense: GPL V3.0
#

"""
    The pulp.utils packages provides various utility functions and classes.
"""
import time as tm
import numpy as np
import errno
import sys
import os

from mpi4py import MPI

def create_output_path(basename=None):
    """ Creates the output directory making sure you don't overwrite a folder. 

    If you are running under the torque/pbs job scheduler, the directory will
    be named according to <BASENAME>.d<JOBID>; if you are not, the directory
    will be named <BASENAME>.<DATE+TIME>. If such a directory should already
    exist, an additional "+NUMBER" suffix will be appended.
    
    If you do not specify a *basename*, it will derive the base-name of the 
    directory from your programs name (sys.argv[0]).
    """
    comm = MPI.COMM_WORLD

    if comm.rank == 0:
        if basename is None:
            basename = sys.argv[0]

        # Determine suffix
        if 'PBS_JOBID' in os.environ:
            job_no = os.environ['PBS_JOBID'].split('.')[0]   # Job Number
            suffix = "d"+job_no
        else:
            suffix = tm.strftime("%Y-%m-%d+%H:%M")
            
        suffix_counter = 0
        dirname = "output/%s.%s" % (basename, suffix)
        while True:
            try:
                os.makedirs(dirname)
            except OSError, e:
                if e.errno != errno.EEXIST:
                    raise e
                suffix_counter += 1
                dirname = "output/%s.%s+%d" % (basename, suffix, suffix_counter)
            else:
                break
    else:
        dirname = None
        
    return comm.bcast(dirname)+"/"

def iter_to_str(iteration, maximum):
    """ Converts an iteration number to string.

    Uses the maximum as second input to guarantee equal length for all.

    """
    cur_trial_len = len(np.str(iteration))
    return ((len(np.str(np.int(maximum)+1))-cur_trial_len) * '0') + np.str(iteration)

def check_basis(test_basis, real_basis):
    """Check if *real_basis* is contained in *test_basis*.
  
    Return the MAE (mean average error; see Jörg's ET paper) and the 
    corresponding permutation for test_basis.

    """
    Ht, Dt = test_basis.shape
    Hr, Dr = real_basis.shape
        
    assert Dt == Dr

    # Calculate error matrix
    error = np.zeros( (Hr, Ht) )
    for hr in xrange(Hr):
        for ht in xrange(Ht):
            error[hr, ht] = np.sum(np.abs(real_basis[hr]-test_basis[ht])) / Dt / Hr
    
    # Allocate tables for dynamic programming
    mae_tab = np.zeros( (Hr, Ht) )
    used_tab = np.empty( (Hr, Ht), dtype=np.object)

    # Initialize first row
    for ht in xrange(Ht):
        tmprow = error[0, :].copy()
        tmprow[ht] = np.inf

        minpos = np.argmin(tmprow)
        mae_tab[0, ht] = error[0, minpos]
        used_tab[0, ht] = [minpos]
        
    # Build table
    for hr in xrange(1, Hr-1):
        for ht in xrange(Ht):

            # Build Matrix
            tmpmtx = np.zeros((Hr, Ht))
            for h0 in xrange(Hr):
                for h1 in xrange(Ht):
                    val = mae_tab[hr-1, h1] + error[hr, h0]
                    if h0 in used_tab[hr-1, h1]:
                        val = np.inf
                    if ht in used_tab[hr-1, h1]:
                        val = np.inf
                    if ht == h0:
                        val = np.inf
                    tmpmtx[h0,h1] = val

            minpos_error = np.argmin(tmpmtx) // Hr
            minpos_prev = np.argmin(tmpmtx) % Hr

            mae_tab[hr, ht] = tmpmtx[minpos_error, minpos_prev]
            used_tab[hr, ht] = used_tab[hr-1, minpos_prev] + [minpos_error]

    # Last row
    for ht in xrange(Ht):
        mae_tab[-1,ht] = mae_tab[-2,ht] + error[-1,ht]
        used_tab[-1,ht] = used_tab[-2,ht] + [ht]

    minpos = np.argmin(mae_tab[-1,:])
    return mae_tab[-1, minpos], np.array(used_tab[-1, minpos])


def compute_MAE_prob(model, anneal, mparams, gparams, MAE_threshold):

    """ Checks Generative Fields MAE-distance between *mparams* (learned) and 
    *gparams* (grouns-truth).

    Returns a tuple (MAE, map) according to the steps ( performed in 
    order, MAE: mean average error, map: function between basis and hidden):


    0_ Compute the responsabilities for the basis functions in mparams.

    i_ Take the arg-maximum of the responsabilities base-wise. (If it is Bimodal, 
    the MAE will take care of it.)

    ii_ If the maximum are not ammong arange(1, H+1) (i.e.one single active 
    cause) then return the Meaningless case : (-1, np.arange(num_basis))

    iii_ Given ii_ , check that the peaks for different data points are at 
    different positions. If not, return the Meaningless case : (-1, np.arange(num_basis))

    iv_ given ii_, iii_ are true, compute the MAE. If MAE < threshold, then 
    return it together with the map, otherwise return the Meaningless   
    case: (-1, np.arange(num_basis))

    """

    #Model magnitudes
    H = model.H
    D = model.D

    #Get the bars to use as input data.
    W_gt = gparams['W']
    num_basis, _ = W_gt.shape
    data = model.select_Hprimes(mparams, {'y':W_gt})

    #Perform e-step (and get the respons-out)
    respons = model.calculate_respons(anneal, mparams, data)
    max_respons = np.max(respons, axis=1)

    #Check that this maximum corresponds to the 1 active cause case
    if not ( ( 0 < max_respons ).all() and ( max_respons < H + 1).all()):
        return (-1, np.arange(num_basis))

    #Inyectivity: Check that they're all different
    max_places = np.argmax(respons, axis=1)
    if not (np.unique(max_places).shape[0] == num_basis):
        return (-1, np.arange(num_basis))

    #Compute the MAE and map
    max_places -= 1 #Causes are numbered 0... H-1

    permutation = np.argsort(max_places)
    used_GF_permuted = np.sort(max_places)

    MAE = np.sum(np.abs(mparams['W'][used_GF_permuted, :] - W_gt[permutation, :]))/num_basis/D

    if MAE > MAE_threshold: return (-1, np.arange(num_basis))
    else:
        if num_basis == H: return (MAE, permutation)
        else: return (MAE, max_places)




def binom(n, m):
    """
    # code from http://www.brpreiss.com/books/opus7/programs/pgm14_10.txt

    XXX There is scipy.comb which does excatly the same! XXX
    """
    b = [0] * (n + 1)
    b[0] = 1
    for i in xrange(1, n + 1):
        b[i] = 1
        j = i - 1
        while j > 0:
            b[j] += b[j - 1]
            j -= 1
    return b[m]

#=========================================================================
# Alternative Implementation
#def check_basis(test_basis, real_basis):
#    """
#       Check if the basis *test_basis* contains all the basis vectors
#       given in *real_basis* and return the MAE (mean average error;
#       see Jörg's ET paper) and the corresponding permutation for
#       test_basis.
#    """
#    Ht, Dt = test_basis.shape
#    Hr, Dr = real_basis.shape
#        
#    assert Dt == Dr
#
#    # First calculate error matrix
#    error = np.zeros( (Hr, Ht) )
#    for hr in xrange(Hr):
#        for ht in xrange(Ht):
#            error[hr, ht] = np.sum(np.abs(real_basis[hr]-test_basis[ht])) / Dt / Hr
#    
#    # Allocate tables for dynamic programming
#    mae_tab = np.zeros( (Hr, Ht) )
#    used_tab = np.empty( (Hr, Ht), dtype=np.object)
#
#    # Initialize first row
#    for ht in xrange(Ht):
#        mae_tab[0, ht] = error[0, ht]
#        used_tab[0, ht] = [ht]
#        
#    #print mae_tab
#    #print used_tab
#
#    # Build table
#    for hr in xrange(1, Hr):
#        for ht in xrange(Ht):
#            srcrow = mae_tab[hr-1, :].copy()
#
#            # Throw out solutins with double-assignments
#            for h in xrange(Ht):
#                if ht in used_tab[hr-1, h]:
#                    srcrow[h] = np.inf
#
#            minpos = np.argmin(srcrow)
#            mae_tab[hr, ht] = srcrow[minpos] + error[hr, ht]
#            used_tab[hr, ht] = used_tab[hr-1, minpos] + [ht]
#
#    #print mae_tab
#    #print used_tab
#
#    minpos = np.argmin(mae_tab[-1,:])
#    return mae_tab[-1, minpos], used_tab[-1, minpos]
#

class MaxMeanQueue():
    """ Queue that only keeps a certain number of elements in stack
    and can return the current mean value of all elements in stack.
    """
    def __init__(self, num_element):
        """ num_element expects the maximum number of elements in the queue
        """
        self._queue = []
        self._num_element = num_element

    def push(self, element):
        num_element = self._num_element
        if len(self._queue) >= num_element:
            self._queue.__delitem__(0)
        self._queue.append(element)
    
    def pop(self):
        popped = self._queue.pop(0)
        return popped
    
    def return_mean(self):
        """ returns np.array with last queue value and mean and variance of first n-1 queue elements
        """
        if len(self._queue) == 1:
            return np.array([self._queue[0], self._queue[0], 0])
        else:
            return np.array([self._queue[-1], np.mean(self._queue[0:-1]), np.std(self._queue[0:-1])])

    def numel(self):
        return len(self._queue)
    